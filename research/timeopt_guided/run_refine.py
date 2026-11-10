# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run idea C (``joint_refine``) from idea B's best lane on a planner run (CUDA env)."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .common import Payload, read_json, write_json


def main() -> None:
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg

    from .collision_screen import PathCollisionChecker
    from .geometry import native_cubic_to_clamped
    from .joint_refine import JointRefiner, RefineConfig
    from .k107_oracle import K107Oracle
    from .run_campaign import export, physical_limits

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--tag", default="hybrid_R1")
    ap.add_argument("--per-role", type=int, default=20)
    ap.add_argument("--config", type=str, default="{}")
    ap.add_argument("--out", default="refine")
    ap.add_argument("--all", action="store_true", help="refine every cell with a K107 lane (no per-role sample)")
    ap.add_argument("--input-tag", default="B_R1", help="method record used as C's input / fallback")
    ap.add_argument("--method-tag", default=None, help="write a complete method record per cell")
    a = ap.parse_args()
    cfg = RefineConfig(**json.loads(a.config))
    bank = {t["task_id"]: t for t in read_json(a.bank / "tasks.json")["tasks"]}
    kp = KinematicsCfg.from_robot_yaml_file("franka.yml").kinematics_config
    lim = physical_limits(kp)
    rows = [r for r in read_json(a.run / "lane_retime.json") if "best_lane" in r]
    rng = random.Random(0)
    pick = []
    if a.all:
        pick = rows
    else:
        for role in sorted({r["role"] for r in rows}):
            rr = [r for r in rows if r["role"] == role]
            pick += rng.sample(rr, min(a.per_role, len(rr)))

    def method_record(d: Path, extra_wall: float, traj: dict | None, route: str) -> None:
        if not a.method_tag:
            return
        base = read_json(d / f"{a.input_tag}.json")
        out = dict(base) | {"wall_s": base["wall_s"] + extra_wall, "route": route, "variant": a.method_tag}
        if traj is not None:
            out["duration_s"] = float(traj["time"][-1])
            np.savez_compressed(d / f"{a.method_tag}_traj.npz", **{k: traj[k] for k in ("time", "q", "qd", "qdd", "jerk")})
        elif (d / f"{a.input_tag}_traj.npz").exists():
            np.savez_compressed(d / f"{a.method_tag}_traj.npz", **dict(np.load(d / f"{a.input_tag}_traj.npz")))
        write_json(d / f"{a.method_tag}.json", out)

    if a.method_tag:  # cells C cannot start from (no K107 lane / planner failure): copy the input method
        have = {(r["task_id"], r["payload"]) for r in pick}
        for f in sorted(a.run.glob(f"cells/*/*/{a.input_tag}.json")):
            if (f.parent.parent.name, f.parent.name) not in have:
                method_record(f.parent, 0.0, None, read_json(f).get("route", "") + "|no_refine")
    oracles, checkers, out = {}, {}, []
    for r in pick:
        d = a.run / "cells" / r["task_id"] / r["payload"]
        rec = read_json(d / f"{a.tag}.json")
        t = bank[r["task_id"]]
        p = Payload.from_json(rec["payload"])
        o = oracles.setdefault(p.label, K107Oracle(p))
        ch = checkers.setdefault(t["scene"], PathCollisionChecker(t["scene"]))
        ch.set_payload(p)
        z = dict(np.load(d / f"{a.tag}_success_lanes.npz"))
        li = r["best_lane"]
        conv = native_cubic_to_clamped(z["start_boundary"][li][:, :7], z["controls"][li][:, :7], z["goal"][li][:7])
        refiner = JointRefiner(o, ch, np.asarray(lim["position"]))
        log = refiner.refine(conv["controls"], conv["knots"], cfg)
        row = {"task_id": r["task_id"], "payload": p.label, "role": r["role"], "planner_duration_s": rec["duration_s"],
               "start_k107_s": log.durations[0] if log.durations else None,
               "final_k107_s": log.durations[-1] if log.durations else None, "accepted": log.accepted,
               "rejected": log.rejected, "oracle_calls": log.oracle_calls, "wall_s": log.wall_s,
               "config": asdict(cfg)}
        if log.best_result is not None and log.accepted:
            tr = o.sample_trajectory(log.best_controls, log.best_result, degree=3, knots=conv["knots"])
            method_record(d, log.wall_s, tr, "refined")
            export(a.run / f"validation_{a.out}" / r["task_id"] / p.label / "refined", tr,
                   {"payload": p.to_json(), "scene": t["scene"], "start": t["start"], "goal": t["goal"],
                    "limits": lim, "role": r["role"], "variant": "C_refined"})
        if not (log.best_result is not None and log.accepted):
            method_record(d, log.wall_s, None, "refine_no_improvement")
        out.append(row)
        print(r["task_id"], p.label, r["role"], row["start_k107_s"], row["final_k107_s"], log.accepted,
              f"{log.wall_s:.2f}s", flush=True)
    write_json(a.run / f"{a.out}.json", out)


if __name__ == "__main__":
    main()
