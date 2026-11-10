# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idea B: retime ALL candidate lanes of a successful solve with K107, keep the fastest (CUDA env).

Geometry is unchanged: every lane is converted EXACTLY (native cubic -> clamped
cubic) and retimed by K107 (TOPP-RA + Newton, C3 without torque rate, Scaled95).
Only lanes that passed the planner's own collision and path checks are eligible.
The selected K107 trajectory is exported for ``independent_validate`` next to the
planner's own trajectory. Reports the duration ratio vs the planner's choice and
the extra compute (sequential K107 calls; batching is future work).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .common import Payload, read_json, write_json


def main() -> None:
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg

    from .geometry import native_cubic_to_clamped
    from .k107_oracle import K107Oracle
    from .run_campaign import export, physical_limits

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--tag", default="hybrid_R1")
    ap.add_argument("--cells", type=Path, required=True)
    a = ap.parse_args()
    bank = {t["task_id"]: t for t in read_json(a.bank / "tasks.json")["tasks"]}
    lim = physical_limits(KinematicsCfg.from_robot_yaml_file("franka.yml").kinematics_config)
    oracles, rows = {}, []
    for c in read_json(a.cells):
        d = a.run / "cells" / c["task_id"] / c["payload"]
        f = d / f"{a.tag}_success_lanes.npz"
        rec = read_json(d / f"{a.tag}.json")
        if not f.exists() or not rec["success"]:
            continue
        p = Payload.from_json(rec["payload"])
        o = oracles.setdefault(p.label, K107Oracle(p))
        z = dict(np.load(f))
        lanes = z["controls"].shape[0]
        eligible = np.ones(lanes, bool)
        for k in ("mask_collision_valid", "mask_path_valid"):
            if k in z:
                eligible &= z[k].astype(bool)
        t0 = time.perf_counter()
        per = []
        for li in range(lanes):
            if not eligible[li]:
                per.append(None)
                continue
            try:
                conv = native_cubic_to_clamped(z["start_boundary"][li][:, :7], z["controls"][li][:, :7],
                                               z["goal"][li][:7])
            except ValueError:
                per.append(None)
                continue
            r = o.retime(conv["controls"], degree=3, knots=conv["knots"], diagnose_infeasible=False)
            per.append((r, conv))
        wall = time.perf_counter() - t0
        feas = [(i, x[0].duration_s) for i, x in enumerate(per) if x is not None and x[0].feasible]
        row = {"task_id": c["task_id"], "payload": p.label, "role": c.get("role"),
               "planner_duration_s": rec["duration_s"], "route": rec.get("route"),
               "eligible_lanes": int(eligible.sum()), "k107_feasible_lanes": len(feas),
               "native_timeopt_success_lanes": int(np.sum(z.get("success", np.zeros(lanes)))),
               "k107_wall_s": wall, "lane_durations": {str(i): dd for i, dd in feas}}
        if feas:
            i, dd = min(feas, key=lambda x: x[1])
            r, conv = per[i]
            row.update(best_lane=i, best_k107_s=dd, ratio=dd / rec["duration_s"],
                       selected_lane=int(z["selected_lane"]))
            sel = int(z["selected_lane"])
            if sel >= 0 and per[sel] is not None and per[sel][0].feasible:
                row["same_lane_k107_s"] = per[sel][0].duration_s
                row["same_lane_ratio"] = per[sel][0].duration_s / rec["duration_s"]
            tr = o.sample_trajectory(conv["controls"], r, degree=3, knots=conv["knots"])
            t = bank[c["task_id"]]
            export(a.run / "validation_B" / c["task_id"] / p.label / "k107_best_lane", tr,
                   {"payload": p.to_json(), "scene": t["scene"], "start": t["start"], "goal": t["goal"],
                    "limits": lim, "role": c.get("role"), "variant": "B_k107_best_lane"})
        rows.append(row)
        print(c["task_id"], p.label, row.get("ratio"), row.get("same_lane_ratio"), f"{wall:.2f}s", flush=True)
    write_json(a.run / "lane_retime.json", rows)


if __name__ == "__main__":
    main()
