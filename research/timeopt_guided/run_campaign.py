# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-2: TimeOpt-guided geometry search on frozen native cells (CUDA env).

Selection (frozen to ``campaign_selection.json`` BEFORE any search runs):
  * every admissible cell where native failed at the highest available budget;
  * a seeded random control sample of native successes (``--success-fraction``).
For each selected cell and each search seed, arms B, C, D run with the SAME
proposal budget. Exports (for ``independent_validate`` in the Pinocchio env):
  * native successful trajectory  -> ``native.traj.{npz,json}``
  * per arm/seed: first feasible and best feasible K107 trajectories
  * per payload: ``parity_<label>.npz`` (random q, qd, qdd and cuRobo tau)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import Payload, read_json, write_json
from .guided_search import ARMS, GuidedSearch, SearchConfig


def physical_limits(kp) -> dict:
    jl = kp.joint_limits
    out = {}
    for k in ("velocity", "acceleration", "jerk"):
        v = getattr(jl, k).detach().cpu().numpy()
        out[k] = (np.abs(v).min(axis=0) if v.ndim == 2 else np.abs(v))[:7].tolist()
    out["position"] = jl.position.detach().cpu().numpy()[:, :7].tolist()
    out["effort"] = jl.effort.detach().cpu().numpy()[:, :7].tolist()
    return out


def export(path: Path, traj: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(traj[k]) for k in ("time", "q", "qd", "qdd", "jerk")}
    np.savez_compressed(str(path) + ".traj.npz", **arrays)
    write_json(Path(str(path) + ".traj.json"), meta)


def best_budget(run: Path) -> int:
    budgets = sorted({int(p.stem.split("_R")[1]) for p in run.glob("cells/*/*/native_R*.json")
                      if p.stem.count("_") == 1})
    return budgets[-1]


def select(run: Path, bank: dict, *, success_fraction: float, seed: int, split: str | None) -> list[dict]:
    rng = np.random.default_rng(seed)
    tasks = {t["task_id"]: t for t in bank["tasks"]}
    picked = []
    for f in sorted(run.glob("cells/*/*/native_R1.json")):
        r1 = read_json(f)
        if split and tasks[r1["task_id"]]["split"] != split:
            continue
        # highest budget available for this cell
        files = sorted(f.parent.glob("native_R*.json"), key=lambda p: int(p.stem.split("_R")[1]))
        files = [p for p in files if p.stem.count("_") == 1]
        top = read_json(files[-1])
        dur = top.get("duration_s")
        if not top["success"] or dur is None or not np.isfinite(dur):  # NaN "success" = failure
            picked.append({"cell": str(f.parent), "native_budget": top["restarts_budget"], "role": "native_failure"})
        elif rng.random() < success_fraction:
            picked.append({"cell": str(f.parent), "native_budget": top["restarts_budget"], "role": "native_success_control",
                           "native_file": files[-1].name})
    return picked


def main() -> None:
    import torch

    from .collision_screen import PathCollisionChecker
    from .geometry import fit_quintic
    from .k107_oracle import K107Oracle

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True, help="mine_native output")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--search-seeds", type=int, default=3)
    ap.add_argument("--max-proposals", type=int, default=128)
    ap.add_argument("--success-fraction", type=float, default=0.15)
    ap.add_argument("--split", choices=("development", "confirmation"), default="development")
    ap.add_argument("--selection-seed", type=int, default=20261110)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--native-seeds", action="store_true",
                    help="also seed search with quintic fits of the native trajectory")
    ap.add_argument("--config", type=Path, default=None, help="JSON of SearchConfig overrides")
    ap.add_argument("--freeze", type=Path, default=None, help="freeze.json (required for confirmation)")
    a = ap.parse_args()
    overrides = read_json(a.config) if a.config else {}
    if a.split == "confirmation":
        from .freeze import check

        if a.freeze is None:
            raise SystemExit("confirmation split requires --freeze (see freeze.py)")
        check(a.freeze, overrides)

    bank = read_json(a.bank / "tasks.json")
    tasks = {t["task_id"]: t for t in bank["tasks"]}
    sel_file = a.output / "campaign_selection.json"
    if sel_file.exists():
        selection = read_json(sel_file)["cells"]
    else:
        selection = select(a.run, bank, success_fraction=a.success_fraction, seed=a.selection_seed, split=a.split)
        write_json(sel_file, {"cells": selection, "split": a.split, "success_fraction": a.success_fraction,
                              "selection_seed": a.selection_seed, "frozen_before_search": True})
    if a.limit:
        selection = selection[: a.limit]
    print(f"{len(selection)} selected cells")

    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg

    kp = KinematicsCfg.from_robot_yaml_file("franka.yml").kinematics_config
    limits = physical_limits(kp)
    oracles, checkers = {}, {}
    rng = np.random.default_rng(0)
    for ci, item in enumerate(selection):
        cell = Path(item["cell"])
        rec = read_json(cell / f"native_R{item['native_budget']}.json")
        t = tasks[rec["task_id"]]
        p = Payload.from_json(rec["payload"])
        if p.label not in oracles:
            oracles[p.label] = K107Oracle(p)
            # payload parity samples for the independent validator
            n = 64
            lo, hi = np.asarray(limits["position"])
            q = rng.uniform(lo, hi, size=(n, 7))
            qd = rng.uniform(-1, 1, size=(n, 7)) * np.asarray(limits["velocity"])
            qdd = rng.uniform(-1, 1, size=(n, 7)) * np.asarray(limits["acceleration"])
            tau = oracles[p.label].torque(q, qd, qdd)
            a.output.mkdir(parents=True, exist_ok=True)
            np.savez(a.output / f"parity_{p.label}.npz", q=q, qd=qd, qdd=qdd, tau_curobo=tau,
                     payload_json=np.array(__import__("json").dumps(p.to_json())))
        oracle = oracles[p.label]
        checker = checkers.setdefault(t["scene"], PathCollisionChecker(t["scene"]))
        checker.set_payload(p)
        out = a.output / rec["task_id"] / p.label
        if (out / "summary.json").exists():
            continue  # resume
        meta = {"payload": p.to_json(), "scene": t["scene"], "start": t["start"], "goal": t["goal"],
                "limits": limits, "task_id": t["task_id"], "role": item["role"]}
        seeds = []
        native_traj = cell / f"native_R{item['native_budget']}_traj.npz"
        native_ok = bool(rec["success"] and rec.get("duration_s") is not None and np.isfinite(rec["duration_s"]))
        if native_ok and native_traj.exists() and all(np.isfinite(v).all() for v in np.load(native_traj).values()):
            z = dict(np.load(native_traj))
            export(out / "native", z, meta | {"kind": "native", "duration_s": rec["duration_s"]})
            if a.native_seeds:
                arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(z["q"], axis=0), axis=1))]
                seeds.append(fit_quintic(arc / max(arc[-1], 1e-9), z["q"]))
        lane_file = cell / f"native_R{item['native_budget']}_selected_lane.npz"
        if native_ok and lane_file.exists():
            # Retimer-only decomposition: K107 on the EXACT native cubic geometry.
            from .geometry import native_cubic_to_clamped

            z = dict(np.load(lane_file))
            conv = native_cubic_to_clamped(z["start_boundary"][:, :7], z["controls"][:, :7], z["goal"][:7])
            res = oracle.retime(conv["controls"], degree=3, knots=conv["knots"])
            write_json(out / "native_geometry_k107.json", res.to_json())
            if res.feasible:
                tr = oracle.sample_trajectory(conv["controls"], res, degree=3, knots=conv["knots"])
                export(out / "native_geometry_k107", tr, meta | {"kind": "native_geometry_k107",
                                                                 "k107_duration_s": res.duration_s})
        search = GuidedSearch(oracle, checker, np.asarray(limits["position"]))
        summary = {"task_id": t["task_id"], "payload": p.label, "role": item["role"],
                   "native": rec, "arms": {}}
        k = ci % len(a.arms)  # rotate arm order per cell: no systematic first-run advantage
        for arm in a.arms[k:] + a.arms[:k]:
            for sd in range(a.search_seeds):
                cfg = SearchConfig(**({"max_proposals": a.max_proposals} | overrides | {"arm": arm, "seed": sd}))
                torch.cuda.synchronize()
                log = search.run(np.asarray(t["start"]), np.asarray(t["goal"]), cfg, seeds=seeds)
                key = f"{arm}_s{sd}"
                write_json(out / f"search_{key}.json", log.to_json())
                info = {"wall_s": log.wall_s, "oracle_calls": log.oracle_calls,
                        "first_feasible_s": log.first_feasible_s,
                        "first_feasible_proposals": log.first_feasible_proposals,
                        "best_duration_s": log.to_json()["best_duration_s"]}
                feas = [c for c in log.candidates if c.result is not None and c.result.feasible]
                if feas:
                    first = feas[0]
                    best = log.best()
                    for tag, c in (("first", first), ("best", best)):
                        if tag == "best" and c.index == first.index:
                            info["best_is_first"] = True
                            continue
                        tr = oracle.sample_trajectory(c.controls, c.result)
                        export(out / f"{key}_{tag}", tr, meta | {
                            "kind": f"search_{arm}_{tag}", "k107_duration_s": c.result.duration_s,
                            "candidate_index": c.index, "t_done_s": c.t_done_s})
                        np.save(out / f"{key}_{tag}_controls.npy", c.controls)
                summary["arms"][key] = info
                print(f"{t['task_id']} {p.label} {key}: {info}", flush=True)
        write_json(out / "summary.json", summary)


if __name__ == "__main__":
    main()
