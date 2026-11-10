# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 native cuRoboV2-Fast runs on every admissible task x payload cell (CUDA env).

Budget ladder: every admissible cell gets ``--restarts`` (default 1) complete
``plan_cspace`` calls; ``--escalate-from`` re-runs only cells that failed in a
previous stage with a larger restart budget (4, then 16). The candidate list for
escalation is frozen BEFORE escalation (from the previous stage's files).

Per cell, writes ``cells/<task_id>/<payload>/native_R<k>.json`` and
  * ``native_R<k>_traj.npz``  dense selected trajectory on success;
  * ``native_R<k>_lanes.npz`` all TimeOpt lanes of the LAST TrajOpt solve of every
    failed restart (exact native cubic geometry + seed interval), for diagnosis.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .common import Payload, read_json, write_json


def cell_dir(root: Path, task_id: str, payload: Payload) -> Path:
    return root / "cells" / task_id / payload.label


def run(bank_dir: Path, out: Path, *, restarts: int, scenes: list[str] | None, escalate_from: int | None,
        split: str | None, limit: int | None, seed: int, variant: str = "native",
        cells_file: Path | None = None) -> None:
    from .native_runner import NativePlanner

    bank = read_json(bank_dir / "tasks.json")
    cells = read_json(bank_dir / "cells.json")
    tasks = {t["task_id"]: t for t in bank["tasks"]}
    tag = f"native_R{restarts}" if variant == "native" else f"{variant.replace(':', '-')}_R{restarts}"
    only = None
    if cells_file is not None:
        only = {(x["task_id"], x["payload"]) for x in read_json(cells_file)}
    todo = []
    for c in cells:
        t = tasks[c["task_id"]]
        if not c["admissible"] or (scenes and t["scene"] not in scenes) or (split and t["split"] != split):
            continue
        p = Payload.from_json(c["payload"])
        if only is not None and (t["task_id"], p.label) not in only:
            continue
        if escalate_from is not None:
            prev = cell_dir(out, t["task_id"], p) / f"native_R{escalate_from}.json"
            if not prev.exists() or read_json(prev)["success"]:
                continue
        todo.append((t, p))
    if limit:
        todo = todo[:limit]
    write_json(out / f"{tag}_frozen_todo.json",
               [{"task_id": t["task_id"], "payload": p.label} for t, p in todo])
    print(f"{tag}: {len(todo)} cells")
    by_scene: dict[str, list] = {}
    for t, p in todo:
        by_scene.setdefault(t["scene"], []).append((t, p))
    for scene, items in by_scene.items():
        if variant.startswith("hybrid"):
            from .native_runner import HybridPlanner

            planner = HybridPlanner(scene, seed=seed, fallback=variant.split(":", 1)[1] if ":" in variant
                                    else "gravity_aware_85")
        else:
            planner = NativePlanner(scene, seed=seed, variant=variant)
        items.sort(key=lambda x: x[1].label)  # group by payload to limit updates
        current = None
        for t, p in items:
            d = cell_dir(out, t["task_id"], p)
            if (d / f"{tag}.json").exists():
                continue
            if current != p.label:
                planner.set_payload(p)
                current = p.label
                parity = planner.gravity_all_models(np.stack([t["start"], t["goal"]]))
            t0 = time.perf_counter()
            attempts = planner.plan(np.asarray(t["start"]), np.asarray(t["goal"]), restarts=restarts)
            wall = time.perf_counter() - t0
            last = attempts[-1]
            rec = {
                "task_id": t["task_id"], "scene": scene, "stratum": t["stratum"], "split": t["split"],
                "payload": p.to_json(), "restarts_budget": restarts, "restarts_used": len(attempts),
                "success": last.success, "duration_s": last.duration_s,
                "wall_s": wall, "attempt_wall_s": [a.wall_s for a in attempts],
                "failure_stages": [a.failure_stage for a in attempts],
                "native_reported_success": [a.native_reported_success or a.success for a in attempts],
                "planner_warmup_s": planner.warmup_s,
                "trajectory_source": None if last.trajectory is None else last.trajectory.get("source"),
                "has_timeopt_dynamics_probe": sorted(parity.keys()),
                "role": "native cuRoboV2-Fast plan_cspace defaults; independent validation pending",
            }
            d.mkdir(parents=True, exist_ok=True)
            if last.success and last.trajectory is not None:
                tr = {k: v for k, v in last.trajectory.items() if isinstance(v, np.ndarray)}
                np.savez_compressed(d / f"{tag}_traj.npz", **tr)
                if last.timeopt_snapshots:  # all lanes of the successful solve (idea B)
                    np.savez_compressed(d / f"{tag}_success_lanes.npz", **last.timeopt_snapshots[-1],
                                        selected_lane=np.array(last.trajectory.get("lane", -1)))
                lane = last.trajectory.get("lane")
                if lane is not None and last.timeopt_snapshots:
                    snap = last.timeopt_snapshots[-1]
                    np.savez_compressed(d / f"{tag}_selected_lane.npz",
                                        **{k: snap[k][lane] for k in ("controls", "start_boundary", "goal")})
            lanes = {}
            for ai, a in enumerate(attempts):
                if a.success or not a.timeopt_snapshots:
                    continue
                snap = a.timeopt_snapshots[-1]
                for k, v in snap.items():
                    lanes[f"a{ai}_{k}"] = v
            if lanes:
                np.savez_compressed(d / f"{tag}_lanes.npz", **lanes)
            rec["variant"] = variant
            rec["route"] = getattr(last, "route", None)
            write_json(d / f"{tag}.json", rec)
            print(f"{t['task_id']} {p.label}: success={last.success} T={last.duration_s} "
                  f"wall={wall:.2f}s stages={rec['failure_stages']}")
        planner.destroy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--escalate-from", type=int, default=None)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--split", choices=("development", "confirmation"), default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="native")
    ap.add_argument("--cells", type=Path, default=None, help="JSON list of {task_id, payload} to run")
    a = ap.parse_args()
    run(a.bank, a.output, restarts=a.restarts, scenes=a.scenes, escalate_from=a.escalate_from,
        split=a.split, limit=a.limit, seed=a.seed, variant=a.variant, cells_file=a.cells)


if __name__ == "__main__":
    main()
