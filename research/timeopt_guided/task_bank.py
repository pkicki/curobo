# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prospective start/goal task bank for nominal-limit payload hard cases (CUDA env).

Strata (reported separately, never pooled):
  broad     uniformly sampled collision-free start/goal pairs;
  enriched  pairs whose straight joint-space line has an interior gravity
            "barrier" (peak static wrist/shoulder utilization >= 0.9 under the
            3 kg offset payload) while both endpoints stay <= 0.8. This is a
            screening heuristic for torque-active geometry, not a failure label.
All proposals (accepted and rejected, with reason) are stored. Task ids are
hash-split into development / confirmation by SOURCE task, so all payloads of a
task share a split.

Cell admissibility (per task x payload), evaluated before any planner run:
  endpoints collision-free WITH that payload sphere, and endpoint static gravity
  utilization <= ENDPOINT_MAX vs 0.95 * nominal effort.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .common import (
    RESULTS, SCENES, Payload, canonical_hash, payload_grid, rest_controls, task_split, write_json,
)
from .geometry import evaluate

ENDPOINT_MAX = 0.9
ENRICH_PAYLOAD = Payload(3.0, (0.08, 0.0, 0.16), 0.06, "enrich_m3_offset")
MIN_JOINT_DISTANCE = 1.0  # rad (L2) between start and goal


def straight_gravity_profile(oracle, q0, q1, n: int = 65) -> np.ndarray:
    q = evaluate(rest_controls(q0, q1), np.linspace(0, 1, n))
    return np.abs(oracle.gravity(q)) / oracle.torque_limit[None]


def generate(output: Path, *, per_scene_broad: int, per_scene_enriched: int, seed: int,
             max_samples: int = 20000) -> dict:
    from .collision_screen import PathCollisionChecker
    from .k107_oracle import K107Oracle

    rng = np.random.default_rng(seed)
    oracle = K107Oracle(ENRICH_PAYLOAD)
    tasks, rejected = [], []
    for scene in ("free", "table", "cluttered"):  # base scenes only
        checker = PathCollisionChecker(scene)
        checker.set_payload(Payload(0.0, (0.0, 0.0, 0.0)))
        pool = checker.checker.sample(max_samples, mask_valid=True).detach().cpu().numpy()[:, :7]
        rng.shuffle(pool)
        counts = {"broad": 0, "enriched": 0}
        want = {"broad": per_scene_broad, "enriched": per_scene_enriched}
        i = 0
        t = time.perf_counter()
        while (counts["broad"] < want["broad"] or counts["enriched"] < want["enriched"]) and i + 1 < len(pool):
            q0, q1 = pool[i], pool[i + 1]
            i += 2
            rec = {"scene": scene, "start": q0.tolist(), "goal": q1.tolist()}
            dist = float(np.linalg.norm(q1 - q0))
            if dist < MIN_JOINT_DISTANCE:
                rejected.append(rec | {"reason": "too_short", "distance": dist})
                continue
            prof = straight_gravity_profile(oracle, q0, q1)
            end_util = float(max(prof[0].max(), prof[-1].max()))
            peak = float(prof[1:-1].max())
            barrier = end_util <= 0.8 and peak >= 0.9
            stratum = "enriched" if barrier and counts["enriched"] < want["enriched"] else "broad"
            if counts[stratum] >= want[stratum]:
                rejected.append(rec | {"reason": f"{stratum}_full", "peak": peak, "end": end_util})
                continue
            counts[stratum] += 1
            core = {"scene": scene, "stratum": stratum, "start": q0.tolist(), "goal": q1.tolist()}
            tid = f"{scene}_{stratum}_{counts[stratum] - 1:03d}_{canonical_hash(core)[:8]}"
            tasks.append(core | {
                "task_id": tid, "split": task_split(tid), "straight_peak_util_3kg": peak,
                "straight_endpoint_util_3kg": end_util, "joint_distance": dist,
            })
        print(f"{scene}: {counts} from {i // 2} pairs in {time.perf_counter() - t:.1f}s")
    out = {"seed": seed, "endpoint_max": ENDPOINT_MAX, "enrich_payload": ENRICH_PAYLOAD.to_json(),
           "payloads": [p.to_json() for p in payload_grid()], "tasks": tasks}
    out["bank_sha256"] = canonical_hash(out)
    write_json(output / "tasks.json", out)
    write_json(output / "rejected_proposals.json", rejected)
    return out


def admissible_cells(bank: dict, *, split: str | None = None) -> list[dict]:
    """Expand tasks x payloads and evaluate endpoint admissibility (needs CUDA)."""
    from .collision_screen import PathCollisionChecker
    from .k107_oracle import K107Oracle

    payloads = [Payload.from_json(p) for p in bank["payloads"]]
    tasks = [t for t in bank["tasks"] if split is None or t["split"] == split]
    cells = []
    checkers = {s: PathCollisionChecker(s) for s in {t["scene"] for t in tasks}}
    for p in payloads:
        oracle = K107Oracle(p)
        for ch in checkers.values():
            ch.set_payload(p)
        for t in tasks:
            ends = np.stack([t["start"], t["goal"]])
            util = float((np.abs(oracle.gravity(ends)) / oracle.torque_limit[None]).max())
            free = bool(checkers[t["scene"]].valid_configs(ends[None]).all())
            reason = None if free and util <= ENDPOINT_MAX else (
                "endpoint_collision_with_payload" if not free else "endpoint_static_gravity")
            cells.append({"task_id": t["task_id"], "payload": p.to_json(), "admissible": reason is None,
                          "reason": reason, "endpoint_gravity_util": util})
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=RESULTS / "task_bank_20261110")
    ap.add_argument("--broad", type=int, default=40)
    ap.add_argument("--enriched", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261110)
    a = ap.parse_args()
    bank = generate(a.output, per_scene_broad=a.broad, per_scene_enriched=a.enriched, seed=a.seed)
    cells = admissible_cells(bank)
    write_json(a.output / "cells.json", cells)
    n = sum(c["admissible"] for c in cells)
    print(f"tasks={len(bank['tasks'])} cells={len(cells)} admissible={n}")


if __name__ == "__main__":
    main()
