# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reconstruct deterministic proposal vectors omitted from frozen search logs.

This is a post hoc evidence-replay tool. It reruns the frozen search settings
to archive exact controls and directions; replay cost is not added to scored
search runtime and the replay cannot alter development or held-out results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import torque_dual_guidance as guidance_module
from . import torque_guided_rescue as rescue_module
from .geometry_feasibility_core import sha256_file
from .geometry_feasibility_scene import IndependentGeometryAuditor
from .torque_active import read_profile
from .torque_guided_rescue import (
    ARMS,
    SCENARIO,
    Evaluator,
    fit_nominal_quintic,
    resolve_registry,
    search,
    source_native,
    source_tasks,
)


def _control_sha(controls: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(controls).tobytes()).hexdigest()


def replay(root: Path, output: Path, *, budget: int, seed: int) -> dict:
    """Replay all frozen arms and save candidate controls and directions."""
    from .torque_guided_freeze import verify_frozen_holdout

    verify_frozen_holdout(root, budget=budget, seed=seed)
    registry = resolve_registry()
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    original_suggest = Evaluator.suggest
    for arm in ARMS:
        records[arm] = {}
        for case, info in registry.items():
            source = info["source_root"]
            discovery = info["discovery_root"]
            task_list, source_rows = source_tasks(source)
            task = next(row for row in task_list if row["case_id"] == case)
            profile_path = discovery / SCENARIO / case / "torque_profile.json"
            profile = read_profile(profile_path)
            native = source_native(source, task, source_rows[case])
            ref_lp = json.loads(
                (discovery / SCENARIO / "refined_lp_summary.json").read_text()
            )
            critical = float(ref_lp[case]["worst_progress"])
            captured = []
            evaluation_work = {}
            work_counts = {"lp_solves": 0, "collision_checks": 0}
            current_iteration = {"value": 0}
            original_lp_guidance = guidance_module.solve_point_lp
            original_lp_rescue = rescue_module.solve_point_lp
            original_collision_check = IndependentGeometryAuditor.check

            def counted_lp(*args, **kwargs):
                work_counts["lp_solves"] += 1
                return original_lp_guidance(*args, **kwargs)

            def counted_collision(self, *args, **kwargs):
                work_counts["collision_checks"] += 1
                return original_collision_check(self, *args, **kwargs)

            def recording_evaluate(self, controls):
                before_lp = work_counts["lp_solves"]
                before_collision = work_counts["collision_checks"]
                before_lp_rnea = self.rnea.rnea_calls
                before_audit_rnea = self.auditor.rnea_calls
                candidate = original_evaluate(self, controls)
                if current_iteration["value"]:
                    evaluation_work[current_iteration["value"]] = {
                        "candidate_evaluation_lp_solves": (
                            work_counts["lp_solves"] - before_lp
                        ),
                        "collision_evaluations": (
                            work_counts["collision_checks"] - before_collision
                        ),
                        "candidate_evaluation_lp_rnea_calls": (
                            self.rnea.rnea_calls - before_lp_rnea
                        ),
                        "candidate_evaluation_screen_rnea_calls": (
                            self.auditor.rnea_calls - before_audit_rnea
                        ),
                    }
                return candidate

            original_evaluate = Evaluator.evaluate

            def recording_suggest(self, incumbent, selected_arm, rng, *, iteration):
                current_iteration["value"] = iteration
                before_lp = work_counts["lp_solves"]
                before_rnea = self.rnea.rnea_calls
                before_audit_rnea = self.auditor.rnea_calls
                proposal, detail = original_suggest(
                    self, incumbent, selected_arm, rng, iteration=iteration
                )
                radius = float(detail["radius"])
                selected_bottleneck = incumbent.bottlenecks[
                    iteration % len(incumbent.bottlenecks)
                ]
                captured.append(
                    {
                        "proposal": iteration,
                        "parent_controls": incumbent.controls.copy(),
                        "parent_controls_sha256": _control_sha(incumbent.controls),
                        "parent_score": float(incumbent.score),
                        "candidate_controls": proposal.copy(),
                        "candidate_controls_sha256": _control_sha(proposal),
                        "direction": (proposal - incumbent.controls) / radius,
                        "upper_active_joints": np.flatnonzero(
                            selected_bottleneck["upper_dual"] > 1e-7
                        ).tolist(),
                        "lower_active_joints": np.flatnonzero(
                            selected_bottleneck["lower_dual"] > 1e-7
                        ).tolist(),
                        "lp_solves_in_guidance": work_counts["lp_solves"] - before_lp,
                        "guidance_rnea_calls": self.rnea.rnea_calls - before_rnea,
                        "guidance_screen_rnea_calls": (
                            self.auditor.rnea_calls - before_audit_rnea
                        ),
                        "radius": radius,
                        **detail,
                    }
                )
                return proposal, detail

            Evaluator.suggest = recording_suggest
            Evaluator.evaluate = recording_evaluate
            guidance_module.solve_point_lp = counted_lp
            rescue_module.solve_point_lp = counted_lp
            IndependentGeometryAuditor.check = counted_collision
            stamp = time.perf_counter()
            try:
                result, _ = search(
                    task,
                    profile,
                    native,
                    arm=arm,
                    budget=budget,
                    seed=seed,
                    critical_progress=critical,
                )
            finally:
                Evaluator.suggest = original_suggest
                Evaluator.evaluate = original_evaluate
                guidance_module.solve_point_lp = original_lp_guidance
                rescue_module.solve_point_lp = original_lp_rescue
                IndependentGeometryAuditor.check = original_collision_check
            replay_wall = time.perf_counter() - stamp
            source_result = root / arm / case / "search_result.json"
            expected = json.loads(source_result.read_text())["report"]
            actual = result["report"]
            fields = (
                "status",
                "score",
                "lp_expansion",
                "gravity_peak",
                "deviation_rad",
            )
            comparisons = []
            for replay_row, actual_row, frozen_row in zip(
                captured, actual["ledger"][1:], expected["ledger"][1:]
            ):
                replay_row.update(
                    {
                        key: actual_row.get(key)
                        for key in (
                            "status",
                            "score",
                            "lp_expansion",
                            "gravity_peak",
                            "deviation_rad",
                            "clearance",
                            "wall_s",
                            "rnea_calls",
                        )
                    }
                )
                replay_row.update(evaluation_work.get(replay_row["proposal"], {}))
                replay_row["lp_evaluations_total"] = int(
                    replay_row.get("lp_solves_in_guidance", 0)
                    + replay_row.get("candidate_evaluation_lp_solves", 0)
                )
                replay_row["rnea_calls_guidance_and_candidate"] = int(
                    replay_row.get("guidance_rnea_calls", 0)
                    + replay_row.get("candidate_evaluation_lp_rnea_calls", 0)
                )
                replay_row["cumulative_scored_wall_s"] = actual_row.get("wall_s")
                replay_row["constructive_timing"] = actual_row.get(
                    "constructive_timing"
                )
                matched = replay_row["proposal"] == frozen_row["proposal"]
                for field in fields:
                    left = replay_row.get(field)
                    right = frozen_row.get(field)
                    if isinstance(left, (float, int)) and isinstance(
                        right, (float, int)
                    ):
                        matched = matched and bool(
                            np.isclose(left, right, rtol=1e-9, atol=1e-10)
                        )
                comparisons.append(matched)
            arrays = {
                "initial_controls": (
                    captured[0]["parent_controls"]
                    if captured
                    else fit_nominal_quintic(
                        native, task["start"], task["goal"], count=22
                    )[0]
                ),
                "parent_controls": np.stack(
                    [row["parent_controls"] for row in captured]
                ),
                "candidate_controls": np.stack(
                    [row["candidate_controls"] for row in captured]
                ),
                "directions": np.stack([row["direction"] for row in captured]),
            }
            npz = output / f"{arm}_{case}_proposals.npz"
            np.savez_compressed(npz, **arrays)
            ledger = []
            for row in captured:
                ledger.append(
                    {
                        key: value
                        for key, value in row.items()
                        if key
                        not in ("parent_controls", "candidate_controls", "direction")
                    }
                )
            report = {
                "arm": arm,
                "case_id": case,
                "seed": seed,
                "budget": budget,
                "replay_wall_s_not_scored": replay_wall,
                "proposal_count": len(captured),
                "original_ledger_count": len(expected["ledger"]) - 1,
                "candidate_metrics_match_original_ledger": bool(
                    comparisons and all(comparisons)
                ),
                "matched_proposals": int(sum(comparisons)),
                "replay_lp_solves": work_counts["lp_solves"],
                "replay_collision_evaluations": work_counts["collision_checks"],
                "npz_path": str(npz.resolve()),
                "npz_sha256": sha256_file(npz),
                "initial_controls_sha256": _control_sha(
                    captured[0]["parent_controls"]
                    if captured
                    else fit_nominal_quintic(
                        native, task["start"], task["goal"], count=22
                    )[0]
                ),
                "original_result_sha256": sha256_file(source_result),
                "proposal_ledger": ledger,
            }
            json_path = output / f"{arm}_{case}_proposals.json"
            json_path.write_text(json.dumps(report, indent=2) + "\n")
            report["json_path"] = str(json_path.resolve())
            report["json_sha256"] = sha256_file(json_path)
            records[arm][case] = report
    manifest = {
        "schema_version": 1,
        "purpose": "post hoc deterministic evidence replay; not scored method search",
        "frozen_search_source_sha256": sha256_file(
            Path(__file__).resolve().parent / "torque_guided_rescue.py"
        ),
        "arms": records,
    }
    (output / "proposal_replay_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    """Replay frozen proposal sequence and print a compact status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()
    result = replay(args.root, args.output, budget=args.budget, seed=args.seed)
    print(
        json.dumps(
            {
                arm: {
                    case: {
                        "proposal_count": row["proposal_count"],
                        "ledger_match": row["candidate_metrics_match_original_ledger"],
                        "matched_proposals": row["matched_proposals"],
                    }
                    for case, row in cases.items()
                }
                for arm, cases in result["arms"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
