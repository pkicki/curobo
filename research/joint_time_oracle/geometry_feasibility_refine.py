# SPDX-License-Identifier: Apache-2.0
"""High-resolution independent torque-LP obstruction audit with dual evidence.

A positive LP value at even ONE regular point disproves torque feasibility
for the fixed original geometric path, under any retiming, subject to
model fidelity and floating-point numerical reliability. It does NOT
disprove feasibility of another geometric path or of the full planner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from .cohort_retime import path_coefficients
from .extreme_torque_path_lp import inspect_curve
from .geometry_feasibility_core import MIN_ORIGINAL_LP_GAP, sha256_file
from .geometry_feasibility_discover import OUT, SOURCE, source_native, source_tasks
from .torque_active import read_profile


def refinement_candidates(
    source: Path, arm: Path, *, case_filter: str | None = None
) -> list[dict]:
    """Return every frozen positive LP case, including cases without a witness."""
    tasks, _ = source_tasks(source)
    selected = []
    positive_statuses = {
        "POSITIVE_ORIGINAL_LP_NEEDS_ALTERNATIVE",
        "CONSTRUCTIVE_WITNESS_PRESCREEN",
    }
    for task in tasks:
        case = task["case_id"]
        if case_filter is not None and case != case_filter:
            continue
        discovery_path = arm / case / "discovery.json"
        if not discovery_path.exists():
            continue
        discovery = json.loads(discovery_path.read_text())
        if discovery.get("status") in positive_statuses:
            selected.append(task)
    return selected


def dual_slack_certificate(a, m, g, bounds: np.ndarray) -> dict:
    a, m, g = (np.asarray(x, dtype=float) for x in (a, m, g))
    low, high = np.asarray(bounds, dtype=float)
    halfwidth = (high - low) / 2
    A = np.vstack(
        [np.column_stack((m, a, -halfwidth)), np.column_stack((-m, -a, -halfwidth))]
    )
    b = np.r_[high - g, g - low]
    primal = linprog(
        c=[0, 0, 1],
        A_ub=A,
        b_ub=b,
        bounds=[(0, None), (None, None), (0, None)],
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
        },
    )
    if not primal.success:
        return {"status": "LP_FAILED", "message": primal.message}
    lam = -np.asarray(primal.ineqlin.marginals, dtype=float)
    dual = float(-lam @ b)
    c = np.array([0.0, 0.0, 1.0])
    reduced = c + A.T @ lam
    dual_gap = float(primal.fun - dual)
    primal_violation = float(max(0, np.max(A @ primal.x - b)))
    dual_violation = float(
        max(0, -np.min(lam), -reduced[0], -reduced[2], abs(reduced[1]) - 1e-9)
    )
    valid = bool(
        dual > MIN_ORIGINAL_LP_GAP
        and abs(dual_gap) < 1e-6
        and primal_violation < 1e-7
        and dual_violation < 1e-7
    )
    return {
        "status": "SOLVED",
        "minimum_required_expansion": float(primal.fun),
        "dual_lower_bound": dual,
        "duality_gap": dual_gap,
        "primal_violation": primal_violation,
        "dual_violation": dual_violation,
        "numerically_strong_infeasibility": valid,
        "constraint_multiplier_nonzero": int(np.sum(lam > 1e-8)),
        "active_torque_inequality_ids": np.where(lam > 1e-7)[0].tolist(),
        "note": "floating-point LP dual check; NOT formal interval-rigorous infeasibility",
    }


def refine(source: Path, arm: Path, *, case_filter: str | None = None) -> dict:
    # The scenario's ``tasks.json`` deliberately contains only constructive W
    # candidates for later validation.  Refinement must include every positive
    # original-path LP screen, including cases where no alternate was found.
    tasks = refinement_candidates(source, arm, case_filter=case_filter)
    _, rows = source_tasks(source)
    results = {}
    for task in tasks:
        case = task["case_id"]
        try:
            profile = read_profile(arm / case / "torque_profile.json")
            geometry = source_native(source, task, rows[case])
            effort = np.stack((profile["effort_lower"], profile["effort_upper"]))
            low = inspect_curve(geometry.evaluate_trimmed, effort, samples=257)
            high = inspect_curve(geometry.evaluate_trimmed, effort, samples=1025)
            worst = max(
                (low, high),
                key=lambda v: v.get("max_required_torque_bound_expansion", -1),
            )
            if worst.get("status") != "SOLVED":
                raise ValueError("necessary LP high-resolution scan failed")
            s = float(worst["worst_progress"])
            # Off-grid recurrence around strongest point and residual/dual verification.
            grid = np.clip(
                s + np.array([-1e-3, -5e-4, -1e-4, -1e-5, 0, 1e-5, 1e-4, 5e-4, 1e-3]),
                1e-9,
                1 - 1e-9,
            )
            arr = np.asarray(geometry.evaluate_trimmed(grid))
            coeff = path_coefficients(arr)
            certificates = [
                dual_slack_certificate(coeff[0, i], coeff[1, i], coeff[2, i], effort)
                for i in range(len(grid))
            ]
            good = [
                (float(grid[i]), v)
                for i, v in enumerate(certificates)
                if v.get("numerically_strong_infeasibility")
            ]
            if not good:
                result = {
                    "status": "NO_STRONG_LP_CERTIFICATE",
                    "low": low,
                    "high": high,
                    "candidates": certificates,
                }
            else:
                location, cert = max(good, key=lambda item: item[1]["dual_lower_bound"])
                result = {
                    "status": "NUMERICALLY_STRONG_ORIGINAL_PATH_INFEASIBILITY",
                    "worst_progress": location,
                    "numerical_dual_certificate": cert,
                    "source_original_geometry_sha256": sha256_file(
                        source / case / "a10_buffers.npz"
                    ),
                    "low": low,
                    "high": high,
                    "note": (
                        "one point disproves fixed native geometry, NOT all "
                        "native replanning routes"
                    ),
                }
        except Exception as exc:
            result = {"status": "ERROR", "reason": repr(exc)}
        results[case] = result
        (arm / case / "refined_original_path_lp.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(case, result["status"], flush=True)
    (arm / "refined_lp_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=OUT)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--scenario", required=True)
    p.add_argument("--case")
    a = p.parse_args()
    refine(a.source, a.root / a.scenario, case_filter=a.case)


if __name__ == "__main__":
    main()
