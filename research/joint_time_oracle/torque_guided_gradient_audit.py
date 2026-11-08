# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce a finite-difference and active-set audit of the LP-dual direction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .cohort_geometry import QuinticGeometry
from .geometry_feasibility_core import sha256_file
from .torque_active import read_profile
from .torque_dual_guidance import (
    AlignedPointDynamics,
    normalized_descent_direction,
    solve_point_lp,
)
from .torque_guided_rescue import (
    ROOTS,
    SCENARIO,
    fit_nominal_quintic,
    source_native,
    source_tasks,
)


def _active_sets(lp) -> dict:
    return {
        "upper": np.flatnonzero(lp.upper_dual > 1e-7).tolist(),
        "lower": np.flatnonzero(lp.lower_dual > 1e-7).tolist(),
    }


def audit(case: str, result_root: Path) -> dict:
    """Sweep derivative steps and finite path perturbations at the frozen bottleneck."""
    source_name, discovery_name = ROOTS[case]
    source_root = result_root.parent / source_name
    discovery_root = result_root.parent / discovery_name / SCENARIO
    tasks, rows = source_tasks(source_root)
    task = next(row for row in tasks if row["case_id"] == case)
    native = source_native(source_root, task, rows[case])
    controls, fit = fit_nominal_quintic(native, task["start"], task["goal"], count=22)
    profile_path = discovery_root / case / "torque_profile.json"
    profile = read_profile(profile_path)
    bounds = np.stack((profile["effort_lower"], profile["effort_upper"]))
    lp_file = discovery_root / case / "refined_original_path_lp.json"
    bottleneck = json.loads(lp_file.read_text())
    progress = float(bottleneck["worst_progress"])
    dynamics = AlignedPointDynamics()
    base_geometry = QuinticGeometry(controls)
    a, m, g = dynamics.coefficients(base_geometry, progress)
    base_lp = solve_point_lp(a, m, g, bounds)
    if base_lp.status != "SOLVED" or base_lp.expansion <= 0:
        raise ValueError("frozen source bottleneck is not a positive certified LP")
    step_sweep = []
    for gradient_step in (1e-3, 5e-4, 2e-4, 1e-4, 5e-5):
        gradient = dynamics.dual_gradient(
            controls, progress, base_lp, step=gradient_step
        )
        direction = normalized_descent_direction(gradient)
        if direction is None:
            raise ValueError("LP dual direction failed normalization")
        checks = []
        for path_step in (1e-3, 3e-4, 1e-4, 3e-5):
            plus_a, plus_m, plus_g = dynamics.coefficients(
                QuinticGeometry(controls + path_step * direction), progress
            )
            minus_a, minus_m, minus_g = dynamics.coefficients(
                QuinticGeometry(controls - path_step * direction), progress
            )
            plus = solve_point_lp(plus_a, plus_m, plus_g, bounds)
            minus = solve_point_lp(minus_a, minus_m, minus_g, bounds)
            checks.append(
                {
                    "path_step": path_step,
                    "plus_epsilon": plus.expansion,
                    "minus_epsilon": minus.expansion,
                    "predicted_directional_derivative": -float(
                        np.linalg.norm(gradient)
                    ),
                    "one_sided_descent_fd": (plus.expansion - base_lp.expansion)
                    / path_step,
                    "central_directional_fd": (plus.expansion - minus.expansion)
                    / (2 * path_step),
                    "plus_active_set": _active_sets(plus),
                    "minus_active_set": _active_sets(minus),
                    "active_set_changed": _active_sets(plus) != _active_sets(minus),
                    "lp_status": [plus.status, minus.status],
                }
            )
        step_sweep.append(
            {
                "gradient_fd_step": gradient_step,
                "gradient_norm": float(np.linalg.norm(gradient)),
                "predicted_directional_derivative": -float(np.linalg.norm(gradient)),
                "direction_finite": bool(np.isfinite(direction).all()),
                "checks": checks,
            }
        )
    controls_path = result_root / "dual_gradient_validation_controls.npz"
    np.savez_compressed(controls_path, controls=controls, progress=np.array(progress))
    result = {
        "case_id": case,
        "task_sha256": rows[case].get("task_sha256"),
        "source_native_geometry_sha256": sha256_file(
            source_root / case / "a10_buffers.npz"
        ),
        "torque_profile_sha256": profile["profile_sha256"],
        "controls_archive": str(controls_path.resolve()),
        "controls_archive_sha256": sha256_file(controls_path),
        "fit": fit,
        "progress": progress,
        "baseline_lp": {
            "status": base_lp.status,
            "epsilon": base_lp.expansion,
            "duality_gap": base_lp.duality_gap,
            "primal_residual": base_lp.primal_residual,
            "dual_residual": base_lp.dual_residual,
            "active_set": _active_sets(base_lp),
        },
        "gradient_step_sweep": step_sweep,
        "rnea_calls": dynamics.rnea_calls,
        "interpretation": (
            "Local envelope guidance is heuristic. Stable norm under the finite-"
            "difference sweep supports derivative consistency; active-set changes "
            "make central differences differ from the one-sided local derivative."
        ),
    }
    (result_root / "dual_gradient_validation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> None:
    """Write a reproducible dual-direction audit for the development example."""
    root = Path(__file__).resolve().parents[2]
    result_root = (
        root / "research/joint_time_oracle/results/torque_lp_guided_rescue_20261108"
    )
    result = audit("trap_cluttered_08", result_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
