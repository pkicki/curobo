# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Try independent IPOPT retiming on captured native curves not ruled out by LP.

The tool preserves the captured cubic curve exactly. A successful numerical
solve is still only a candidate and requires the independent physical,
collision, and adaptive receipt workflow before it can be called feasible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

from .cohort_geometry import NativeGeometry
from .cohort_retime import path_coefficients, problem, solve, states
from .geometry_feasibility_scene import IndependentGeometryAuditor
from .torque_active import read_profile


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot encode {type(value).__name__}")


def retime_capture(
    native_arm: Path,
    profile_arm: Path,
    case: str,
    tag: str,
    *,
    grid_count: int = 257,
    maxiter: int = 300,
    initial_squared_speed: float = 0.3,
) -> dict:
    """Retiming one exact captured curve and writing a compact audit receipt."""
    if grid_count < 65 or maxiter < 1 or initial_squared_speed <= 0:
        raise ValueError("invalid fixed-path retiming configuration")
    folder = native_arm / case
    profile = read_profile(profile_arm / case / "torque_profile.json")
    archive = folder / f"{tag}_native_geometry.npz"
    if not archive.exists():
        raise FileNotFoundError(archive)
    with np.load(archive) as saved:
        buffers = {name: np.asarray(saved[name]) for name in saved.files}
    lane = 0
    geometry = NativeGeometry(buffers, lane)
    start, end = geometry.nonstationary_interval()
    native_knots = np.arange(1, buffers["controls"].shape[1] + 3) / (
        buffers["controls"].shape[1] + 3
    )
    knots = (native_knots - start) / (end - start)
    fn = geometry.evaluate_trimmed
    initial = np.full(32, initial_squared_speed, dtype=float)
    old_profile = os.environ.get("ORACLE_TORQUE_PROFILE")
    os.environ["ORACLE_TORQUE_PROFILE"] = str(
        (profile_arm / case / "torque_profile.json").resolve()
    )
    try:
        nlp = problem(fn, knots, grid_count, initial)
        controls, solver = solve(nlp, initial, maxiter)
    finally:
        if old_profile is None:
            os.environ.pop("ORACLE_TORQUE_PROFILE", None)
        else:
            os.environ["ORACLE_TORQUE_PROFILE"] = old_profile

    trajectory = states(fn, controls, count=2049, knots=knots)
    progress = trajectory["progress"]
    path = np.asarray(fn(progress))
    coefficients = path_coefficients(path)
    timing = BSpline(
        np.r_[np.zeros(3), np.linspace(0.0, 1.0, 30), np.ones(3)], controls, 3
    )
    squared_speed = timing(progress)
    torque = (
        coefficients[2]
        + coefficients[1] * squared_speed[:, None]
        + 0.5 * coefficients[0] * timing(progress, nu=1)[:, None]
    )

    base = np.load(
        Path(__file__).resolve().parents[2]
        / "research/joint_time_oracle/results/native_reconstruction_20261106/"
        "baseline_validation_states.npz"
    )
    physical = {
        family: np.asarray(base[family + "_bounds"])
        for family in ("position", "velocity", "acceleration", "jerk")
    }
    physical["torque"] = np.stack((profile["effort_lower"], profile["effort_upper"]))
    values = {
        "position": trajectory["q"],
        "velocity": trajectory["qd"],
        "acceleration": trajectory["qdd"],
        "jerk": trajectory["jerk"],
        "torque": torque,
    }
    physical_margins = {}
    scaled95_margins = {}
    utilization = {}
    for family, data in values.items():
        bounds = physical[family]
        physical_margins[family] = float(
            min(np.min(data - bounds[0]), np.min(bounds[1] - data))
        )
        midpoint = (bounds[0] + bounds[1]) / 2
        half = (bounds[1] - bounds[0]) / 2
        if family == "position":
            scaled_bounds = bounds
        else:
            scaled_bounds = np.stack((midpoint - 0.95 * half, midpoint + 0.95 * half))
        scaled95_margins[family] = float(
            min(np.min(data - scaled_bounds[0]), np.min(scaled_bounds[1] - data))
        )
        utilization[family] = float(
            np.max(np.maximum((data - midpoint) / half, (midpoint - data) / half))
        )

    tasks = json.loads((profile_arm / "tasks.json").read_text())
    task = next(row for row in tasks if row["case_id"] == case)
    endpoint_residuals = {
        "start_position_inf_rad": float(
            np.max(np.abs(trajectory["q"][0] - np.asarray(task["start"])))
        ),
        "goal_position_inf_rad": float(
            np.max(np.abs(trajectory["q"][-1] - np.asarray(task["goal"])))
        ),
        "start_velocity_inf_rad_s": float(np.max(np.abs(trajectory["qd"][0]))),
        "goal_velocity_inf_rad_s": float(np.max(np.abs(trajectory["qd"][-1]))),
        "start_acceleration_inf_rad_s2": float(np.max(np.abs(trajectory["qdd"][0]))),
        "goal_acceleration_inf_rad_s2": float(np.max(np.abs(trajectory["qdd"][-1]))),
    }
    auditor = IndependentGeometryAuditor(task["scene"])
    scene = auditor.check(trajectory["q"], min_clearance_m=0.005)
    physical_pass = all(value >= -1e-6 for value in physical_margins.values())
    scaled95_pass = all(value >= -1e-6 for value in scaled95_margins.values())
    output = {
        "case_id": case,
        "capture_tag": tag,
        "lane": lane,
        "native_geometry_archive": str(archive.resolve()),
        "native_geometry_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "torque_profile_sha256": profile["profile_sha256"],
        "geometry_exact": True,
        "retiming_config": {
            "grid_count": grid_count,
            "max_iterations": maxiter,
            "initial_squared_speed": initial_squared_speed,
            "backend": "cohort_retime IPOPT pointwise",
        },
        "solver": {
            key: solver.get(key)
            for key in (
                "status",
                "message",
                "solver_converged",
                "iterations",
                "duration_s",
                "primal_min_margin",
                "stationarity_inf",
                "collocation_points",
                "elapsed_s",
                "max_scaled_control",
            )
        },
        "sampled_duration_s": float(trajectory["time"][-1]),
        "sample_count": len(progress),
        "endpoint_residuals": endpoint_residuals,
        "physical_family_min_margins": physical_margins,
        "scaled95_family_min_margins": scaled95_margins,
        "physical_dynamic_pass": physical_pass,
        "scaled95_dynamic_pass": scaled95_pass,
        "maximum_utilization": utilization,
        "independent_scene_audit": {
            key: scene[key]
            for key in (
                "position_violation_rad",
                "world_clearance_m",
                "self_clearance_m",
                "sampled_acceptable",
            )
        },
        "qualified": False,
        "scope": (
            "Retiming diagnostic for exact captured geometry. IPOPT local "
            "infeasibility is not an infeasibility proof; sampled dynamics and "
            "collision are not continuous-time certification."
        ),
    }
    output_file = folder / f"{tag}_independent_retime.json"
    output_file.write_text(json.dumps(output, indent=2, default=_json_default) + "\n")
    np.savez_compressed(
        folder / f"{tag}_independent_retime_states.npz",
        time=trajectory["time"],
        progress=progress,
        q=trajectory["q"],
        qd=trajectory["qd"],
        qdd=trajectory["qdd"],
        jerk=trajectory["jerk"],
        torque=torque,
    )
    return output


def main() -> None:
    """Retime selected pre-TimeOpt tags and print their independent summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-arm", required=True, type=Path)
    parser.add_argument("--profile-arm", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--tag", action="append", required=True)
    parser.add_argument("--grid-count", type=int, default=257)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--initial-squared-speed", type=float, default=0.3)
    args = parser.parse_args()
    records = [
        retime_capture(
            args.native_arm,
            args.profile_arm,
            args.case,
            tag,
            grid_count=args.grid_count,
            maxiter=args.maxiter,
            initial_squared_speed=args.initial_squared_speed,
        )
        for tag in args.tag
    ]
    print(json.dumps(records, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
