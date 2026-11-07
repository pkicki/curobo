# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay torque-stress trajectories around independently observed extrema."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from curobo._src.motion.motion_trajectory_timeopt import TrajectoryTimeOpt
from curobo._src.motion.motion_trajectory_timeopt_cfg import TrajectoryTimeOptCfg
from .cohort_explore import planner_for
from .cohort_geometry import NativeGeometry, QuinticGeometry
from .cohort_retime import states
from .torque_active import read_profile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("ORACLE_COHORT_SOURCE", ROOT / "research/joint_time_oracle/results/cohort_pilot_isolated_20261106"))
DEFAULT = ROOT / "research/joint_time_oracle/results/torque_stress_final_20261107"


def _near_times(diag: dict, duration: float, samples: np.ndarray) -> np.ndarray:
    """Return grid plus shrinking neighborhoods of every reported extremum."""
    centers: list[float] = []
    for family in diag.get("families", {}).values():
        for contract in ("physical", "scaled95"):
            value = family.get(contract, {}).get("time_s")
            if value is not None:
                centers.append(float(value))
    centers.extend(float(diag[k]) for k in ("world_min_time_s", "self_min_time_s") if diag.get(k) is not None)
    offsets = np.array([-2., -1., -.25, 0., .25, 1., 2.]) * duration / 8192
    extra = [(c + offsets).clip(0, duration) for c in centers]
    return np.unique(np.r_[samples, *extra]) if extra else np.asarray(samples)


def _native_states(case: str, method: str, times: np.ndarray, task: dict,
                   row: dict, profile: dict, planner) -> dict:
    """Replay the exact stored native path and rerun its frozen TimeOpt setup."""
    source = SOURCE / case
    with np.load(source / "native_input.npz") as archive:
        inputs = {k: np.asarray(archive[k]) for k in archive.files}
    with np.load(source / "a10_buffers.npz") as archive:
        frozen = {k: np.asarray(archive[k]) for k in archive.files}
    lane = int(row["methods"]["A10"]["lane"])
    device = planner.default_joint_state.position.device
    cast = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32).contiguous()
    solver = TrajectoryTimeOpt(TrajectoryTimeOptCfg(
        robot=planner.kinematics.config, batch_size=int(frozen["controls"].shape[0]),
        n_knots=int(inputs["knot"].shape[1]), max_iterations=10 if method == "A10" else 40,
        use_cuda_graph=False, torque_lower=cast(profile["effort_lower"]),
        torque_upper=cast(profile["effort_upper"]), limit_scale=.95, jerk_limit_scale=.95))
    result = solver.timeopt(cast(inputs["knot"]), cast(frozen["start_boundary"][:, 0]),
                            cast(frozen["goal"]), cast(inputs["knot_dt"]))
    duration = float(result.duration[lane].item())
    query = np.unique(np.r_[times, _near_times({}, duration, np.array([], dtype=float))]).astype(np.float32)
    batched = torch.as_tensor(query, device=device).repeat(frozen["controls"].shape[0], 1).contiguous()
    state = solver.sample(batched)
    return {"time": query,
            "q": state.position[lane].detach().cpu().numpy().copy(),
            "qd": state.velocity[lane].detach().cpu().numpy().copy(),
            "qdd": state.acceleration[lane].detach().cpu().numpy().copy(),
            "jerk": state.jerk[lane].detach().cpu().numpy().copy()}


def replay(regime: str, root: Path, *, case_filter: str | None = None) -> int:
    """Write exact trajectory samples plus extrema neighborhoods for one regime."""
    arm = root / regime
    tasks = json.loads((arm / "tasks.json").read_text())
    original_rows = {r["case_id"]: r for r in
                     (json.loads(line) for line in (SOURCE / "pilot.jsonl").read_text().splitlines())}
    stress_rows = {r["case_id"]: r for r in
                   (json.loads(line) for line in (arm / "pilot.jsonl").read_text().splitlines())}
    diagnostics = json.loads((arm / "independent_sampled.json").read_text())
    if case_filter:
        tasks = [t for t in tasks if t["case_id"] == case_filter]
    cache: dict[str, object] = {}
    count = 0
    for task in tasks:
        case = task["case_id"]
        folder = arm / case
        profile = read_profile(folder / "torque_profile.json")
        for method in ("A10", "A40", "B", "C", "D"):
            key = f"{case}:{method}"
            diag = diagnostics.get(key)
            source = folder / f"{method.lower()}_dense.npz"
            if not source.exists() or not isinstance(diag, dict):
                continue
            with np.load(source) as saved:
                base = {k: np.asarray(saved[k]) for k in saved.files}
            duration = float(base["time"][-1])
            query = _near_times(diag, duration, np.asarray(base["time"], dtype=float))
            if method in ("A10", "A40"):
                scene = task["scene"]
                if scene not in cache:
                    cache[scene] = planner_for(scene)
                arrays = _native_states(case, method, query, task, original_rows[case], profile, cache[scene])
                # Include the independent diagnostic times exactly in the native replay.
                if not np.array_equal(arrays["time"], query.astype(np.float32)):
                    query = np.asarray(arrays["time"], dtype=float)
            elif method in ("B", "D"):
                controls = np.asarray(base["controls"], dtype=float)
                knots = np.asarray(base["knots"], dtype=float)
                geo = (NativeGeometry(np.load(SOURCE / case / "a10_buffers.npz"),
                                      int(original_rows[case]["methods"]["A10"]["lane"]))
                       if method == "B" else
                       QuinticGeometry(np.asarray(json.loads((folder / "c_best.json").read_text())["coefficients"])))
                progress = np.asarray(base["progress"], dtype=float)
                normalized = np.interp(query, base["time"], progress)
                arrays = states(geo.evaluate_trimmed if method == "B" else geo.evaluate,
                                controls, knots=knots, extra=normalized)
            else:
                best = json.loads((folder / "c_best.json").read_text())
                geo = QuinticGeometry(np.asarray(best["coefficients"]))
                duration = float(best["duration"])
                query = _near_times(diag, duration, np.asarray(base["time"], dtype=float))
                progress = np.clip(query / duration, 0., 1.)
                q, qd, qdd, jerk = geo.evaluate(progress)
                arrays = {"time": query, "q": q, "qd": qd / duration,
                          "qdd": qdd / duration**2, "jerk": jerk / duration**3}
            np.savez_compressed(folder / f"{method.lower()}_adaptive_dense.npz", **arrays)
            count += 1
        print(regime, case, "adaptive replay complete", flush=True)
    for planner in cache.values():
        planner.destroy()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", required=True, choices=("moderate", "severe"))
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--case")
    args = parser.parse_args()
    print("adaptive trajectories", replay(args.regime, args.root, case_filter=args.case))


if __name__ == "__main__":
    main()
