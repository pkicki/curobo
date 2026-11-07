# SPDX-License-Identifier: Apache-2.0
"""Native A10/A40 on frozen geometry with explicit derated effort limits.

This is a retiming stress test, NOT a rerun of the geometric motion planner.
It reuses the exact original TrajOpt interpolation input and robot inertials;
the only changed physical constraint is the frozen actuator torque interval.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from curobo._src.motion.motion_trajectory_timeopt import TrajectoryTimeOpt
from curobo._src.motion.motion_trajectory_timeopt_cfg import TrajectoryTimeOptCfg
from .cohort_explore import planner_for, sample
from .torque_active import read_profile

ROOT = Path(__file__).resolve().parents[2]
PILOT = Path(os.environ.get("ORACLE_COHORT_SOURCE", ROOT / "research/joint_time_oracle/results/cohort_pilot_isolated_20261106"))
STRESS = ROOT / "research/joint_time_oracle/results/torque_stress_20261107"


def execute(regime: str, output: Path, case_filter: str | None = None) -> list[dict]:
    arm = output / regime
    tasks = json.loads((arm / "tasks.json").read_text())
    original = {r["case_id"]: r for r in
                (json.loads(s) for s in (PILOT / "pilot.jsonl").read_text().splitlines())}
    if case_filter:
        tasks = [t for t in tasks if t["case_id"] == case_filter]
        if not tasks:
            raise ValueError("case not admitted in this regime: " + case_filter)
    cache: dict[str | None, object] = {}
    existing = {}
    previous = arm / "pilot.jsonl"
    if previous.exists():
        existing = {r["case_id"]: r for r in (json.loads(x) for x in previous.read_text().splitlines())}
    for task in tasks:
        case, scene = task["case_id"], task["scene"]
        dest = arm / case
        dest.mkdir(parents=True, exist_ok=True)
        row = {"case_id": case, "cohort": "torque_exploratory",
               "population": task["population"], "regime": regime,
               "source_task_sha256": task["task_sha256"],
               "methods": {}, "errors": []}
        try:
            profile = read_profile(dest / "torque_profile.json")
            if profile["source_case_id"] != case or profile["regime"] != regime:
                raise ValueError("inconsistent case/regime/profile")
            row["torque_profile_sha256"] = profile["profile_sha256"]
            src = PILOT / case
            with np.load(src / "native_input.npz") as a:
                inputs = {k: np.asarray(a[k]) for k in a.files}
            with np.load(src / "a10_buffers.npz") as a:
                frozen = {k: np.asarray(a[k]) for k in a.files}
            lane = int(original[case]["methods"]["A10"]["lane"])
            if scene not in cache:
                cache[scene] = planner_for(scene)
            planner = cache[scene]
            device = planner.default_joint_state.position.device
            cast = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32).contiguous()
            count = int(frozen["controls"].shape[0])
            if inputs["knot"].shape[0] != count:
                raise ValueError("native interpolation lanes mismatch")
            for name, iterations in (("A10", 10), ("A40", 40)):
                cfg = TrajectoryTimeOptCfg(
                    robot=planner.kinematics.config,
                    batch_size=count, n_knots=int(inputs["knot"].shape[1]),
                    max_iterations=iterations, use_cuda_graph=False,
                    torque_lower=cast(profile["effort_lower"]),
                    torque_upper=cast(profile["effort_upper"]),
                    limit_scale=.95, jerk_limit_scale=.95)
                solver = TrajectoryTimeOpt(cfg)
                torch.cuda.synchronize()
                started = time.perf_counter()
                result = solver.timeopt(
                    cast(inputs["knot"]),
                    cast(frozen["start_boundary"][:, 0]),
                    cast(frozen["goal"]),
                    cast(inputs["knot_dt"]))
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                geom_error = max(
                    float(np.max(np.abs(getattr(solver, attr).detach().cpu().numpy() - frozen[key])))
                    for key, attr in (("controls", "_controls"),
                                      ("start_boundary", "_start_boundary"),
                                      ("goal", "_goal")))
                if geom_error > 1e-6:
                    raise RuntimeError(f"frozen geometry changed: {geom_error}")
                duration = float(result.duration[lane].item())
                success = bool(result.success[lane].item())
                status = int(result.status[lane].item())
                row["methods"][name] = {
                    "native_success": success, "native_status": status,
                    "duration_s": duration if success and math.isfinite(duration) and duration > 0 else None,
                    "geometry_error_rad": geom_error, "wall_s": elapsed,
                    "solver_converged": False, "verification": {}}
                np.savez_compressed(dest / (name.lower() + "_buffers.npz"),
                                    timing=result.timing_controls.detach().cpu().numpy().copy(),
                                    times=result.time.detach().cpu().numpy().copy(),
                                    status=result.status.detach().cpu().numpy().copy(),
                                    success=result.success.detach().cpu().numpy().copy())
                if row["methods"][name]["duration_s"] is not None:
                    np.savez_compressed(dest / (name.lower() + "_dense.npz"),
                                        **sample(solver, lane, duration, 8193))
                else:
                    row["errors"].append({"stage": name,
                                          "reason": "native retimer returned no feasible lane"})
        except Exception as exc:
            row["errors"].append({"stage": "execution", "exception": repr(exc),
                                  "traceback": traceback.format_exc()})
        existing[case] = row
        previous.write_text("".join(json.dumps(existing[k])+"\n" for k in sorted(existing)))
        print(regime, case, {x: y.get("duration_s") for x, y in row["methods"].items()},
              row["errors"], flush=True)
    for planner in cache.values():
        planner.destroy()
    return list(existing.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", required=True, choices=("moderate", "severe"))
    parser.add_argument("--output", type=Path, default=STRESS)
    parser.add_argument("--case")
    args = parser.parse_args()
    execute(args.regime, args.output, args.case)


if __name__ == "__main__":
    main()
