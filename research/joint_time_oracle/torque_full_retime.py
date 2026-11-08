# SPDX-License-Identifier: Apache-2.0
"""Matched independent IPOPT retiming of every saved full-native geometry.

This Full-R arm is essential: Full native vs D mixes TimeOpt solver quality
with geometry, while Full-R vs D uses the same fixed-path solver as B and D.
Runs offline in the Pinocchio/cyipopt environment, not the CUDA environment.
All failures are retained; output qualification is separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear

from .cohort_geometry import NativeGeometry
from .cohort_retime import path_coefficients, problem, solve, states
from .torque_active import read_profile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "research/joint_time_oracle/results/torque_stress_final_20261107"


def source_sha(path):
    """Return the SHA256 digest of a trajectory archive."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def match_geometry(
    folder: Path,
    native_name: str,
    lane: int,
    profile_file: Path,
    *,
    grid: int,
    maxiter: int,
    refinements: int,
) -> dict:
    """Optimize a fixed timing law while preserving native cubic geometry."""
    parent = native_name.split("L", 1)[0]
    with np.load(folder / (parent.lower() + "_buffers.npz")) as z:
        buffers = {k: np.asarray(z[k]) for k in z.files}
    if not 0 <= lane < len(buffers["controls"]):
        raise ValueError("invalid native lane")
    if not bool(buffers["success"][lane]):
        raise ValueError("native lane is not feasible")
    geo = NativeGeometry(buffers, lane)
    start, end = geo.nonstationary_interval()
    fn = geo.evaluate_trimmed
    count = buffers["controls"].shape[1]
    native_knots = np.arange(1, count + 3) / (count + 3)
    knots = (native_knots - start) / (end - start)
    basis = BSpline(
        np.r_[np.zeros(3), np.linspace(0, 1, 30), np.ones(3)], np.eye(32), 3
    )
    sgrid = np.linspace(0, 1, 1001)
    speed_target = (
        basis(start + (end - start) * sgrid)
        @ buffers["timing"][lane]
        / (end - start) ** 2
    )
    initial = lsq_linear(basis(sgrid), speed_target, bounds=(1e-6, np.inf)).x
    old = os.environ.get("ORACLE_TORQUE_PROFILE")
    os.environ["ORACLE_TORQUE_PROFILE"] = str(profile_file.resolve())
    controls = initial
    extras = np.array([], dtype=float)
    passes = []
    try:
        for step in range(refinements):
            p = problem(fn, knots, grid, controls, extras)
            stamp = time.perf_counter()
            candidate, info = solve(p, controls, maxiter)
            info["complete_solve_wall_s"] = time.perf_counter() - stamp
            s = states(fn, candidate, knots=knots)
            geometric = np.asarray(fn(s["progress"]))
            coeff = path_coefficients(geometric)
            time_basis = BSpline(
                np.r_[np.zeros(3), np.linspace(0, 1, 30), np.ones(3)], candidate, 3
            )
            x = time_basis(s["progress"])
            dx = time_basis(s["progress"], nu=1)
            torque = coeff[2] + coeff[1] * x[:, None] + 0.5 * coeff[0] * dx[:, None]
            all_values = np.stack((s["qd"], s["qdd"], s["jerk"], torque))
            normalized = np.abs(
                (all_values - p.center[:, None, :]) / p.width[:, None, :]
            )
            worst = float(np.max(normalized))
            info.update(
                {
                    "dense_max_scaled_utilization": worst,
                    "dense_scaled_eligible": worst <= 1.0 + 1e-6,
                    "refinement": step,
                    "dense_duration_s": float(s["time"][-1]),
                }
            )
            passes.append({k: v for k, v in info.items() if k != "history"})
            controls = candidate
            if worst <= 1.0 + 1e-6:
                break
            peak = np.unravel_index(int(np.argmax(normalized)), normalized.shape)
            near = float(s["progress"][peak[1]])
            extras = np.unique(
                np.r_[
                    extras,
                    near,
                    np.clip(near - 1e-7, 0.0, 1.0),
                    np.clip(near + 1e-7, 0.0, 1.0),
                ]
            )
        return {
            "trajectory": s,
            "controls": controls,
            "knots": knots,
            "solver": info,
            "passes": passes,
            "geometry_exact": True,
            "source_native_controls_sha256": hashlib.sha256(
                np.ascontiguousarray(buffers["controls"][lane]).tobytes()
            ).hexdigest(),
            "source_native_timing_sha256": hashlib.sha256(
                np.ascontiguousarray(buffers["timing"][lane]).tobytes()
            ).hexdigest(),
        }
    finally:
        if old is None:
            os.environ.pop("ORACLE_TORQUE_PROFILE", None)
        else:
            os.environ["ORACLE_TORQUE_PROFILE"] = old


def execute(
    arm: Path,
    *,
    case_filter: str | None,
    grid: int,
    maxiter: int,
    refinements: int,
    retry_existing: bool = False,
) -> dict:
    """Retiming all saved native candidates with the matched IPOPT solver."""
    if grid < 65 or maxiter < 1 or refinements < 1:
        raise ValueError("invalid matched retiming numerical settings")
    native = arm / "native_full"
    tasks = json.loads((arm / "tasks.json").read_text())
    output = {}
    contract = {
        "grid": grid,
        "ipopt_maxiter": maxiter,
        "refinements": refinements,
        "optimizer": "same cohort_retime.pointwise IPOPT backend as B and D",
        "notes": "independent sampled and adaptive validation required",
    }
    config = native / "matched_retime_config.json"
    if config.exists() and json.loads(config.read_text()) != contract:
        raise ValueError(
            "refusing to change matched retiming configuration mid-campaign"
        )
    config.write_text(json.dumps(contract, indent=2) + "\n")
    for task in tasks:
        case = task["case_id"]
        if case_filter and case_filter != case:
            continue
        folder = native / case
        parent_file = folder / "native_candidates.json"
        candidates = json.loads(parent_file.read_text()) if parent_file.exists() else {}
        saved = folder / "matched_retime_candidates.json"
        rows = json.loads(saved.read_text()) if saved.exists() else {}
        profile = read_profile(arm / case / "torque_profile.json")
        if retry_existing:
            history = folder / "matched_retime_attempt_history.jsonl"
            for key in candidates:
                method = "R" + key
                if method in rows:
                    with history.open("a") as stream:
                        stream.write(json.dumps(rows[method]) + "\n")
                    del rows[method]
        for key, item in sorted(candidates.items()):
            if item.get("status") != "SUCCESS":
                continue
            method = "R" + key
            if method in rows and rows[method].get("status") in ("COMPLETE", "ERROR"):
                continue
            if method in rows:
                history = folder / "matched_retime_attempt_history.jsonl"
                with history.open("a") as stream:
                    stream.write(json.dumps(rows[method]) + "\n")
            source_native = folder / (key.lower() + "_dense.npz")
            source_native_sha = source_sha(source_native)
            duplicate = next(
                (
                    name
                    for name, row in rows.items()
                    if row.get("status") == "COMPLETE"
                    and row.get("source_native_dense_sha256") == source_native_sha
                ),
                None,
            )
            if duplicate is not None:
                duplicate_archive = folder / (duplicate.lower() + "_dense.npz")
                alias_archive = folder / (method.lower() + "_dense.npz")
                with np.load(duplicate_archive) as duplicate_data:
                    np.savez_compressed(
                        alias_archive,
                        **{name: duplicate_data[name] for name in duplicate_data.files},
                    )
                source_row = rows[duplicate]
                alias = dict(source_row)
                alias.update(
                    {
                        "method": method,
                        "source_native_candidate": key,
                        "restart": item["restart"],
                        "lane": item["lane"],
                        "source_native_dense_sha256": source_native_sha,
                        "dense_sha256": source_sha(alias_archive),
                        "retimed_exact_duplicate_of": duplicate,
                        "trajectory_geometry_exact": True,
                    }
                )
                rows[method] = alias
                saved.write_text(json.dumps(rows, indent=2) + "\n")
                print(
                    case, method, "EXACT_DUPLICATE", alias.get("duration_s"), flush=True
                )
                continue
            result = {
                "method": method,
                "source_native_candidate": key,
                "restart": item["restart"],
                "lane": item["lane"],
                "torque_profile_sha256": profile["profile_sha256"],
                "source_native_dense_sha256": source_native_sha,
                "status": "ERROR",
            }
            try:
                solved = match_geometry(
                    folder,
                    key,
                    int(item["lane"]),
                    arm / case / "torque_profile.json",
                    grid=grid,
                    maxiter=maxiter,
                    refinements=refinements,
                )
                archive = folder / (method.lower() + "_dense.npz")
                np.savez_compressed(
                    archive,
                    **solved["trajectory"],
                    controls=solved["controls"],
                    knots=solved["knots"],
                )
                result.update(
                    {
                        "status": "COMPLETE",
                        "duration_s": solved["solver"]["dense_duration_s"],
                        "solver_converged": bool(solved["solver"]["solver_converged"]),
                        "dense_scaled_eligible": bool(
                            solved["solver"]["dense_scaled_eligible"]
                        ),
                        "solver_status": solved["solver"]["status"],
                        "stationarity_inf": solved["solver"]["stationarity_inf"],
                        "constraint_multipliers": solved["solver"].get(
                            "constraint_multipliers"
                        ),
                        "source_native_controls_sha256": solved[
                            "source_native_controls_sha256"
                        ],
                        "source_native_timing_sha256": solved[
                            "source_native_timing_sha256"
                        ],
                        "trajectory_geometry_exact": True,
                        "dense_sha256": source_sha(archive),
                        "passes": solved["passes"],
                    }
                )
            except Exception as exc:
                result["reason"] = repr(exc)
            rows[method] = result
            saved.write_text(json.dumps(rows, indent=2) + "\n")
            print(case, method, result["status"], result.get("duration_s"), flush=True)
        output[case] = rows
    return output


def main():
    """Run independent matched retiming for one registered stress arm."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT)
    p.add_argument("--regime", required=True)
    p.add_argument("--case")
    p.add_argument("--grid", type=int, default=513)
    p.add_argument("--maxiter", type=int, default=500)
    p.add_argument("--refinements", type=int, default=3)
    p.add_argument(
        "--retry-existing",
        action="store_true",
        help="archive and rerun prior matched retiming outputs for selected cases",
    )
    a = p.parse_args()
    rows = execute(
        a.root / a.regime,
        case_filter=a.case,
        grid=a.grid,
        maxiter=a.maxiter,
        refinements=a.refinements,
        retry_existing=a.retry_existing,
    )
    print(
        json.dumps(
            {
                "cases": len(rows),
                "computed": sum(
                    x.get("status") == "COMPLETE"
                    for r in rows.values()
                    for x in r.values()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
