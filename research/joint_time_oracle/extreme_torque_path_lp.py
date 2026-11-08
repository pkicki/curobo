# SPDX-License-Identifier: Apache-2.0
"""A necessary fixed-path torque feasibility test with an LP at each path point.

For any regular C2 path q(s), with x=s_dot**2>=0 and y=x'/2 unrestricted,
torque is g(s)+m(s)*x+a(s)*y, where
  a=M(q)*q_s
  m=M(q)*q_ss+C(q,q_s)*q_s.

If there is NO (x>=0,y real) satisfying all joint effort bounds at ANY
interior s, this specific geometric curve is infeasible for every timing law.
This is a *relaxed* pointwise necessary condition (ignoring all other
constraints); LP FEASIBILITY does NOT prove the curve has a feasible timing.

The normalized minimum slack epsilon from a linear program can provide a
robustly positive numerical infeasibility witness, never a formal
floating-point mathematical proof or a claim about ALL geometric paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from .cohort_geometry import NativeGeometry, QuinticGeometry
from .cohort_retime import path_coefficients
from .torque_active import read_profile

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "research/joint_time_oracle/results/cohort_pilot_isolated_20261106"
SOURCE = ROOT / "research/joint_time_oracle/results/torque_stress_final_20261107/severe"
STRESS = ROOT / "research/joint_time_oracle/results/extreme_torque_witness_20261108"


def pointwise_slack(
    a: np.ndarray, m: np.ndarray, g: np.ndarray, bounds: np.ndarray
) -> dict:
    """Minimum expansion of actuator bounds over x>=0, y unbounded.

    LP: min eps>=0, s.t.
       low_j - eps * halfwidth_j <= g_j + m_j*x + a_j*y
         <= high_j + eps*halfwidth_j.
    Slacks are normalized to half interval width (Nm). Positivity of the
    optimum is a relaxed fixed-geometry infeasibility witness.
    """
    a, m, g = (np.asarray(z, dtype=float) for z in (a, m, g))
    effort = np.asarray(bounds, dtype=float)
    if (
        effort.shape != (2, len(g))
        or a.shape != g.shape
        or m.shape != g.shape
        or not np.isfinite(np.r_[a, m, g, effort.ravel()]).all()
        or not np.all(effort[0] < effort[1])
    ):
        raise ValueError("invalid pointwise torque LP coefficient/bound shapes")
    lower, upper = effort
    width = (upper - lower) / 2
    lhs = np.vstack(
        (np.column_stack((m, a, -width)), np.column_stack((-m, -a, -width)))
    )
    rhs = np.r_[upper - g, g - lower]
    result = linprog(
        c=np.array([0.0, 0.0, 1.0]),
        A_ub=lhs,
        b_ub=rhs,
        bounds=[(0, None), (None, None), (0, None)],
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
        },
    )
    if not result.success or result.x is None:
        return {"status": "LP_NUMERICAL_FAILURE", "message": result.message}
    u = result.x
    violations = lhs @ u - rhs
    residual = float(np.max(violations))
    return {
        "status": "SOLVED",
        "min_relaxed_expansion": float(u[2]),
        "witness_x": float(u[0]),
        "witness_half_x_prime": float(u[1]),
        "constraint_residual_Nm": residual,
        "strongly_infeasible": bool(u[2] > 1e-3 and residual < 1e-7),
    }


def inspect_curve(geometry, bounds: np.ndarray, *, samples: int = 257) -> dict:
    """Evaluate the relaxed torque-feasibility LP along one fixed curve."""
    if samples < 3:
        raise ValueError("need >=3 interior samples")
    # Exclude progress 0/1; their prescribed rest endpoint contract is
    # checked independently. Each interior LP ignores endpoint restrictions.
    s = np.linspace(0, 1, samples + 2)[1:-1]
    path = np.asarray(geometry(s), dtype=float)
    if path.shape != (4, len(s), bounds.shape[1]):
        raise ValueError(
            "geometry must evaluate q,q_s,q_ss,q_sss with joint dimensions"
        )
    coeff = path_coefficients(path)
    checks = [
        pointwise_slack(coeff[0, i], coeff[1, i], coeff[2, i], bounds)
        for i in range(len(s))
    ]
    valid = [(i, r) for i, r in enumerate(checks) if r["status"] == "SOLVED"]
    if len(valid) != len(checks):
        return {
            "status": "LP_NUMERICAL_FAILURE",
            "solved": len(valid),
            "samples": len(checks),
        }
    k, result = max(valid, key=lambda item: item[1]["min_relaxed_expansion"])
    peak = float(result["min_relaxed_expansion"])
    return {
        "status": "SOLVED",
        "samples": len(s),
        "max_required_torque_bound_expansion": peak,
        "worst_progress": float(s[k]),
        "strong_fixed_path_infeasibility_witness": bool(result["strongly_infeasible"]),
        "worst_lp": result,
        "scope": (
            "necessary pointwise torque feasibility; positive relaxed expansion "
            "disproves this exact path under any timing, subject to numeric accuracy"
        ),
    }


def run(
    root: Path,
    source: Path,
    pilot: Path,
    *,
    regimes: list[str] | None = None,
    cases: list[str] | None = None,
    samples: int = 257,
) -> dict:
    """Run relaxed torque LP checks for registered original and alternate paths."""
    arms = regimes or sorted(
        p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("extreme_f")
    )
    source_rows = [
        json.loads(line)
        for line in (pilot / "pilot.jsonl").read_text().splitlines()
        if line.strip()
    ]
    source_task = {x["case_id"]: x for x in source_rows}
    source_coefs = {}
    outcomes = {}
    for regime in arms:
        arm = root / regime
        tasks = json.loads((arm / "tasks.json").read_text())
        outcomes[regime] = {}
        for task in tasks:
            case = task["case_id"]
            if cases is not None and case not in cases:
                continue
            try:
                profile = read_profile(arm / case / "torque_profile.json")
                bounds = np.array([profile["effort_lower"], profile["effort_upper"]])
                if case not in source_coefs:
                    with np.load(pilot / case / "a10_buffers.npz") as data:
                        buffers = {k: np.asarray(data[k]) for k in data.files}
                    lane = int(source_task[case]["methods"]["A10"]["lane"])
                    native = NativeGeometry(buffers, lane)
                    c = json.loads((source / case / "c_best.json").read_text())
                    alternative = QuinticGeometry(np.asarray(c["coefficients"]))
                    source_coefs[case] = (native, alternative)
                original, modified = source_coefs[case]
                b = inspect_curve(original.evaluate_trimmed, bounds, samples=samples)
                d = inspect_curve(modified.evaluate, bounds, samples=samples)
                out = {
                    "status": "COMPLETE",
                    "original_fixed_curobo": b,
                    "alternative_quintic": d,
                    "source_case": case,
                    "regime": regime,
                    "profile_sha256": profile["profile_sha256"],
                }
            except Exception as exc:
                out = {"status": "ERROR", "reason": repr(exc)}
            outcomes[regime][case] = out
            folder = arm / case
            (folder / "path_torque_lp.json").write_text(
                json.dumps(out, indent=2) + "\n"
            )
            print(regime, case, out["status"], flush=True)
    (root / "path_lp_summary.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    return outcomes


def main():
    """Run the fixed-path torque infeasibility diagnostic."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=STRESS)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--pilot", type=Path, default=PILOT)
    p.add_argument("--regime", action="append")
    p.add_argument("--cases", help="comma-separated frozen case IDs")
    p.add_argument("--samples", type=int, default=257)
    a = p.parse_args()
    out = run(
        a.root,
        a.source,
        a.pilot,
        regimes=a.regime,
        cases=a.cases.split(",") if a.cases else None,
        samples=a.samples,
    )
    print(
        json.dumps(
            {"arms": len(out), "tested": sum(len(v) for v in out.values())}, indent=2
        )
    )


if __name__ == "__main__":
    main()
