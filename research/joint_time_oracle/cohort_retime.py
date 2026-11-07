# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retiming exact native and quintic geometries with a common pointwise IPOPT NLP."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cyipopt
import numpy as np
import pinocchio as pin
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear

from .cohort_geometry import NativeGeometry, QuinticGeometry
from .fixed_path_pointwise import PointwiseTiming
from .torque_active import read_profile

ROOT = Path(__file__).resolve().parents[2]
PILOT = Path(
    os.environ.get(
        "ORACLE_COHORT_SOURCE",
        ROOT / "research/joint_time_oracle/results/cohort_pilot_isolated_20261106",
    )
)
RESULTS = Path(
    os.environ.get(
        "ORACLE_COHORT_RESULTS",
        ROOT / "research/joint_time_oracle/results/cohort_joint_pilot_20261106",
    )
)


def dynamics() -> tuple:
    """Build the model-aligned independent dynamics workspace."""
    model = pin.buildModelFromUrdf(
        str(ROOT / "curobo/content/assets/robot/franka_description/franka_panda.urdf")
    )
    inertia = pin.Inertia(0.01, np.zeros(3), np.eye(3) * 0.0001)
    frame = model.frames[model.getFrameId("right_gripper", pin.BODY)]
    model.inertias[frame.parentJoint] -= frame.placement.act(inertia)
    for name in ("panda_link8", "panda_leftfinger", "panda_rightfinger", "panda_hand"):
        frame = model.frames[model.getFrameId(name, pin.BODY)]
        model.inertias[frame.parentJoint] += frame.placement.act(inertia)
    state = pin.neutral(model)
    for name in ("panda_finger_joint1", "panda_finger_joint2"):
        state[model.joints[model.getJointId(name)].idx_q] = 0.04
    qi = [model.joints[model.getJointId(f"panda_joint{i}")].idx_q for i in range(1, 8)]
    vi = [model.joints[model.getJointId(f"panda_joint{i}")].idx_v for i in range(1, 8)]
    return model, model.createData(), state, qi, vi


def path_coefficients(path: np.ndarray) -> np.ndarray:
    """Evaluate affine squared-speed torque coefficients using Pinocchio RNEA."""
    model, data, state, qi, vi = dynamics()
    out = np.empty((3, path.shape[1], 7))
    for i in range(path.shape[1]):
        state[qi] = path[0, i]
        v = np.zeros(model.nv)
        a = np.zeros(model.nv)
        gravity = pin.rnea(model, data, state, v, a)[vi].copy()
        a[vi] = path[1, i]
        accel = pin.rnea(model, data, state, v, a)[vi].copy() - gravity
        v[vi] = path[1, i]
        a[vi] = path[2, i]
        moving = pin.rnea(model, data, state, v, a)[vi].copy() - gravity
        out[:, i] = accel, moving, gravity
    return out


def problem(
    geometry,
    knots: np.ndarray,
    grid_count: int,
    initial: np.ndarray,
    extra: np.ndarray | None = None,
) -> PointwiseTiming:
    """Build pointwise constraints for either exact geometry under one contract."""
    eps = 1e-8
    interior = knots[(knots > eps) & (knots < 1 - eps)]
    grid = np.unique(
        np.r_[
            np.linspace(0, 1, grid_count),
            interior - eps,
            interior,
            interior + eps,
            [] if extra is None else extra,
        ]
    )
    p = object.__new__(PointwiseTiming)
    p.grid = grid
    p.path = np.asarray(geometry(grid))
    p.coeff = path_coefficients(p.path)
    p.scale = float(np.median(initial))
    basis = BSpline(
        np.r_[np.zeros(3), np.linspace(0, 1, 30), np.ones(3)], np.eye(32), 3
    )
    p.b = [basis(grid, nu=i) * p.scale for i in range(3)]
    nodes, weights = np.polynomial.legendre.leggauss(6)
    quadrature = (np.arange(511)[:, None] + (nodes[None, :] + 1) / 2) / 511
    p.q = basis(quadrature.ravel()) * p.scale
    p.qw = np.tile(weights / (2 * 511), 511)
    bounds = np.load(
        ROOT
        / "research/joint_time_oracle/results/native_reconstruction_20261106/baseline_validation_states.npz"
    )
    raw = np.stack(
        [
            bounds[name + "_bounds"]
            for name in ("velocity", "acceleration", "jerk", "torque")
        ]
    )
    profile_path = os.environ.get("ORACLE_TORQUE_PROFILE")
    if profile_path:
        profile = read_profile(profile_path)
        expected_nominal = np.stack(
            (profile["nominal_effort_lower"], profile["nominal_effort_upper"])
        )
        if not np.allclose(raw[3], expected_nominal, rtol=1e-5, atol=1e-5):
            raise ValueError(
                "frozen nominal effort limits differ from current Pinocchio reference"
            )
        raw[3] = np.stack((profile["effort_lower"], profile["effort_upper"]))
    p.center = (raw[:, 1] + raw[:, 0]) / 2
    p.width = (raw[:, 1] - raw[:, 0]) * 0.475
    # TimeOpt scales asymmetric torque bounds toward zero; the other families are symmetric.
    p.center[3] *= 0.95
    return p


def solve(
    p: PointwiseTiming, initial: np.ndarray, maxiter: int
) -> tuple[np.ndarray, dict]:
    """Solve the bounded-below, upper-unbounded nonlinear program."""
    x0 = initial / p.scale
    history = []
    support = np.any(np.stack([np.abs(b) > 0 for b in p.b]), axis=0)
    pattern = np.broadcast_to(
        support[None, :, None, :], (4, len(support), 7, 32)
    ).reshape(-1, 32)
    rows, cols = np.nonzero(np.concatenate((pattern, pattern)))

    class Problem:
        def objective(self, x):
            return p.objective(x)

        def gradient(self, x):
            return p.gradient(x)

        def constraints(self, x):
            return p.values_jac(x)[0]

        def jacobianstructure(self):
            return rows, cols

        def jacobian(self, x):
            return p.values_jac(x)[1][rows, cols]

        def intermediate(self, *args):
            history.append(list(args[:8]))

    n = len(p.values_jac(x0)[0])
    engine = cyipopt.Problem(
        n=32,
        m=n,
        problem_obj=Problem(),
        lb=np.full(32, 1e-8),
        ub=np.full(32, 1e19),
        cl=np.zeros(n),
        cu=np.full(n, 1e19),
    )
    for key, value in [
        ("hessian_approximation", "limited-memory"),
        ("max_iter", maxiter),
        ("tol", 1e-7),
        ("print_level", 0),
    ]:
        engine.add_option(key, value)
    stamp = time.perf_counter()
    x, r = engine.solve(x0)
    elapsed = time.perf_counter() - stamp
    jac = p.values_jac(x)[1]
    stationarity = p.gradient(x) + jac.T @ r["mult_g"] - r["mult_x_L"] + r["mult_x_U"]
    # Pointwise constraints are [lower(velocity,accel,jerk,torque), upper(...)]
    # with each family comprising len(grid) x 7 entries. Save summary multipliers,
    # not large full Jacobians, for subsequent active-constraint analysis.
    if np.asarray(r["mult_g"]).size != 2 * 4 * len(p.grid) * 7:
        raise ValueError("unexpected IPOPT multiplier ordering/size")
    dual = np.abs(np.asarray(r["mult_g"], dtype=np.float64)).reshape(
        2, 4, len(p.grid), 7
    )
    signed_dual = np.asarray(r["mult_g"], dtype=np.float64).reshape(
        2, 4, len(p.grid), 7
    )
    names = ("velocity", "acceleration", "jerk", "torque")
    dual_summary = {
        name: {
            "max_abs": float(dual[:, i].max()),
            "sum_abs": float(dual[:, i].sum()),
            "signed_sum": float(signed_dual[:, i].sum()),
            "lower_side_sum_abs": float(dual[0, i].sum()),
            "upper_side_sum_abs": float(dual[1, i].sum()),
            "lower_side_signed_sum": float(signed_dual[0, i].sum()),
            "upper_side_signed_sum": float(signed_dual[1, i].sum()),
            "count_gt_1e_5": int(np.sum(dual[:, i] > 1e-5)),
        }
        for i, name in enumerate(names)
    }
    info = {
        "constraint_multipliers": dual_summary,
        "multiplier_contract": {
            "constraint_order": "lower then upper; velocity, acceleration, jerk, torque",
            "constraint_scaling": "dimensionless pointwise margins normalized by 0.95 half-range",
            "reported_sign": "raw IPOPT mult_g sign retained; absolute summaries aid magnitude comparison",
            "interpretation": "local sensitivity only; multipliers are not global optimality evidence",
        },
        "status": int(r["status"]),
        "message": r["status_msg"].decode(errors="replace"),
        "solver_converged": int(r["status"]) == 0,
        "iterations": len(history),
        "duration_s": p.objective(x),
        "primal_min_margin": float(p.values_jac(x)[0].min()),
        "collocation_points": len(p.grid),
        "stationarity_inf": float(np.max(np.abs(stationarity))),
        "max_scaled_control": float(x.max()),
        "upper_control_bound": None,
        "lower_control_bound": 1e-8,
        "lower_control_bound_active": bool(x.min() < 1.01e-8),
        "elapsed_s": elapsed,
        "history": history,
    }
    return x * p.scale, info


def states(
    geometry,
    controls: np.ndarray,
    count: int = 8193,
    knots: np.ndarray | None = None,
    extra: np.ndarray | None = None,
) -> dict:
    """Reconstruct the exact geometry and variable timing at dense path samples."""

    def unique_progress(values: np.ndarray, tolerance: float = 1e-14) -> np.ndarray:
        ordered = np.sort(np.asarray(values, dtype=float))
        if len(ordered) < 2:
            return ordered
        keep = np.r_[True, np.diff(ordered) > tolerance]
        return ordered[keep]

    s = np.linspace(0, 1, count)
    if knots is not None:
        interior = knots[(knots > 1e-9) & (knots < 1 - 1e-9)]
        s = unique_progress(np.r_[s, interior - 1e-9, interior, interior + 1e-9])
    if extra is not None:
        s = unique_progress(np.r_[s, np.clip(extra, 0, 1)])
    basis = BSpline(np.r_[np.zeros(3), np.linspace(0, 1, 30), np.ones(3)], controls, 3)

    def evaluate(progress):
        path = np.asarray(geometry(progress))
        x, dx, ddx = (basis(progress, nu=i) for i in range(3))
        speed = np.sqrt(x)[:, None]
        qd = path[1] * speed
        qdd = path[2] * x[:, None] + 0.5 * path[1] * dx[:, None]
        jerk = speed * (
            path[3] * x[:, None]
            + 1.5 * path[2] * dx[:, None]
            + 0.5 * path[1] * ddx[:, None]
        )
        coeff = path_coefficients(path)
        tau = coeff[2] + coeff[1] * x[:, None] + 0.5 * coeff[0] * dx[:, None]
        return path[0], qd, qdd, jerk, tau

    coarse = evaluate(s)
    extra = []
    for values in coarse:
        for joint in range(values.shape[1]):
            for index in (
                int(np.argmin(values[:, joint])),
                int(np.argmax(values[:, joint])),
            ):
                extra.extend(
                    np.clip(
                        s[index] + np.array([-1.0, -0.5, 0.5, 1.0]) / (count - 1), 0, 1
                    )
                )
    s = unique_progress(np.r_[s, extra])
    q, qd, qdd, jerk, _ = evaluate(s)
    nodes, weights = np.polynomial.legendre.leggauss(6)
    length = np.diff(s)
    sample = s[:-1, None] + length[:, None] * (nodes[None, :] + 1) / 2
    times = np.r_[
        0,
        np.cumsum(
            length * np.sum(weights[None, :] / 2 / np.sqrt(basis(sample)), axis=1)
        ),
    ]
    return {"time": times, "progress": s, "q": q, "qd": qd, "qdd": qdd, "jerk": jerk}


def run_case(case: str, method: str, grid: int, maxiter: int) -> None:
    """Run B or D using frozen task artifacts and retain all solver evidence."""
    row = next(
        json.loads(line)
        for line in (PILOT / "pilot.jsonl").read_text().splitlines()
        if json.loads(line)["case_id"] == case
    )
    folder = PILOT / case
    dest = RESULTS / case
    dest.mkdir(parents=True, exist_ok=True)
    profile_path = os.environ.get("ORACLE_TORQUE_PROFILE")
    if profile_path and read_profile(profile_path)["source_case_id"] != case:
        raise ValueError("torque profile belongs to another source task")
    if method == "B":
        buffers = np.load(folder / "a10_buffers.npz")
        lane = row["methods"]["A10"]["lane"]
        geo = NativeGeometry(buffers, lane)
        start, end = geo.nonstationary_interval()
        fn = geo.evaluate_trimmed
        count = buffers["controls"].shape[1]
        native_knots = np.arange(1, count + 3) / (count + 3)
        knots = (native_knots - start) / (end - start)
        speed = buffers["timing"][lane]
        sgrid = np.linspace(0, 1, 1001)
        basis = BSpline(
            np.r_[np.zeros(3), np.linspace(0, 1, 30), np.ones(3)], np.eye(32), 3
        )
        # Represent the native timing law after the affine geometric coordinate change.
        target = basis(start + (end - start) * sgrid) @ speed / (end - start) ** 2
        init = lsq_linear(basis(sgrid), target, bounds=(1e-6, np.inf)).x
        metadata = {
            "trim_interval": [start, end],
            "seed_squared_speed_fit_max_abs": float(
                np.max(np.abs(basis(sgrid) @ init - target))
            ),
        }
    else:
        c = json.loads((dest / "c_best.json").read_text())
        geo = QuinticGeometry(np.asarray(c["coefficients"]))
        fn = geo.evaluate
        knots = np.unique(geo.spline.t)
        init = np.full(32, 1 / float(c["duration"]) ** 2)
        metadata = {"geometry": "exact quintic controls"}
    extras = np.array([])
    passes = []
    controls = init
    if profile_path:
        metadata["torque_profile_sha256"] = read_profile(profile_path)["profile_sha256"]
    for refinement in range(3):
        p = problem(fn, knots, grid, controls, extras)
        endpoint = np.max(np.abs(p.path[1:3, :, :][:, [0, -1], :]))
        if endpoint > 1e-7:
            raise ValueError(
                f"path derivatives incompatible with rest endpoints: {endpoint}"
            )
        controls, info = solve(p, controls, maxiter)
        info.update(metadata)
        q = states(fn, controls, knots=knots)
        info["dense_duration_s"] = float(q["time"][-1])
        info["dense_samples"] = len(q["time"])
        path = np.asarray(fn(q["progress"]))
        coeff = path_coefficients(path)
        basis = BSpline(
            np.r_[np.zeros(3), np.linspace(0, 1, 30), np.ones(3)], controls, 3
        )
        x = basis(q["progress"])
        dx = basis(q["progress"], nu=1)
        torque = coeff[2] + coeff[1] * x[:, None] + 0.5 * coeff[0] * dx[:, None]
        values = np.stack((q["qd"], q["qdd"], q["jerk"], torque))
        margins = 1 - np.abs((values - p.center[:, None, :]) / p.width[:, None, :])
        info["dense_min_margin"] = float(margins.min())
        info["refinement"] = refinement
        passes.append(info)
        (dest / f"{method.lower()}_grid{grid}_pass{refinement}.json").write_text(
            json.dumps(info, indent=2) + "\n"
        )
        np.savez_compressed(
            dest / f"{method.lower()}_grid{grid}_pass{refinement}.npz",
            **q,
            controls=controls,
            knots=knots,
        )
        if margins.min() >= -1e-6:
            break
        indices = np.unique(np.argmin(margins, axis=1).ravel())
        newpoints = q["progress"][indices]
        extras = np.unique(
            np.r_[
                extras,
                newpoints,
                np.clip(newpoints - 1e-8, 0, 1),
                np.clip(newpoints + 1e-8, 0, 1),
            ]
        )
    info["refinement_passes"] = [
        {k: v for k, v in item.items() if k != "history"} for item in passes
    ]
    if info["dense_min_margin"] < 0:
        factors = [1.0]
        for family, order in enumerate((1, 2, 3)):
            normalized = np.abs((values[family] - p.center[family]) / p.width[family])
            factors.append(float(np.max(normalized)) ** (1 / order))
        factor = max(factors) * 1.0001
        for _ in range(8):
            restored = states(fn, controls / factor**2, knots=knots)
            restored_path = np.asarray(fn(restored["progress"]))
            restored_coeff = path_coefficients(restored_path)
            restored_basis = BSpline(
                np.r_[np.zeros(3), np.linspace(0, 1, 30), np.ones(3)],
                controls / factor**2,
                3,
            )
            rx = restored_basis(restored["progress"])
            rdx = restored_basis(restored["progress"], nu=1)
            rtau = (
                restored_coeff[2]
                + restored_coeff[1] * rx[:, None]
                + 0.5 * restored_coeff[0] * rdx[:, None]
            )
            rv = np.stack((restored["qd"], restored["qdd"], restored["jerk"], rtau))
            margin = float(
                (1 - np.abs((rv - p.center[:, None, :]) / p.width[:, None, :])).min()
            )
            if margin >= 0:
                break
            factor *= 1.005
        info["restoration"] = {
            "time_factor": factor,
            "dense_min_margin": margin,
            "duration_s": float(restored["time"][-1]),
            "sampled_scaled_feasible": margin >= 0,
        }
        np.savez_compressed(
            dest / f"{method.lower()}_restored_dense.npz",
            **restored,
            controls=controls / factor**2,
            knots=knots,
        )
    np.savez_compressed(
        dest / f"{method.lower()}_grid{grid}.npz", **q, controls=controls, knots=knots
    )
    (dest / f"{method.lower()}_grid{grid}.json").write_text(
        json.dumps(info, indent=2) + "\n"
    )
    np.savez_compressed(
        dest / f"{method.lower()}_dense.npz", **q, controls=controls, knots=knots
    )
    (dest / f"{method.lower()}_attempt.json").write_text(
        json.dumps(info, indent=2) + "\n"
    )
    print(
        case,
        method,
        json.dumps({k: v for k, v in info.items() if k != "history"}),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("method", choices=["B", "D"])
    parser.add_argument("--grid", type=int, default=513)
    parser.add_argument("--maxiter", type=int, default=150)
    args = parser.parse_args()
    run_case(args.case, args.method, args.grid, args.maxiter)


if __name__ == "__main__":
    main()
