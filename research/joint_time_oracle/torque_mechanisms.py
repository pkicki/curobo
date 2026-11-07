# SPDX-License-Identifier: Apache-2.0
"""Independent Pinocchio torque decomposition for frozen B versus D geometry.

Explains which torque terms differ; does NOT establish a causal attribution.
Outputs are diagnostics, not continuous-time feasibility certificates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .torque_active import read_profile, torque_utilization

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "research/joint_time_oracle/results/torque_stress_final_20261107"
Rnea = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def decompose_rnea(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    rnea: Rnea,
) -> dict[str, np.ndarray]:
    """Separate M(q)qdd, C(q,qd)qd, and g(q) using RNEA linearity.

    rnea(q,v,a) returns the seven controlled-joint torques; external forces,
    friction and actuation saturation are NOT included. The zero-velocity
    inertia call removes velocity-dependent bias exactly for a rigid-body
    model. The velocity call isolates its velocity-dependent bias.
    """
    q, v, a = (
        np.asarray(t, dtype=np.float64) for t in (positions, velocities, accelerations)
    )
    if q.ndim != 2 or q.shape != v.shape or q.shape != a.shape:
        raise ValueError("q, velocity and acceleration must be matching [N,dof] arrays")
    if not all(np.isfinite(x).all() for x in (q, v, a)):
        raise ValueError("nonfinite robot states")
    n, dof = q.shape
    zero = np.zeros(dof, dtype=np.float64)
    gravity = np.empty_like(q)
    inertia = np.empty_like(q)
    velocity_bias = np.empty_like(q)
    total = np.empty_like(q)
    for k in range(n):
        g = np.asarray(rnea(q[k], zero, zero), dtype=np.float64)
        mi = np.asarray(rnea(q[k], zero, a[k]), dtype=np.float64)
        cv = np.asarray(rnea(q[k], v[k], zero), dtype=np.float64)
        tau = np.asarray(rnea(q[k], v[k], a[k]), dtype=np.float64)
        if any(x.shape != (dof,) for x in (g, mi, cv, tau)):
            raise ValueError("RNEA callback returned incorrect torque shape")
        gravity[k], inertia[k], velocity_bias[k], total[k] = g, mi - g, cv - g, tau
    residual = total - (inertia + velocity_bias + gravity)
    if not np.isfinite(residual).all():
        raise ValueError("nonfinite inverse-dynamics decomposition")
    return dict(
        total=total,
        inertia=inertia,
        velocity_bias=velocity_bias,
        gravity=gravity,
        identity_residual=residual,
    )


def common_physical_times(trajectory: dict, *, count: int = 501) -> dict:
    """Sample both solutions by *normalized physical time*, not path progress."""
    t = np.asarray(trajectory["time"], dtype=np.float64)
    if t.ndim != 1 or len(t) < 3 or np.any(np.diff(t) < 0):
        raise ValueError("trajectory timestamps must be monotonically increasing")
    if count < 3:
        raise ValueError("need at least three samples")
    arrays = {}
    for name in ("q", "qd", "qdd"):
        value = np.asarray(trajectory[name], dtype=np.float64)
        if value.ndim != 2 or value.shape[0] != len(t):
            raise ValueError("invalid trajectory array " + name)
        arrays[name] = value
    # Adaptive verification may insert an extra point exactly on an existing
    # sample. np.interp expects increasing abscissae; collapse such duplicates
    # only when they represent the same physical state to numerical precision.
    unique, first, counts = np.unique(t, return_index=True, return_counts=True)
    for name, value in arrays.items():
        for index, count_at_time in zip(first, counts):
            if count_at_time > 1:
                duplicate = value[index : index + count_at_time]
                if not np.allclose(duplicate, duplicate[0], rtol=1e-10, atol=1e-10):
                    raise ValueError(
                        "duplicate timestamps have inconsistent states in " + name
                    )
        arrays[name] = value[first]
    t = unique
    if len(t) < 3:
        raise ValueError("trajectory needs at least three distinct physical times")
    u = np.linspace(0.0, 1.0, count)
    t_query = t[0] + u * (t[-1] - t[0])
    result = {"normalized_time": u, "time": t_query}
    for name, x in arrays.items():
        result[name] = np.column_stack(
            [np.interp(t_query, t, x[:, j]) for j in range(x.shape[1])]
        )
    return result


def metrics(
    components: dict[str, np.ndarray],
    bounds: np.ndarray,
    q: np.ndarray,
    duration: float,
) -> dict:
    total = components["total"]
    utilization = torque_utilization(total, bounds)
    k, j = np.unravel_index(int(np.argmax(utilization)), utilization.shape)
    family_names = ("inertia", "velocity_bias", "gravity")
    lim = float(bounds[1, j] if total[k, j] >= 0 else -bounds[0, j])
    terms = {name: float(components[name][k, j]) for name in family_names}
    return {
        "duration_s": float(duration),
        "max_torque_utilization": float(utilization[k, j]),
        "max_torque_joint": int(j + 1),
        "max_torque_normalized_time": float(k / (len(utilization) - 1)),
        "max_torque_Nm": float(total[k, j]),
        "torque_limit_magnitude_Nm": lim,
        "peak_torque_components_Nm": terms,
        "peak_signed_component_utilization": {
            key: val / lim for key, val in terms.items()
        },
        "rms_components_Nm": {
            name: float(np.sqrt(np.mean(components[name] ** 2)))
            for name in (*family_names, "total")
        },
        "mean_max_joint_utilization": float(np.mean(np.max(utilization, axis=1))),
        "fraction_of_samples_above_85pct": float(
            np.mean(np.any(utilization >= 0.85, axis=1))
        ),
        "fraction_of_samples_above_95pct": float(
            np.mean(np.any(utilization >= 0.95, axis=1))
        ),
        "joint_space_path_length_rad": float(
            np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1))
        ),
        "rnea_identity_max_error_Nm": float(
            np.max(np.abs(components["identity_residual"]))
        ),
    }


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_case(arm: Path, case: str, *, count: int = 501) -> dict:
    """Do not overwrite historic trajectories or validation receipts."""
    folder = arm / case
    profile_file = folder / "torque_profile.json"
    profile = read_profile(profile_file)
    if profile["source_case_id"] != case:
        raise ValueError("torque profile/source case mismatch")
    bounds = np.array([profile["effort_lower"], profile["effort_upper"]])
    from .cohort_retime import dynamics

    model, data, state, qi, vi = dynamics()
    zero = np.zeros(model.nv)

    def rnea(q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        state[qi] = q
        zero[vi] = v
        accel = np.zeros(model.nv)
        accel[vi] = a
        import pinocchio as pin

        return pin.rnea(model, data, state, zero, accel)[vi].copy()

    reports = {}
    npz = {}
    files = {"torque_profile": sha(profile_file)}
    for method in ("B", "D"):
        path = folder / (method.lower() + "_dense.npz")
        if not path.exists():
            reports[method] = {"status": "MISSING_TRAJECTORY"}
            continue
        files[method + "_dense"] = sha(path)
        with np.load(path) as z:
            traj = {key: z[key] for key in ("time", "q", "qd", "qdd")}
        sampled = common_physical_times(traj, count=count)
        result = decompose_rnea(sampled["q"], sampled["qd"], sampled["qdd"], rnea)
        m = metrics(result, bounds, sampled["q"], traj["time"][-1] - traj["time"][0])
        audit_path = folder / (method.lower() + "_pin_torque.npz")
        if audit_path.exists():
            with np.load(audit_path) as audit:
                audit_time = np.asarray(audit["time"], dtype=np.float64)
                audit_torque = np.asarray(audit["torque"], dtype=np.float64)
            cached = np.column_stack(
                [
                    np.interp(sampled["time"], audit_time, audit_torque[:, j])
                    for j in range(audit_torque.shape[1])
                ]
            )
            m["previous_pin_audit_max_difference_Nm"] = float(
                np.max(np.abs(result["total"] - cached))
            )
            files[method + "_pin_torque"] = sha(audit_path)
        if m["rnea_identity_max_error_Nm"] > 1e-7:
            raise ValueError("Pinocchio decomposition identity failed")
        reports[method] = {"status": "DECOMPOSED", "metrics": m}
        for k, x in (*sampled.items(), *result.items()):
            npz[method.lower() + "_" + k] = x
    contrast = {}
    if all(reports.get(m, {}).get("status") == "DECOMPOSED" for m in ("B", "D")):
        b, d = reports["B"]["metrics"], reports["D"]["metrics"]
        contrast = {
            "B_to_D_improvement_pct": 100
            * (b["duration_s"] - d["duration_s"])
            / b["duration_s"],
            "path_length_increase_pct": 100
            * (d["joint_space_path_length_rad"] / b["joint_space_path_length_rad"] - 1),
            "peak_torque_utilization_delta_D_minus_B": d["max_torque_utilization"]
            - b["max_torque_utilization"],
            "gravity_rms_delta_Nm": d["rms_components_Nm"]["gravity"]
            - b["rms_components_Nm"]["gravity"],
            "inertia_rms_delta_Nm": d["rms_components_Nm"]["inertia"]
            - b["rms_components_Nm"]["inertia"],
            "velocity_bias_rms_delta_Nm": d["rms_components_Nm"]["velocity_bias"]
            - b["rms_components_Nm"]["velocity_bias"],
        }
    out = arm / "mechanisms" / case
    out.mkdir(parents=True, exist_ok=True)
    if npz:
        np.savez_compressed(out / "torque_components.npz", **npz)
    doc = {
        "case_id": case,
        "regime": profile["regime"],
        "profile_sha256": profile["profile_sha256"],
        "input_sha256": files,
        "sampling": "uniform normalized physical time; interpolation diagnostics",
        "physical_model": "aligned Pinocchio rigid-body RNEA; no payload or joint friction",
        "methods": reports,
        "contrast": contrast,
        "interpretation": "descriptive physical decomposition; not a causal explanation or a certificate",
    }
    (out / "analysis.json").write_text(json.dumps(doc, indent=2) + "\n")
    return doc


def path_coordinate_case(arm: Path, case: str, *, count: int = 501) -> dict:
    """Analyze the torque-limited path acceleration interval for frozen B and D."""
    from scipy.interpolate import BSpline

    from .cohort_geometry import NativeGeometry, QuinticGeometry
    from .cohort_retime import dynamics, path_coefficients
    from .torque_active import torque_utilization

    source = ROOT / "research/joint_time_oracle/results/cohort_pilot_isolated_20261106"
    source_tasks = {
        row["case_id"]: row
        for row in (
            json.loads(line)
            for line in (source / "pilot.jsonl").read_text().splitlines()
        )
    }
    if case not in source_tasks:
        raise ValueError("missing original source trajectory metadata: " + case)
    source_folder = source / case
    arm_folder = arm / case
    profile = read_profile(arm_folder / "torque_profile.json")
    bounds = np.stack((profile["effort_lower"], profile["effort_upper"]))
    s = np.linspace(0.0, 1.0, count)
    basis_knots = np.r_[np.zeros(3), np.linspace(0.0, 1.0, 30), np.ones(3)]
    result = {"case_id": case, "regime": profile["regime"], "methods": {}}
    files = {"torque_profile_sha256": sha(arm_folder / "torque_profile.json")}
    model, data, state, qi, vi = dynamics()
    zero_velocity = np.zeros(model.nv)
    zero_acceleration = np.zeros(model.nv)
    for method in ("B", "D"):
        if method == "B":
            with np.load(source_folder / "a10_buffers.npz") as archive:
                native_buffers = {
                    key: np.asarray(archive[key]) for key in archive.files
                }
            lane = int(source_tasks[case]["methods"]["A10"]["lane"])
            geometry = NativeGeometry(native_buffers, lane)
            start, end = geometry.nonstationary_interval()
            geometry_fn = geometry.evaluate_trimmed
            knots = np.arange(1, native_buffers["controls"].shape[1] + 3)
            knots = (knots / (native_buffers["controls"].shape[1] + 3) - start) / (
                end - start
            )
        else:
            c = json.loads((arm_folder / "c_best.json").read_text())
            geometry = QuinticGeometry(np.asarray(c["coefficients"]))
            geometry_fn = geometry.evaluate
            knots = np.unique(geometry.spline.t)
        trajectory_path = arm_folder / (method.lower() + "_dense.npz")
        if not trajectory_path.exists():
            result["methods"][method] = {"status": "MISSING_TRAJECTORY"}
            continue
        with np.load(trajectory_path) as archive:
            timing = np.asarray(archive["controls"], dtype=np.float64)
            duration = float(np.asarray(archive["time"])[-1])
        speed = BSpline(basis_knots, timing, 3, extrapolate=False)
        q, qs, qss, qsss = (np.asarray(value) for value in geometry_fn(s))
        x = speed(s)
        xp = speed(s, nu=1)
        coefficient = path_coefficients(np.stack((q, qs, qss, qsss)))
        a = coefficient[0]
        m = coefficient[1]
        gravity = coefficient[2]
        inertial_component = m * x[:, None]
        acceleration_component = 0.5 * a * xp[:, None]
        reconstructed = gravity + inertial_component + acceleration_component
        qd = qs * np.sqrt(np.maximum(x, 0.0))[:, None]
        qdd = qss * x[:, None] + 0.5 * qs * xp[:, None]
        direct = np.empty_like(reconstructed)
        for index in range(count):
            state[qi] = q[index]
            velocity = zero_velocity.copy()
            acceleration = zero_acceleration.copy()
            velocity[vi] = qd[index]
            acceleration[vi] = qdd[index]
            import pinocchio as pin

            direct[index] = pin.rnea(model, data, state, velocity, acceleration)[vi]

        coefficient_xprime = 0.5 * a
        lower = np.full_like(a, -np.inf)
        upper = np.full_like(a, np.inf)
        eps = 1e-9
        for j in range(a.shape[1]):
            active = np.abs(coefficient_xprime[:, j]) > eps
            lhs = bounds[0, j] - gravity[:, j] - inertial_component[:, j]
            rhs = bounds[1, j] - gravity[:, j] - inertial_component[:, j]
            lo = np.empty(count)
            hi = np.empty(count)
            lo[active] = lhs[active] / coefficient_xprime[active, j]
            hi[active] = rhs[active] / coefficient_xprime[active, j]
            lo[~active], hi[~active] = -np.inf, np.inf
            lower[:, j], upper[:, j] = np.minimum(lo, hi), np.maximum(lo, hi)
            impossible = ~active & (
                (gravity[:, j] + inertial_component[:, j] < bounds[0, j])
                | (gravity[:, j] + inertial_component[:, j] > bounds[1, j])
            )
            lower[impossible, j] = np.inf
            upper[impossible, j] = -np.inf
        global_lower = np.max(lower, axis=1)
        global_upper = np.min(upper, axis=1)
        limiting_lower = np.argmax(lower, axis=1) + 1
        limiting_upper = np.argmin(upper, axis=1) + 1
        utilization = torque_utilization(direct, bounds)
        peak_sample, peak_joint = np.unravel_index(
            np.argmax(utilization), utilization.shape
        )
        interval_width = global_upper - global_lower
        finite_interval = np.isfinite(interval_width)
        restrictive_sample = (
            int(np.argmin(np.where(finite_interval, interval_width, np.inf)))
            if np.any(finite_interval)
            else None
        )
        ds = np.diff(s)
        mid_utilization = np.maximum(
            utilization[:-1].max(axis=1), utilization[1:].max(axis=1)
        )
        midpoint_speed = np.sqrt(np.maximum(0.5 * (x[:-1] + x[1:]), 1e-12))
        dt_weights = ds / midpoint_speed
        curvature = np.sqrt(
            np.maximum(
                np.sum(qs * qs, axis=1) * np.sum(qss * qss, axis=1)
                - np.sum(qs * qss, axis=1) ** 2,
                0.0,
            )
        ) / np.maximum(np.linalg.norm(qs, axis=1) ** 3, 1e-12)
        geometric_speed = np.linalg.norm(qs, axis=1)
        curvature_mask = geometric_speed > max(
            float(np.max(geometric_speed)) * 1e-3, 1e-8
        )
        regular_curvature = curvature[curvature_mask]
        summary = {
            "duration_s": duration,
            "joint_path_length_rad": float(np.trapezoid(np.linalg.norm(qs, axis=1), s)),
            "curvature_regular_sample_fraction": float(np.mean(curvature_mask)),
            "mean_regular_curvature_rad_inverse": float(np.mean(regular_curvature)),
            "p95_regular_curvature_rad_inverse": float(
                np.percentile(regular_curvature, 95)
            ),
            "max_regular_curvature_rad_inverse": float(np.max(regular_curvature)),
            "max_torque_utilization": float(np.max(utilization)),
            "max_torque_joint": int(peak_joint + 1),
            "max_torque_path_progress": float(s[peak_sample]),
            "configuration_at_max_torque_rad": q[peak_sample].tolist(),
            "max_torque_components_Nm": {
                "gravity": float(gravity[peak_sample, peak_joint]),
                "velocity_curvature": float(
                    m[peak_sample, peak_joint] * x[peak_sample]
                ),
                "path_acceleration": float(
                    0.5 * a[peak_sample, peak_joint] * xp[peak_sample]
                ),
            },
            "fraction_duration_above_90pct_torque": float(
                np.sum(dt_weights[mid_utilization >= 0.90]) / np.sum(dt_weights)
            ),
            "fraction_duration_above_95pct_torque": float(
                np.sum(dt_weights[mid_utilization >= 0.95]) / np.sum(dt_weights)
            ),
            "torque_equation_max_error_Nm": float(
                np.max(np.abs(reconstructed - direct))
            ),
            "feasible_global_xprime_fraction": float(
                np.mean(global_lower <= global_upper)
            ),
            "most_restrictive_xprime_path_progress": (
                None if restrictive_sample is None else float(s[restrictive_sample])
            ),
            "most_restrictive_xprime_configuration_rad": (
                None if restrictive_sample is None else q[restrictive_sample].tolist()
            ),
            "most_restrictive_xprime_interval": (
                None
                if restrictive_sample is None
                else [
                    float(global_lower[restrictive_sample]),
                    float(global_upper[restrictive_sample]),
                ]
            ),
            "active_lower_joint_fraction": {
                str(j): float(np.mean(limiting_lower == j)) for j in range(1, 8)
            },
            "active_upper_joint_fraction": {
                str(j): float(np.mean(limiting_upper == j)) for j in range(1, 8)
            },
            "physical_constraint_bounds_Nm": {
                "lower": profile["effort_lower"],
                "upper": profile["effort_upper"],
            },
        }
        result["methods"][method] = {"status": "ANALYZED", "metrics": summary}
        result.setdefault("profiles", {})[method] = {
            "s": s,
            "q": q,
            "x": x,
            "x_prime": xp,
            "gravity": gravity,
            "velocity_curvature_term": inertial_component,
            "acceleration_term": acceleration_component,
            "total_torque": direct,
            "joint_torque_xprime_lower": lower,
            "joint_torque_xprime_upper": upper,
            "global_xprime_lower": global_lower,
            "global_xprime_upper": global_upper,
            "limiting_lower_joint": limiting_lower,
            "limiting_upper_joint": limiting_upper,
            "curvature": curvature,
            "torque_utilization": utilization,
        }
        source_hash = sha(trajectory_path)
        files[method.lower() + "_dense_sha256"] = source_hash
    out = arm / "mechanisms" / case
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for method, profiles in result.pop("profiles", {}).items():
        arrays.update(
            {method.lower() + "_" + name: values for name, values in profiles.items()}
        )
    np.savez_compressed(out / "path_torque_intervals.npz", **arrays)
    result["source_sha256"] = files
    result["interpretation"] = (
        "Path-coordinate torque intervals are descriptive along the selected timing profiles; "
        "they do not prove which term causally determines the duration."
    )
    (out / "path_torque_intervals.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--regime", required=True, choices=("moderate", "severe"))
    parser.add_argument(
        "--cases", help="comma-separated source case IDs; default all admitted"
    )
    parser.add_argument("--samples", type=int, default=501)
    parser.add_argument(
        "--path-coordinates",
        action="store_true",
        help="also compute torque-limited x-prime intervals along each path",
    )
    args = parser.parse_args()
    arm = args.root / args.regime
    tasks = json.loads((arm / "tasks.json").read_text())
    selected = set(args.cases.split(",")) if args.cases else None
    rows = []
    for task in tasks:
        case = task["case_id"]
        if selected is not None and case not in selected:
            continue
        try:
            item = evaluate_case(arm, case, count=args.samples)
            if args.path_coordinates:
                path_coordinate_case(arm, case, count=args.samples)
            rows.append(
                {
                    "case_id": case,
                    **item["contrast"],
                    "status": "COMPLETE" if item["contrast"] else "INCOMPLETE",
                }
            )
        except Exception as exc:
            rows.append({"case_id": case, "status": "ERROR", "reason": str(exc)})
    output = {"regime": args.regime, "cases": rows}
    path = arm / "mechanisms" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": len(rows),
                "complete": sum(r["status"] == "COMPLETE" for r in rows),
                "output": str(path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
