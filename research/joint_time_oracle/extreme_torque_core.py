# SPDX-License-Identifier: Apache-2.0
"""Pure-numerical primitives for adversarial torque-feasibility witnesses.

Synthetic actuator derating does not modify robot dynamics. A time dilation
rho>=1 of a rigid-body, fixed-endpoint-rest trajectory preserves its geometry:
q_rho(t)=q(t/rho), qd/rho, qdd/rho**2, jerk/rho**3, and
tau_rho = gravity(q) + (tau_original - gravity(q))/rho**2.
The analytical formula is a SCREEN ONLY; independent Pinocchio RNEA and
world/self collision checks on exact spline reconstructions are required.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .torque_active import (
    TorqueStress,
    canonical_hash,
    checked_bounds,
    torque_utilization,
)

# Preregistered, independent of candidate outcomes or native failure status.
DEFAULT_FACTORS = (1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65)
DEFAULT_SLOWDOWNS = (1.0, 1.05, 1.10, 1.20, 1.35, 1.50, 1.75, 2.0, 2.5, 3.0, 4.0)
TORQUE_HEADROOM = 0.005  # witness must be <=99.5% of *scaled95* effort bound
ENDPOINT_GRAVITY_RESERVE = 0.85


def sha256_file(path: Path) -> str:
    """Compute the SHA256 digest of a file in bounded memory."""
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def regime_name(factor: float) -> str:
    """Return the registered factor label for an actuator derating."""
    if not 0.5 <= factor <= 1.0:
        raise ValueError("relative derating must lie in [0.5,1]")
    milli = round(factor * 1000)
    if abs(milli / 1000 - factor) > 1e-12:
        raise ValueError("factor must be represented by integer thousandths")
    return f"extreme_f{milli:04d}"


def derived_profile(old: dict, factor: float, *, source_sha: str) -> TorqueStress:
    """Derate a FROZEN severe profile, not a candidate-specific fitted limit."""
    name = regime_name(factor)
    expected_sha = old.get("source_task_sha256")
    if expected_sha != source_sha or old.get("regime") != "severe":
        raise ValueError("source task SHA or severe source regime mismatch")
    nominal = checked_bounds(
        np.array([old["nominal_effort_lower"], old["nominal_effort_upper"]])
    )
    alpha = float(old["derate"]) * factor
    static = float(old["endpoint_gravity_utilization"])
    # Static endpoint feasibility is a necessary physical contract for
    # zero-velocity, zero-acceleration endpoint states; leave extra reserve.
    reserve = static / alpha if alpha else float("inf")
    reasons = []
    if alpha <= 0 or alpha > 1:
        reasons.append("invalid physical actuator derating")
    if reserve > ENDPOINT_GRAVITY_RESERVE:
        reasons.append("endpoint gravity exceeds 85% of modified effort bounds")
    torque = nominal * alpha
    return TorqueStress(
        case_id=name + "::" + old["source_case_id"],
        source_case_id=old["source_case_id"],
        regime=name,
        target_baseline_utilization=float(old["target_baseline_utilization"]) / factor,
        derate=alpha,
        nominal_effort_lower=nominal[0].tolist(),
        nominal_effort_upper=nominal[1].tolist(),
        effort_lower=torque[0].tolist(),
        effort_upper=torque[1].tolist(),
        observed_baseline_utilization=float(old["observed_baseline_utilization"]),
        endpoint_gravity_utilization=static,
        expected_reference_utilization=float(old["observed_baseline_utilization"])
        / alpha,
        static_reserve_fraction=ENDPOINT_GRAVITY_RESERVE,
        admitted=not reasons,
        rejection_reason="; ".join(reasons) if reasons else None,
        source_task_sha256=source_sha,
        source_torque_sha256=old["source_torque_sha256"],
    )


def dilate_states(
    states: dict[str, np.ndarray], slowdown: float
) -> dict[str, np.ndarray]:
    """No interpolation: rescale the exact original sample timestamps/states."""
    if not np.isfinite(slowdown) or slowdown < 1:
        raise ValueError("slowdown must be >=1")
    expected = ("time", "q", "qd", "qdd", "jerk")
    if not all(key in states for key in expected):
        raise ValueError("source trajectory must contain time, q, qd, qdd, jerk")
    t = np.asarray(states["time"], dtype=float)
    if t.ndim != 1 or len(t) < 3 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("source trajectory times are not strictly increasing")
    q = np.asarray(states["q"], dtype=float)
    if (
        q.ndim != 2
        or q.shape[0] != len(t)
        or any(
            np.asarray(states[key]).shape != q.shape
            or not np.isfinite(states[key]).all()
            for key in ("qd", "qdd", "jerk")
        )
    ):
        raise ValueError("trajectory state arrays must be finite and aligned")
    if (
        np.max(np.abs(states["qd"][[0, -1]])) > 1e-4
        or np.max(np.abs(states["qdd"][[0, -1]])) > 1e-3
    ):
        raise ValueError("time dilating non-rest endpoint contract is not allowed")
    out = {key: np.asarray(value).copy() for key, value in states.items()}
    out["time"] = t[0] + (t - t[0]) * slowdown
    for key, order in (("qd", 1), ("qdd", 2), ("jerk", 3)):
        out[key] = np.asarray(states[key], dtype=float) / slowdown**order
    if "controls" in out:  # IPOPT speed spline x(s)=sdot**2
        out["controls"] = np.asarray(states["controls"], dtype=float) / slowdown**2
    return out


def implied_torque(
    tau_original: np.ndarray, gravity: np.ndarray, slowdown: float
) -> np.ndarray:
    """Rigid-body time-scaling identity, for numerical screening only."""
    t = np.asarray(tau_original, dtype=np.float64)
    g = np.asarray(gravity, dtype=np.float64)
    if (
        t.shape != g.shape
        or t.ndim != 2
        or not np.isfinite(t).all()
        or not np.isfinite(g).all()
    ):
        raise ValueError("torque and gravity arrays must be finite [samples,dof]")
    if not np.isfinite(slowdown) or slowdown < 1:
        raise ValueError("invalid slowdown")
    return g + (t - g) / slowdown**2


def screen_slowdowns(
    tau: np.ndarray,
    gravity: np.ndarray,
    bounds: np.ndarray,
    slowdowns=DEFAULT_SLOWDOWNS,
    *,
    required_headroom: float = TORQUE_HEADROOM,
) -> tuple[float | None, list[dict]]:
    """Select the shortest preregistered candidate meeting SCALED95 torque."""
    effort = checked_bounds(bounds)
    if not 0 <= required_headroom < 0.05:
        raise ValueError("invalid headroom")
    checked = []
    chosen = None
    for rho in slowdowns:
        predicted = implied_torque(tau, gravity, float(rho))
        physical = float(np.max(torque_utilization(predicted, effort)))
        scaled = float(np.max(torque_utilization(predicted, effort * 0.95)))
        eligible = scaled <= 1 - required_headroom + 1e-12
        checked.append(
            {
                "slowdown": float(rho),
                "predicted_physical_torque_utilization": physical,
                "predicted_scaled95_torque_utilization": scaled,
                "screen_pass": eligible,
            }
        )
        if eligible and chosen is None:
            chosen = float(rho)
    return chosen, checked


def source_manifest(source: Path, profiles: list[dict], config: dict) -> dict:
    """Build a checksummed manifest for the frozen source and stress arms."""
    manifest = {
        "schema_version": 1,
        "source": str(source),
        "profiles": profiles,
        "protocol": config,
        "status": "PRESCREEN_ONLY; not independently qualified",
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    return manifest


def check_manifest(payload: dict) -> bool:
    """Check the canonical manifest signature."""
    d = dict(payload)
    sig = d.pop("manifest_sha256", None)
    return isinstance(sig, str) and canonical_hash(d) == sig
