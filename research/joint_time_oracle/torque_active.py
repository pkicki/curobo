# SPDX-License-Identifier: Apache-2.0
"""Deterministic, model-preserving torque derating for controlled stress tests.

Derating is a synthetic actuator constraint, NOT an attached payload and NOT
the manufacturer's Panda torque rating. Inverse dynamics and kinematics stay
unchanged. Both endpoints must admit the nominal static gravity torque with
specified reserve; intermediate configurations are intentionally allowed to
require re-planning.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
from typing import Any

import numpy as np


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def checked_bounds(bounds: np.ndarray) -> np.ndarray:
    result = np.asarray(bounds, dtype=np.float64)
    if (result.ndim != 2 or result.shape[0] != 2 or result.shape[1] == 0
            or not np.isfinite(result).all() or not (result[0] < 0).all()
            or not (result[1] > 0).all()):
        raise ValueError("effort bounds must have finite [2,dof] form, lower<0<upper")
    return result


def torque_utilization(tau: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Asymmetric signed-side effort utilization; supports [N,dof] and [dof]."""
    lim = checked_bounds(bounds)
    values = np.asarray(tau, dtype=np.float64)
    if values.ndim not in (1, 2) or values.shape[-1] != lim.shape[1]:
        raise ValueError("torque values must have shape [dof] or [N,dof]")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite torque")
    return np.where(values >= 0, values / lim[1], values / lim[0])


@dataclass(frozen=True)
class TorqueStress:
    case_id: str
    source_case_id: str
    regime: str
    target_baseline_utilization: float
    derate: float
    nominal_effort_lower: list[float]
    nominal_effort_upper: list[float]
    effort_lower: list[float]
    effort_upper: list[float]
    observed_baseline_utilization: float
    endpoint_gravity_utilization: float
    expected_reference_utilization: float
    static_reserve_fraction: float
    admitted: bool
    rejection_reason: str | None
    source_task_sha256: str
    source_torque_sha256: str

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["profile_sha256"] = canonical_hash(payload)
        return payload


def make_profile(
    *,
    case_id: str,
    source_case_id: str,
    regime: str,
    nominal_bounds: np.ndarray,
    baseline_torques: np.ndarray,
    endpoint_gravity_torques: np.ndarray,
    target: float,
    source_task_sha256: str,
    source_torque_sha256: str,
    min_derate: float = 0.15,
    max_derate: float = 0.95,
    static_reserve_fraction: float = 0.85,
    min_resulting_peak: float = 0.78,
) -> TorqueStress:
    """Choose one derating factor from frozen native A40 data, not B/C/D.

    The max of the target factor and the static endpoint factor keeps gravity
    inside the derated bounds. Any configuration that fails the protocol is
    recorded but is not silently discarded or replaced.
    """
    if not (0.5 <= target <= 1.5):
        raise ValueError("target utilization must be in [0.5, 1.5]")
    if not (0 < min_derate < max_derate <= 1):
        raise ValueError("invalid derating range")
    if not (0 < static_reserve_fraction < 1):
        raise ValueError("invalid endpoint static reserve")
    lim = checked_bounds(nominal_bounds)
    sample = np.asarray(baseline_torques, dtype=np.float64)
    gravity = np.asarray(endpoint_gravity_torques, dtype=np.float64)
    if sample.ndim != 2 or sample.shape[1] != lim.shape[1]:
        raise ValueError("baseline torques must be [samples,dof]")
    if gravity.shape != (2, lim.shape[1]):
        raise ValueError("endpoint gravity must be [2,dof]")
    peak = float(np.max(torque_utilization(sample, lim)))
    static = float(np.max(torque_utilization(gravity, lim)))
    necessary = static / static_reserve_fraction
    chosen = max(min_derate, peak / target, necessary)
    reasons = []
    if chosen > max_derate:
        reasons.append("derating would violate static endpoint reserve or maximum factor")
    if chosen <= 0 or not math.isfinite(chosen):
        reasons.append("nonfinite derating")
    derate = float(min(chosen, max_derate))
    if peak / derate < min_resulting_peak:
        reasons.append("torque utilization too low to be an informative stress test")
    effort = lim * derate
    if np.max(torque_utilization(gravity, effort)) > static_reserve_fraction + 1e-12:
        reasons.append("static endpoint infeasible at frozen derating")
    return TorqueStress(
        case_id=case_id, source_case_id=source_case_id, regime=regime,
        target_baseline_utilization=float(target), derate=derate,
        nominal_effort_lower=lim[0].tolist(), nominal_effort_upper=lim[1].tolist(),
        effort_lower=effort[0].tolist(), effort_upper=effort[1].tolist(),
        observed_baseline_utilization=peak, endpoint_gravity_utilization=static,
        expected_reference_utilization=peak/derate,
        static_reserve_fraction=static_reserve_fraction,
        admitted=not reasons, rejection_reason="; ".join(reasons) if reasons else None,
        source_task_sha256=source_task_sha256, source_torque_sha256=source_torque_sha256,
    )


def read_profile(path) -> dict:
    from pathlib import Path
    profile = json.loads(Path(path).read_text())
    signature = profile.pop("profile_sha256", None)
    if not signature or canonical_hash(profile) != signature:
        raise ValueError("torque profile checksum mismatch")
    if profile["admitted"] is not True:
        raise ValueError("torque profile not admitted: " + str(profile["rejection_reason"]))
    bounds = checked_bounds(np.stack((profile["effort_lower"], profile["effort_upper"])))
    nominal = checked_bounds(np.stack((profile["nominal_effort_lower"],
                                      profile["nominal_effort_upper"])))
    if not np.allclose(bounds, nominal * profile["derate"], rtol=1e-12, atol=1e-12):
        raise ValueError("inconsistent torque derating")
    profile["profile_sha256"] = signature
    return profile


def write_profile(path, profile: TorqueStress):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_json(), indent=2, allow_nan=False) + "\n")
