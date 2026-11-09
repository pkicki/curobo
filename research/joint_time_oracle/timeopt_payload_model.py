# SPDX-License-Identifier: Apache-2.0
"""Original-actuator-limit payload model shared across research adapters.

This is a *physical inertia addition*, not a reduction in joint effort
limits. The initial scope is a spherical rigid body at attached_object,
fixed to panda_hand by the identity transform in the frozen franka.yml.
The exact same spherical inertia is attached to the panda_hand BODY
frame in Pinocchio. Realism is limited by this rigid simplified payload,
unknown grasp forces and unmodelled payload flex; no hardware safety
claim. Collision parity must be qualified separately before scoring.

Both model adapters must pass an independent native-vs-Pinocchio RNEA
parity gate before any payload scenario is admitted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MASSES_KG = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
PAYLOAD_LINK = "attached_object"
PIN_FRAME = "panda_hand"
RADIUS_M = 0.06


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Payload:
    mass_kg: float
    radius_m: float = RADIUS_M
    native_link: str = PAYLOAD_LINK
    pin_frame: str = PIN_FRAME

    def __post_init__(self):
        if not math.isfinite(self.mass_kg) or self.mass_kg < 0:
            raise ValueError("payload mass must be nonnegative and finite")
        if not math.isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError("payload radius must be positive and finite")
        if self.native_link != PAYLOAD_LINK or self.pin_frame != PIN_FRAME:
            raise ValueError("unregistered payload mounting frame")

    def as_record(self) -> dict:
        record = asdict(self)
        record["central_inertia_kg_m2"] = (
            2.0 / 5.0 * self.mass_kg * self.radius_m**2
        )
        record["payload_sha256"] = digest(record)
        return record


def from_record(record: dict) -> Payload:
    record = dict(record)
    signature = record.pop("payload_sha256", None)
    inertia = record.pop("central_inertia_kg_m2", None)
    payload = Payload(**record)
    expected = payload.as_record()
    if signature != expected["payload_sha256"] or not math.isclose(
        float(inertia), expected["central_inertia_kg_m2"], abs_tol=1e-12
    ):
        raise ValueError("payload model signature or inertia mismatch")
    return payload


def native_robot(payload: Payload):
    """Load a fresh native RobotCfg, mutate inertial tensors only, never effort."""
    import torch

    from curobo._src.types.robot import RobotCfg
    from curobo._src.util.config_io import join_path, resolve_config
    from curobo.content import get_robot_configs_path

    robot = RobotCfg.create(
        resolve_config(join_path(get_robot_configs_path(), "franka.yml"))
    )
    kp = robot.kinematics.kinematics_config
    if payload.native_link not in kp.link_name_to_idx_map:
        raise RuntimeError("attached_object not present in native dynamics model")
    i = int(kp.link_name_to_idx_map[payload.native_link])
    if kp.link_masses_com is None or kp.link_inertias is None:
        raise RuntimeError("native inertial tensor fields unavailable")
    reference = kp.joint_limits.effort.detach().clone()
    masses = kp.link_masses_com
    inertias = kp.link_inertias
    if not np.allclose(
        masses[i, :3].detach().cpu().numpy(), np.zeros(3), atol=1e-7
    ):
        raise RuntimeError("payload link COM is not zero; transform parity unverified")
    # Registered spherical body at the tool frame: no additional COM shift.
    with torch.no_grad():
        masses[i, 3] += payload.mass_kg
        inertia = 0.4 * payload.mass_kg * payload.radius_m**2
        inertias[i, :3] += inertia
    if not torch.equal(reference, kp.joint_limits.effort):
        raise RuntimeError("nominal effort bounds were changed by payload loading")
    return robot


def pinocchio_payload_model(payload: Payload):
    """Independent baseline Pinocchio model + the same spherical rigid inertia."""
    import pinocchio as pin

    from .cohort_retime import dynamics

    model, _, neutral, qi, vi = dynamics()
    if payload.mass_kg:
        idx = model.getFrameId(payload.pin_frame, pin.BODY)
        if idx >= len(model.frames):
            raise RuntimeError("Pinocchio payload frame does not exist")
        frame = model.frames[idx]
        inertia = pin.Inertia(
            payload.mass_kg, np.zeros(3),
            np.eye(3) * (0.4 * payload.mass_kg * payload.radius_m**2)
        )
        model.inertias[frame.parentJoint] += frame.placement.act(inertia)
    return model, model.createData(), neutral, qi, vi


def nominal_effort_bounds() -> np.ndarray:
    path = ROOT / (
        "research/joint_time_oracle/results/native_reconstruction_20261106/"
        "baseline_validation_states.npz"
    )
    with np.load(path) as data:
        return np.asarray(data["torque_bounds"], dtype=np.float64)


def compare_rnea(q: np.ndarray, qd: np.ndarray, qdd: np.ndarray,
                 native_tau: np.ndarray, payload: Payload, *,
                 atol_nm: float = 0.05, rtol: float = 0.005) -> dict:
    """Independent native-vs-Pinocchio parity gate, never assume equal models."""
    import pinocchio as pin

    q, qd, qdd, native_tau = map(
        lambda x: np.asarray(x, dtype=np.float64), (q, qd, qdd, native_tau)
    )
    if q.ndim != 2 or q.shape[1] != 7 or any(
        value.shape != q.shape for value in (qd, qdd, native_tau)
    ):
        raise ValueError("expected [n,7] joint positions, rates, and torques")
    model, data, base, qi, vi = pinocchio_payload_model(payload)
    reference = np.empty_like(q)
    velocity = np.zeros(model.nv)
    acceleration = np.zeros(model.nv)
    for index in range(len(q)):
        base[qi] = q[index]
        velocity[vi] = qd[index]
        acceleration[vi] = qdd[index]
        reference[index] = pin.rnea(model, data, base, velocity, acceleration)[vi]
    difference = np.abs(reference - native_tau)
    threshold = atol_nm + rtol * np.abs(reference)
    return {"parity_pass": bool(np.all(difference <= threshold)),
            "maximum_difference_nm": float(np.max(difference)),
            "maximum_relative_error": float(np.max(
                difference / np.maximum(np.abs(reference), 0.1)
            )),
            "atol_nm": atol_nm, "rtol": rtol, "samples": len(q),
            "payload_sha256": payload.as_record()["payload_sha256"],
            "interpretation": "must be sourced from an independent native GPU RNEA evaluation"}


def classify_endpoint_gravity(gravity: np.ndarray, limits: np.ndarray,
                              *, reserve: float = 0.95) -> dict:
    """Trivial static endpoint impossibility is a separate cohort category."""
    gravity = np.asarray(gravity, dtype=np.float64)
    limits = np.asarray(limits, dtype=np.float64)
    if gravity.shape != (2, 7) or limits.shape != (2, 7):
        raise ValueError("unexpected endpoint gravity/limit shape")
    if not np.isfinite(gravity).all() or not np.isfinite(limits).all():
        raise ValueError("nonfinite endpoints")
    if not (0 < reserve <= 1):
        raise ValueError("invalid reserve")
    from .torque_active import torque_utilization

    ratio = float(np.max(torque_utilization(gravity, limits)))
    return {"endpoint_utilization": ratio,
            "endpoint_static_feasible_nominal": bool(ratio <= 1.0),
            "endpoint_with_reserved_margin": bool(ratio <= reserve),
            "category": "STATIC_ENDPOINT_IMPOSSIBLE" if ratio > 1.0
                        else "ENDPOINT_ADMISSIBLE"}
