# SPDX-License-Identifier: Apache-2.0
"""Research-only matched rigid payload for native cuRobo, K107 and Pinocchio.

Payload mass modifies DYNAMICS, never nominal actuator limits. The payload
is a spherical rigid body attached to the existing franka.yml fixed
"attached_object" link. All three systems must use the SAME mass, COM and
inertia. Collision-sphere geometry and payload pose are independent of
the mass and must be checked separately against the scene geometry.
Do not accept a benchmark unless native-vs-independent RNEA parity passes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math

import numpy as np


@dataclass(frozen=True)
class Payload:
    mass_kg: float
    com_xyz_m: tuple[float,float,float] = (0.,0.,0.)
    radius_m: float = 0.055
    attached_link: str = "attached_object"
    hand_link: str = "panda_hand"

    def validate(self):
        if (not math.isfinite(self.mass_kg) or self.mass_kg<0 or self.mass_kg>15
                or len(self.com_xyz_m)!=3
                or not all(math.isfinite(v) and abs(v)<=.35 for v in self.com_xyz_m)
                or not math.isfinite(self.radius_m) or not .01<=self.radius_m<=.25
                or self.attached_link!="attached_object" or self.hand_link!="panda_hand"):
            raise ValueError("unsupported or nonphysical research payload")
        return self

    def record(self)->dict:
        self.validate()
        data=asdict(self)
        data["payload_sha256"]=hashlib.sha256(json.dumps(
            data,sort_keys=True,allow_nan=False).encode()).hexdigest()
        return data


def nominal_effort_only(actual:np.ndarray,nominal:np.ndarray)->None:
    a=np.asarray(actual,dtype=float)
    b=np.asarray(nominal,dtype=float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[0]!=2 or not np.allclose(a,b,atol=1e-6,rtol=1e-6):
        raise ValueError("payload study forbids actuator torque derating")


def apply_curobo_payload(robot,payload:Payload):
    """Mutate only a freshly loaded private RobotCfg, before MotionPlanner init.

    Avoid changing a shared singleton. This uses the same KinematicsParams
    link-mass representation as time_optimal_curobo.CuroboRobotModel.
    """
    payload.validate()
    kp=robot.kinematics.kinematics_config
    if payload.attached_link not in kp.link_name_to_idx_map:
        raise ValueError("cuRobo model lacks attached_object; no payload dynamics parity")
    index=int(kp.link_name_to_idx_map[payload.attached_link])
    masses=kp.link_masses_com
    inertias=kp.link_inertias
    if masses is None or inertias is None or masses.shape[-1]!=4:
        raise ValueError("native cuRobo payload inertia tensors unavailable")
    # New RobotCfg, fixed attachment. Coordinates are expressed in local link.
    import torch
    with torch.no_grad():
        masses[index,:3]=torch.as_tensor(payload.com_xyz_m,device=masses.device,dtype=masses.dtype)
        masses[index,3]=payload.mass_kg
        i=2./5.*payload.mass_kg*payload.radius_m**2
        inertias[index].zero_()
        # cuRobo uses [Ixx,Iyy,Izz,Ixy,Ixz,Iyz].
        inertias[index,:3]=i
    return robot


def apply_pinocchio_payload(model,payload:Payload):
    """Add equivalent rigid-body inertia to Pinocchio hand parent joint.

    In the bundled franka configuration attached_object is an extra
    FIXED cuRobo link at identity transform on panda_hand. If that
    identity assumption changes the agent must update the parity
    calibration and freeze a new experiment.
    """
    payload.validate()
    import pinocchio as pin
    frame=model.getFrameId(payload.hand_link,pin.BODY)
    if frame>=len(model.frames):
        raise ValueError("Pinocchio model has no panda_hand body frame")
    hand=model.frames[frame]
    joint=hand.parentJoint
    placement=hand.placement
    com=placement.act(np.asarray(payload.com_xyz_m))
    inertia=2./5.*payload.mass_kg*payload.radius_m**2*np.eye(3)
    inertia_joint=placement.rotation@inertia@placement.rotation.T
    model.inertias[joint]+=pin.Inertia(payload.mass_kg,com,inertia_joint)
    return model


def rnea_parity(native:np.ndarray,independent:np.ndarray,*,atol_nm:float=.15,
                rtol:float=.015)->dict:
    a=np.asarray(native,dtype=float)
    b=np.asarray(independent,dtype=float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=7 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid model-alignment witness shape or nonfinite torque")
    difference=np.abs(a-b)
    bound=atol_nm+rtol*np.maximum(np.abs(a),np.abs(b))
    okay=bool(np.all(difference<=bound))
    return {"passed":okay,"samples":len(a),"worst_abs_nm":float(difference.max()),
            "worst_ratio":float((difference/bound).max()),
            "atol_nm":atol_nm,"rtol":rtol,
            "interpretation":"same q,qd,qdd and same payload; independent Pinocchio vs native cuRobo RNEA"}
