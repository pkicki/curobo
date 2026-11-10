# SPDX-License-Identifier: Apache-2.0
"""Frozen nominal-actuator payload contracts and independent rigid-body model.

A payload is an actual rigid inertia at a specified Panda tool frame, not a
scalar multiplying joint effort, and not a changed torque limit. In cuRobo
native runs the GPU's attached-object inertia and collision model must be
independently checked against this Pinocchio model BEFORE scoring. A
mismatched or unsupported native attachment is BLOCKED, never a negative
planning example. The attached collision sphere is a conservative research
proxy, not a manufacturer-validated payload enclosure.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Payload:
    mass_kg: float
    com_tool_m: tuple[float, float, float] = (0.0, 0.0, 0.12)
    radius_m: float = 0.06
    inertia_diagonal_kg_m2: tuple[float, float, float] = (0.001, 0.001, 0.001)
    tool_frame: str = "panda_hand"

    def validate(self):
        if not 0.0 <= self.mass_kg <= 5.0:
            raise ValueError("research mass outside registered 0..5 kg grid")
        if len(self.com_tool_m) != 3 or len(self.inertia_diagonal_kg_m2) != 3:
            raise ValueError("payload vectors must be 3D")
        if not all(np.isfinite(v) for v in (*self.com_tool_m,
                   *self.inertia_diagonal_kg_m2,self.radius_m)):
            raise ValueError("nonfinite payload")
        if not all(x > 0 for x in self.inertia_diagonal_kg_m2):
            raise ValueError("positive inertia required")
        if not 0.015 <= self.radius_m <= 0.25:
            raise ValueError("invalid payload collision radius")
        if self.tool_frame != "panda_hand":
            raise ValueError("unregistered Panda payload frame")
        return self

    def serialize(self):
        self.validate()
        return asdict(self)


MASS_GRID_KG = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0)
COM_TOOL_M = ((0.0, 0.0, 0.12), (0.04, 0.0, 0.16))
SEED = 20261110


def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as inp:
        for piece in iter(lambda:inp.read(1024*1024),b""):
            h.update(piece)
    return h.hexdigest()


def attach_pinocchio_payload(model, payload:Payload):
    """Copy model; add tool-local inertia transformed into parent joint."""
    import pinocchio as pin
    p=payload.validate()
    out=model.copy()
    if p.mass_kg == 0:
        return out
    fid=out.getFrameId(p.tool_frame, pin.BODY)
    if fid >= len(out.frames):
        raise ValueError("missing payload tool frame: "+p.tool_frame)
    frame=out.frames[fid]
    com=np.asarray(p.com_tool_m,dtype=float)
    inertia=pin.Inertia(p.mass_kg,com,
                        np.diag(np.asarray(p.inertia_diagonal_kg_m2)))
    out.inertias[frame.parentJoint] += frame.placement.act(inertia)
    return out


def model_aligned_payload(payload:Payload):
    """Return Pinocchio model, data, neutral, robot q/v selectors."""
    from .cohort_retime import dynamics
    model,_,q,qi,vi=dynamics()
    updated=attach_pinocchio_payload(model,payload)
    return updated,updated.createData(),q.copy(),qi,vi


def signed_gravity_profile(payload:Payload,q:np.ndarray)->np.ndarray:
    import pinocchio as pin
    model,data,state,qi,vi=model_aligned_payload(payload)
    positions=np.asarray(q,dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 7:
        raise ValueError("expected N-by-7 joint configurations")
    out=np.empty_like(positions)
    zero=np.zeros(model.nv)
    for i,config in enumerate(positions):
        state[qi]=config
        out[i]=pin.rnea(model,data,state,zero,zero)[vi]
    return out


def compare_native_inverse_dynamics(native_torque:np.ndarray,
                                    independent_torque:np.ndarray,
                                    *,atol_nm:float=0.15,rtol:float=0.01)->dict:
    """Research admission gate: do not benchmark a model mismatch."""
    a=np.asarray(native_torque,dtype=float)
    b=np.asarray(independent_torque,dtype=float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=7 or (
        not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("native/independent torque sample shapes disagree")
    errors=np.abs(a-b)
    tolerance=atol_nm+rtol*np.abs(b)
    return {"aligned":bool(np.all(errors<=tolerance)),
            "max_abs_difference_nm":float(errors.max()),
            "max_tolerance_ratio":float(np.max(errors/tolerance)),
            "sample_count":len(a),"atol_nm":atol_nm,"rtol":rtol}


def payload_sphere_world_clearance(auditor,payload:Payload,
                                    configurations:np.ndarray)->float|None:
    """World-only sphere proxy; self-collision requires full attached model."""
    import pinocchio as pin
    p=payload.validate()
    q=np.asarray(configurations,dtype=float)
    if q.ndim!=2 or q.shape[1]!=7:
        raise ValueError("invalid configurations")
    if p.mass_kg==0:
        return None
    fid=auditor.model.getFrameId(p.tool_frame,pin.BODY)
    if fid>=len(auditor.model.frames):
        raise ValueError("missing payload tool frame")
    out=float("inf")
    com=np.asarray(p.com_tool_m)
    for x in q:
        auditor.q[auditor.qi]=x
        pin.framesForwardKinematics(auditor.model,auditor.data,auditor.q)
        pose=auditor.data.oMf[fid]
        center=pose.rotation@com+pose.translation
        for _,origin,rotation,half in auditor.cuboids:
            delta=np.abs(rotation.inv().apply(center-origin))-half
            signed=float(np.linalg.norm(np.maximum(delta,0))+
                         min(float(np.max(delta)),0)-p.radius_m)
            out=min(out,signed)
        for _,origin,radius in auditor.spheres:
            out=min(out,float(np.linalg.norm(center-origin)-radius-p.radius_m))
        for _,start,end,radius in auditor.capsules:
            axis=end-start
            denom=float(axis@axis)
            if denom<1e-12:raise ValueError("invalid obstacle capsule")
            t=float(np.clip((center-start)@axis/denom,0,1))
            out=min(out,float(np.linalg.norm(center-(start+t*axis))-radius-p.radius_m))
    return None if np.isinf(out) else out


def frozen_profiles()->list[dict]:
    profiles=[]
    for mass in MASS_GRID_KG:
        for com in COM_TOOL_M:
            p=Payload(mass_kg=mass,com_tool_m=com)
            profiles.append({"profile_id":f"nominal_m{mass:g}_com{com[0]:g}_{com[2]:g}",
                             "payload":p.serialize(),"effort_scale":1.0,
                             "source":"nominal Panda efforts; rigid inertia added at hand",
                             "collision_status":"REQUIRES_NATIVE_AND_INDEPENDENT_ATTACHED_OBJECT_PARITY"})
    return profiles


def freeze(output:Path)->dict:
    from .cohort_retime import ROOT
    profiles=frozen_profiles()
    base=ROOT/"curobo/content/assets/robot/franka_description/franka_panda.urdf"
    cfg=ROOT/"curobo/content/configs/robot/franka.yml"
    document={"schema_version":1,"seed":SEED,"profiles":profiles,
              "robot_urdf_sha256":sha256_file(base),
              "nominal_robot_config_sha256":sha256_file(cfg),
              "nominal_effort_scale":1.0,
              "critical_gate":"native and Pinocchio dynamics+attached collision parity required"}
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() and json.loads(output.read_text())!=document:
        raise ValueError("immutable payload profile registration changed")
    if not output.exists():
        output.write_text(json.dumps(document,indent=2)+"\n")
    return document


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    a=parser.parse_args()
    print(json.dumps(freeze(a.output),indent=2))


if __name__=="__main__":
    main()
