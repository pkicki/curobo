# SPDX-License-Identifier: Apache-2.0
"""Per-payload native CUDA-vs-Pinocchio RNEA and four-sphere collision parity.

This is mandatory for promotion from a dynamics-only heavy-payload
scenario to a valid robot+payload collision task. A single source
configuration hash is NOT a parity check. Fail closed if cuRobo cannot
expose exact native collision spheres on tested joint states.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
import torch

from curobo._src.robot.dynamics.dynamics import Dynamics
from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
from curobo._src.robot.kinematics.kinematics import CudaRobotModel
from curobo._src.types.device_cfg import DeviceCfg
from curobo.types import JointState

from .cohort_retime import ROOT
from .payload_model_parity import (
    assert_native_independent_dynamics_parity,
    independent_payload_auditor,independent_payload_model,native_payload_robot,
)
from .payload_timeopt_contract import Payload,canonical_sha,immutable_json,source_sha


def native_sphere_set(robot,positions:np.ndarray)->list[np.ndarray]:
    model=CudaRobotModel(robot.kinematics)
    groups=[]
    for q in positions:
        jt=torch.as_tensor(q,device="cuda",dtype=torch.float32)[None]
        current=model.get_robot_as_spheres(jt)
        if isinstance(current,list) and len(current)==1 and isinstance(current[0],list):
            current=current[0]
        groups.append(np.array([
            [*sphere.pose[:3],float(sphere.radius)] for sphere in current],
            dtype=float).reshape(-1,4))
    return groups


def independent_sphere_set(auditor,positions:np.ndarray)->list[np.ndarray]:
    groups=[]
    for q in positions:
        auditor.q[auditor.qi]=q
        pin.framesForwardKinematics(auditor.model,auditor.data,auditor.q)
        center=np.empty_like(auditor.local)
        for indices,frame in auditor.grouped:
            pose=auditor.data.oMf[frame]
            center[indices]=pose.translation+(pose.rotation@auditor.local[indices].T).T
        groups.append(np.column_stack((center,auditor.radii)))
    return groups


def sphere_parity(native:list[np.ndarray],independent:list[np.ndarray],
                  *,tol_m:float=2e-3)->dict:
    if len(native)!=len(independent) or len(native)<8:
        raise ValueError("too few matched sphere observations")
    worst=0.
    for a,b in zip(native,independent):
        if len(a)!=len(b) or a.shape[1:]!=(4,) or b.shape[1:]!=(4,):
            raise ValueError("native/independent sphere count mismatch")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("invalid sphere centers/radii")
        # Match each native center/radius with a UNIQUE independent
        # center/radius: protects against duplicate or missing payload spheres.
        from scipy.optimize import linear_sum_assignment
        cost=np.linalg.norm(a[:,None,:3]-b[None,:,:3],axis=-1)+(
            10.*np.abs(a[:,None,3]-b[None,:,3]))
        row,col=linear_sum_assignment(cost)
        if len(row)!=len(a):
            raise ValueError("incomplete sphere matching")
        worst=max(worst,float(np.max(cost[row,col])))
    if worst>tol_m:
        raise ValueError(f"native and Pinocchio payload collision sphere mismatch {worst:.6g}m")
    return {"status":"PASS_COLLISION_SPHERES","sample_poses":len(native),
            "maximum_matched_sphere_center_and_radius_error_m":worst,
            "match_tolerance_m":tol_m}


def qualify(payload:Payload,*,scene:str|None,seed:int=20261110,n:int=24)->dict:
    if not torch.cuda.is_available():
        raise RuntimeError("native CUDA parity host required")
    robot,meta=native_payload_robot(payload,with_payload_spheres=True)
    indep=independent_payload_model(payload)
    auditor=independent_payload_auditor(payload,scene)
    rng=np.random.default_rng(seed)
    lower,upper=auditor.position_bounds
    home=np.array([0.,-.785,0.,-2.356,0.,1.571,.785])
    samples=np.zeros((n,3,7),dtype=float)
    for i in range(n):
        samples[i,0]=np.clip(home+rng.normal(0,.7,7),lower+.08,upper-.08)
        samples[i,1]=rng.normal(0,.8,7)
        samples[i,2]=rng.normal(0,2.0,7)
    samples[0,1:]=0
    samples[1,1]=0;samples[1,2]=rng.normal(0,.5,7)
    device=DeviceCfg(device=torch.device("cuda"),dtype=torch.float32)
    solver=Dynamics(DynamicsCfg(kinematics_config=robot.kinematics.kinematics_config,
                               device_cfg=device))
    solver.setup_batch_size(batch_size=n)
    names=[f"panda_joint{i}" for i in range(1,8)]
    def native_torque(q,v,a):
        t=lambda z:torch.as_tensor(z,device="cuda",dtype=torch.float32)
        joint=JointState(position=t(q),velocity=t(v),acceleration=t(a),
                         joint_names=names)
        return solver.compute_inverse_dynamics(joint).detach().cpu().numpy()
    physics=assert_native_independent_dynamics_parity(native_torque,indep,samples)
    native=native_sphere_set(robot,samples[:,0])
    independent=independent_sphere_set(auditor,samples[:,0])
    collision=sphere_parity(native,independent)
    return {"schema":1,"status":"PASS_NATIVE_PINOCCHIO_RNEA_AND_COLLISION_SPHERES",
            "payload":{"mass_kg":payload.mass_kg,
                       "com_xyz_m":list(payload.com_xyz_m)},
            "scene":scene,"seed":seed,"samples":n,
            "nominal_effort":meta["effort_parity"],
            "native_model":meta,"dynamics":physics,"collision":collision,
            "scope":"sampled four-sphere payload collision and torque model parity; not hardware safety"}


def verify_task_payload_receipt(task:dict)->dict:
    """Fail-closed source attestation for every qualified payload planning call."""
    if task.get("payload_collision_qualified") is not True:
        raise ValueError("payload collision model has not been independently parity qualified")
    file=Path(task["payload_collision_receipt_file"])
    if not file.is_file() or source_sha(file)!=task["payload_collision_receipt_sha256"]:
        raise ValueError("missing/stale native/independent payload parity SHA")
    receipt=json.loads(file.read_text())
    mass=task["payload"]
    if (receipt.get("status")!="PASS_NATIVE_PINOCCHIO_RNEA_AND_COLLISION_SPHERES"
        or receipt.get("payload",{}).get("mass_kg")!=mass["mass_kg"]
        or receipt.get("payload",{}).get("com_xyz_m")!=list(mass["com_xyz_m"])
        or receipt.get("scene")!=task.get("scene")):
        raise ValueError("task/receipt payload mass, COM, scene or status mismatch")
    if task.get("joint_limit_scale")!=1.0 or not task.get("nominal_actuator_limits"):
        raise ValueError("non-nominal original effort limits in payload task")
    if canonical_sha({k:v for k,v in task.items() if k!="task_sha256"})!=task.get("task_sha256"):
        raise ValueError("promoted task canonical digest mismatch")
    return receipt


def admit_task(task:dict,receipt:dict,receipt_file:Path)->dict:
    payload=Payload(**task["payload"])
    if receipt["status"]!="PASS_NATIVE_PINOCCHIO_RNEA_AND_COLLISION_SPHERES":
        raise ValueError("payload parity receipt not passed")
    if (receipt["payload"]["mass_kg"]!=payload.mass_kg or
        list(receipt["payload"]["com_xyz_m"])!=list(payload.com_xyz_m) or
        receipt["scene"]!=task["scene"]):
        raise ValueError("payload parity receipt for different task model/scene")
    if task.get("payload_collision_qualified") is not False:
        raise ValueError("only original dynamics-only task may be promoted")
    promoted={k:v for k,v in task.items() if k!="task_sha256"}
    promoted["payload_collision_qualified"]=True
    promoted["payload_collision_receipt_file"]=str(receipt_file.resolve())
    promoted["payload_collision_receipt_sha256"]=source_sha(receipt_file)
    promoted["source_collision_scope"]="native and independent same four-sphere payload parity verified on sampled states"
    promoted["task_sha256"]=canonical_sha(promoted)
    return promoted


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mass",type=float,required=True)
    p.add_argument("--com",type=float,nargs=3,default=[0,0,0])
    p.add_argument("--scene",default="free",choices=("free","table","cluttered"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=20261110)
    p.add_argument("--samples",type=int,default=24)
    args=p.parse_args()
    scenes={"free":None,"table":"collision_test.yml",
            "cluttered":"collision_primitives_3d.yml"}
    payload=Payload(args.mass,tuple(args.com))
    report=qualify(payload,scene=scenes[args.scene],seed=args.seed,n=args.samples)
    immutable_json(args.output,report)
    print(json.dumps({"status":report["status"],
                      "dynamics":report["dynamics"],"collision":report["collision"]},
                     indent=2))


if __name__=="__main__":
    main()
