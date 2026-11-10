# SPDX-License-Identifier: Apache-2.0
"""Build payload parity receipt from independently logged native CUDA states.

Input NPZ keys: q[N,7], qd[N,7], qdd[N,7], native_tau[N,7],
native_world_clearance_m[N], native_self_clearance_m[N]. These MUST be
sampled from the same numerical robot model and attached sphere at the
same configurations; no scalar payload-only approximation is accepted.
A parity failure blocks the scored planner comparison. Collision model
uses sampled sphere distance, not mesh collision certification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from .cohort_retime import ROOT
from .payload_contract import Payload,compare_native_inverse_dynamics,sha256_file
from .payload_contract import model_aligned_payload
from .payload_geometry_audit import PayloadGeometryAuditor
from .retimer_oracle import FAMILIES


def model_receipt(task:dict,payload:Payload,native:Path,*,world_tol_m:float=.001,
                  torque_atol_nm:float=.15,torque_rtol:float=.01)->dict:
    with np.load(native) as z:
        required=("q","qd","qdd","native_tau",
                  "native_world_clearance_m","native_self_clearance_m")
        if not all(k in z.files for k in required):
            raise ValueError("native parity samples incomplete")
        arrays={k:np.asarray(z[k],dtype=float) for k in required}
    q,qd,qdd=(arrays[k] for k in ("q","qd","qdd"))
    if q.ndim!=2 or q.shape[1]!=7 or qd.shape!=q.shape or qdd.shape!=q.shape:
        raise ValueError("native sample shape mismatch")
    model,data,state,qi,vi=model_aligned_payload(payload)
    v=np.zeros(model.nv);a=np.zeros(model.nv)
    independent=np.empty_like(q)
    auditor=PayloadGeometryAuditor(task["scene"],payload)
    world=[];self_clear=[]
    for i,pose in enumerate(q):
        state[qi]=pose;v.fill(0.);a.fill(0.)
        v[vi]=qd[i];a[vi]=qdd[i]
        independent[i]=pin.rnea(model,data,state,v,a)[vi]
        check=auditor.check(pose[None],min_clearance_m=0)
        world.append(check["world_clearance_m"])
        self_clear.append(check["self_clearance_m"])
    torque=compare_native_inverse_dynamics(arrays["native_tau"],independent,
                        atol_nm=torque_atol_nm,rtol=torque_rtol)
    calc_world=np.asarray(world,dtype=float)
    calc_self=np.asarray(self_clear,dtype=float)
    # Infinite distance means the scene has no obstacle: native must
    # report the same type of result. NaN is always a failed comparison.
    def compare(a,b):
        if a.shape!=b.shape or np.isnan(a).any() or np.isnan(b).any():
            return False,float("inf")
        finite=np.isfinite(a)&np.isfinite(b)
        if not np.array_equal(np.isfinite(a),np.isfinite(b)):
            return False,float("inf")
        peak=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.
        return peak<=world_tol_m,peak
    world_pass,world_peak=compare(arrays["native_world_clearance_m"],calc_world)
    self_pass,self_peak=compare(arrays["native_self_clearance_m"],calc_self)
    return {"schema_version":1,"task_sha256":task["task_sha256"],
            "payload":payload.serialize(),"native_parity_npz_sha256":sha256_file(native),
            "native_pinocchio_payload_parity_verified":torque["aligned"],
            "attached_collision_parity_verified":world_pass and self_pass,
            "nominal_effort_scale":1.0,
            "world_max_difference_m":world_peak,
            "self_max_difference_m":self_peak,
            "collision_tolerance_m":world_tol_m,
            "torque_parity":torque,
            "samples":len(q),
            "status":"ADMITTED" if torque["aligned"] and world_pass and self_pass
                     else "BLOCKED_PAYLOAD_MODEL_OR_COLLISION_MISMATCH"}


def make_contract(task:dict,payload:Payload,native:Path,
                  *,output:Path,limits_source:Path)->dict:
    parity=model_receipt(task,payload,native)
    with np.load(limits_source) as z:
        bounds={k:np.asarray(z[k+"_bounds"],dtype=float)
                for k in ("position","velocity","acceleration","jerk","torque")}
    limits={
        "families":list(FAMILIES),
        "signed_torque_bounds_sha256":sha256_file(limits_source),
        "endpoint_contract":"strict_rest_velocity_and_acceleration",
        "physical_bounds":{k:v.tolist() for k,v in bounds.items()},
        "scaled95":True,
        "torque_rate":"POSTHOC_ONLY_SEPARATE_CONTRACT",
        "constraint_source_sha256":sha256_file(limits_source)}
    model={"payload":payload.serialize(),
           "nominal_effort_scale":1.0,
           "native_pinocchio_payload_parity_verified":
             parity["native_pinocchio_payload_parity_verified"],
           "attached_collision_parity_verified":
             parity["attached_collision_parity_verified"],
           "dynamics_parity_receipt_sha256":parity["native_parity_npz_sha256"],
           "collision_primitive":"panda_hand_tool_local_sphere",
           "robot_urdf_sha256":sha256_file(ROOT/
             "curobo/content/assets/robot/franka_description/franka_panda.urdf"),
           "robot_config_sha256":sha256_file(ROOT/
             "curobo/content/configs/robot/franka.yml")}
    receipt={"schema_version":1,"case_id":task["case_id"],
             "task_sha256":task["task_sha256"],"model":model,
             "limits":limits,"parity":parity}
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() and json.loads(output.read_text())!=receipt:
        raise ValueError("research parity contract already frozen")
    if not output.exists():
        output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",required=True,type=Path)
    p.add_argument("--payload",required=True,type=Path)
    p.add_argument("--native-samples",required=True,type=Path)
    p.add_argument("--limits",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    result=make_contract(json.loads(a.task.read_text()),
                         Payload(**json.loads(a.payload.read_text())),
                         a.native_samples,output=a.output,
                         limits_source=a.limits)
    print(json.dumps({"status":result["parity"]["status"]},indent=2))


if __name__=="__main__":
    main()
