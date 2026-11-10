# SPDX-License-Identifier: Apache-2.0
"""Independent mass-aware Pinocchio+attached-sphere physical/Scaled95 audit.

This is deliberately NOT cohort_validate: that historical validator uses
the no-payload model and ignores attached-object sphere geometry. Trajectory
must be generated from frozen exact geometry and timing controls externally.
A positive result is SAMPLED dense+adaptive physical qualification, never
continuous-time/hardware safety or proof of global time optimality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from .cohort_retime import ROOT
from .cohort_torque_report import TOLERANCES,ENDPOINT_TOL
from .payload_parity_qualify import verify_task_payload_receipt
from .payload_model_parity import independent_payload_auditor
from .payload_timeopt_contract import Payload,immutable_json,source_sha


def _family(values:np.ndarray,bounds:np.ndarray)->dict:
    lo,hi=np.asarray(bounds,dtype=float)
    z=np.asarray(values,dtype=float)
    excess=np.maximum(lo[None]-z,z-hi[None])
    index,joint=np.unravel_index(int(excess.argmax()),excess.shape)
    return {"violation":float(max(0.,excess[index,joint])),
            "signed_residual":float(excess[index,joint]),
            "joint":int(joint+1),"sample":int(index)}


def _signed95(bounds:np.ndarray,family:str)->np.ndarray:
    lo,hi=np.asarray(bounds,dtype=float)
    if family=="position":return np.asarray(bounds).copy()
    if family=="torque":return np.stack((lo*.95,hi*.95))
    center=(lo+hi)/2
    return np.stack((center+.95*(lo-center),center+.95*(hi-center)))


def audit(task:dict,trajectory:Path,*,min_samples:int=8193)->dict:
    verify_task_payload_receipt(task)
    path=Path(trajectory)
    if not path.is_file():raise FileNotFoundError(path)
    input_hash=source_sha(path)
    with np.load(path) as values:
        raw={k:np.asarray(values[k],dtype=float) for k in values.files
             if k in ("time","q","qd","qdd","jerk","progress")}
    required=("time","q","qd","qdd","jerk")
    if any(k not in raw for k in required):raise ValueError("missing dense joint states")
    t=raw["time"];q=raw["q"];v=raw["qd"];a=raw["qdd"];j=raw["jerk"]
    if (len(t)<min_samples or any(z.shape!=(len(t),7) for z in (q,v,a,j))
        or not all(np.isfinite(z).all() for z in (t,q,v,a,j))
        or np.any(np.diff(t)<=0) or abs(t[0])>1e-8):
        raise ValueError("invalid dense physical states or clock")
    payload=Payload(**task["payload"])
    auditor=independent_payload_auditor(payload,task["scene"])
    limits_path=ROOT/"research/joint_time_oracle/results/native_reconstruction_20261106/baseline_validation_states.npz"
    with np.load(limits_path) as src:
        limits={name:np.asarray(src[name+"_bounds"],dtype=float) for name in
                ("position","velocity","acceleration","jerk","torque")}
    torque_bounds=limits["torque"]
    model=auditor.model;data=auditor.data;state=auditor.neutral.copy()
    vel=np.zeros(model.nv);acc=np.zeros(model.nv);tau=np.empty_like(q)
    for i,config in enumerate(q):
        state[auditor.qi]=config
        vel.fill(0);acc.fill(0)
        vel[auditor.vi]=v[i];acc[auditor.vi]=a[i]
        tau[i]=pin.rnea(model,data,state,vel,acc)[auditor.vi]
    if source_sha(path)!=input_hash:raise ValueError("trajectory changed during independent audit")
    collision=auditor.check(q,min_clearance_m=0.)
    family={}
    for name,arr in (("position",q),("velocity",v),("acceleration",a),
                     ("jerk",j),("torque",tau)):
        family[name]={
          "physical":_family(arr,limits[name]),
          "scaled95":_family(arr,_signed95(limits[name],name))}
    endpoints={"position_start":float(np.max(np.abs(q[0]-task["start"]))),
               "position_goal":float(np.max(np.abs(q[-1]-task["goal"]))),
               "velocity":float(np.max(np.abs(v[[0,-1]]))),
               "acceleration":float(np.max(np.abs(a[[0,-1]])))}
    endpoint_ok=all(endpoints[name]<=threshold
                     for name,threshold in ENDPOINT_TOL.items())
    collision_ok=bool(collision["sampled_acceptable"])
    results={name:all(family[key][name]["violation"]<=tol for key,tol in TOLERANCES.items())
             for name in ("physical","scaled95")}
    physical=bool(results["physical"] and endpoint_ok and collision_ok)
    scaled=bool(physical and results["scaled95"])
    return {"schema":1,"case_id":task["case_id"],
            "task_sha256":task["task_sha256"],
            "trajectory_sha256":input_hash,
            "payload_parity_sha256":task["payload_collision_receipt_sha256"],
            "payload_collision_physical":collision_ok,
            "samples":len(t),"duration_s":float(t[-1]),
            "physical":physical,"scaled95":scaled,
            "endpoints":endpoints,"families":family,
            "world_clearance_m":collision["world_clearance_m"],
            "self_clearance_m":collision["self_clearance_m"],
            "original_actuator_limits":True,
            "source_physical_limits_sha256":source_sha(limits_path),
            "qualification":"INDEPENDENT_SAMPLED_PINOCCHIO_RNEA_WITH_ATTACHED_FOUR_SPHERES",
            "caution":"not a continuous-time collision/torque-rate or hardware safety certificate"}


def combine(base:dict,adaptive:dict)->dict:
    """Both archives must independently pass all physical/Scaled95 families."""
    keys=("task_sha256","payload_parity_sha256","case_id")
    if any(base.get(k)!=adaptive.get(k) for k in keys):
        raise ValueError("dense/adaptive task or payload provenance mismatch")
    if base.get("trajectory_sha256")==adaptive.get("trajectory_sha256"):
        raise ValueError("adaptive archive must be distinct exact replay")
    return {"schema":1,"case_id":base["case_id"],
            "task_sha256":base["task_sha256"],
            "payload_parity_sha256":base["payload_parity_sha256"],
            "physical":bool(base["physical"] and adaptive["physical"]),
            "scaled95":bool(base["scaled95"] and adaptive["scaled95"]),
            "payload_collision_physical":bool(
                base["payload_collision_physical"] and adaptive["payload_collision_physical"]),
            "base_samples":base["samples"],"adaptive_samples":adaptive["samples"],
            "base_trajectory_sha256":base["trajectory_sha256"],
            "adaptive_trajectory_sha256":adaptive["trajectory_sha256"],
            "duration_s":base["duration_s"],
            "scope":"two independent sampled trajectories; continuous-time proof unavailable"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",type=Path,required=True)
    p.add_argument("--dense",type=Path,required=True)
    p.add_argument("--adaptive",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    task=json.loads(a.task.read_text())
    base=audit(task,a.dense,min_samples=8193)
    if a.adaptive is not None:
        follow=audit(task,a.adaptive,min_samples=8193)
        receipt=combine(base,follow)
        receipt["trajectory_sha256"]=base["trajectory_sha256"]
    else:
        receipt={**base,"physical":False,"scaled95":False,
                 "payload_collision_physical":False,
                 "scope":"dense pilot only, ADAPTIVE MISSING, never count qualified"}
    immutable_json(a.out,receipt)
    print(json.dumps({"physical":receipt["physical"],
                      "scaled95":receipt["scaled95"],
                      "case_id":receipt["case_id"]},indent=2))


if __name__=="__main__":
    main()
