# SPDX-License-Identifier: Apache-2.0
"""Reproducible independent Pinocchio vs cuRobo CUDA RNEA payload parity.

Three process modes keep the CPU Pinocchio and CUDA retimer environments
isolated: generate -> pin, native -> compare. This is a required hard
gate before any native/K107 planning success is declared qualified.
Only frozen input masses/COM, model/limit SHAs and the same q,qd,qdd
may be compared. Use complete dynamic states (not gravity alone).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .cohort_retime import ROOT
from .geometry_feasibility_core import sha256_file
from .payload_parity import Payload, apply_pinocchio_payload, rnea_parity
from .torque_active import read_profile

FILES=(
    ROOT/"curobo/content/assets/robot/franka_description/franka_panda.urdf",
    ROOT/"curobo/content/configs/robot/franka.yml",
    ROOT/"research/joint_time_oracle/results/native_reconstruction_20261106/baseline_validation_states.npz"
)


def init(task:dict,profile:dict,output:Path,*,seed:int=20261109)->None:
    import pinocchio as pin
    from .cohort_retime import dynamics
    model,_,neutral,qi,vi=dynamics()
    from .payload_parity import apply_pinocchio_payload
    p=Payload(**{k:v for k,v in task["payload"].items() if k!="payload_sha256"})
    apply_pinocchio_payload(model,p)
    if task["task_sha256"]!=profile["source_task_sha256"]:
        raise ValueError("source task/profile mismatch")
    rng=np.random.default_rng(seed)
    with np.load(FILES[2]) as f:
        lower,upper=f["position_bounds"]
    home=np.asarray(task["start"])
    goal=np.asarray(task["goal"])
    qs=np.vstack((home,goal,rng.uniform(lower+.15,upper-.15,size=(18,7))))
    vs=np.vstack((np.zeros((2,7)),rng.normal(0,.4,size=(18,7))))
    acc=np.vstack((np.zeros((2,7)),rng.normal(0,.8,size=(18,7))))
    output.mkdir(parents=True,exist_ok=True)
    path=output/"states.npz"
    if path.exists():raise ValueError("states are frozen; do not regenerate in-place")
    np.savez_compressed(path,q=qs,qd=vs,qdd=acc)
    meta={"task_sha256":task["task_sha256"],"payload":task["payload"],
          "profile_sha256":profile["profile_sha256"],
          "model_inputs_sha256":{str(p.resolve()):sha256_file(p) for p in FILES},
          "states_sha256":sha256_file(path),"seed":seed}
    (output/"parity_input.json").write_text(json.dumps(meta,indent=2)+"\n")


def pin_run(output:Path,task:dict):
    import pinocchio as pin
    from .cohort_retime import dynamics
    meta=json.loads((output/"parity_input.json").read_text())
    model,data,neutral,qi,vi=dynamics()
    apply_pinocchio_payload(model,Payload(**{
        k:v for k,v in task["payload"].items() if k!="payload_sha256"}))
    data=model.createData()
    with np.load(output/"states.npz") as z:
        q,qd,qdd=(np.asarray(z[k]) for k in ("q","qd","qdd"))
    torques=[]
    for x,v,a in zip(q,qd,qdd):
        state=neutral.copy();state[qi]=x
        fullv=np.zeros(model.nv);fulla=np.zeros(model.nv)
        fullv[vi]=v;fulla[vi]=a
        torques.append(pin.rnea(model,data,state,fullv,fulla)[vi].copy())
    dest=output/"pin_torques.npz"
    if dest.exists():raise ValueError("cannot overwrite independent Pinocchio reference")
    np.savez_compressed(dest,torque=np.asarray(torques))
    return sha256_file(dest)


def native_run(output:Path,task:dict,profile:dict,retimer_repo:Path):
    import torch
    sys.path.insert(0,str((retimer_repo/"scripts").resolve()))
    sys.path.insert(0,str((retimer_repo/"src").resolve()))
    from joint_time_oracle_k107_bridge import prepare_robot
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for native parity")
    with np.load(output/"states.npz") as z:
        arrays={k:np.asarray(z[k]) for k in ("q","qd","qdd")}
    nominal=np.stack((profile["nominal_effort_lower"],profile["nominal_effort_upper"]))
    model,_=prepare_robot(task["payload"],nominal)
    states=[torch.as_tensor(arrays[k],device="cuda",dtype=torch.float32)
            for k in ("q","qd","qdd")]
    torques=model.dynamics.inverse_dynamics(*states).detach().cpu().numpy()
    dest=output/"native_torques.npz"
    if dest.exists():raise ValueError("cannot overwrite native CUDA reference")
    np.savez_compressed(dest,torque=torques)
    return sha256_file(dest)


def finish(output:Path,task:dict)->dict:
    meta=json.loads((output/"parity_input.json").read_text())
    if meta["task_sha256"]!=task["task_sha256"] or meta["payload"]!=task["payload"]:
        raise ValueError("wrong task/payload parity input")
    if meta["states_sha256"]!=sha256_file(output/"states.npz"):
        raise ValueError("frozen parity states have changed")
    if any(sha256_file(p)!=v for p,v in
           ((Path(k),x) for k,x in meta["model_inputs_sha256"].items())):
        raise ValueError("model or actuator bounds changed after parity generation")
    with np.load(output/"pin_torques.npz") as z:
        reference=np.asarray(z["torque"])
    with np.load(output/"native_torques.npz") as z:
        native=np.asarray(z["torque"])
    diag=rnea_parity(native,reference)
    result={"schema_version":1,"status":"PASS" if diag["passed"] else "FAIL",
            **diag,"task_sha256":task["task_sha256"],"payload":task["payload"],
            "model_inputs_sha256":meta["model_inputs_sha256"],
            "states_sha256":meta["states_sha256"],
            "pin_sha256":sha256_file(output/"pin_torques.npz"),
            "native_sha256":sha256_file(output/"native_torques.npz")}
    path=output/"parity_receipt.json"
    if path.exists():raise ValueError("parity receipt must be frozen")
    path.write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("generate","pin","native","compare"),required=True)
    p.add_argument("--cohort",type=Path,required=True)
    p.add_argument("--case",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--retimer-repo",type=Path)
    a=p.parse_args()
    arm=a.cohort/"nominal_payload"
    case=next(x for x in json.loads((arm/"tasks.json").read_text()) if x["case_id"]==a.case)
    profile=read_profile(arm/a.case/"torque_profile.json")
    if a.stage=="generate":init(case,profile,a.output);out={"stage":a.stage}
    elif a.stage=="pin":out={"pin_sha256":pin_run(a.output,case)}
    elif a.stage=="native":
        if a.retimer_repo is None:p.error("native mode requires --retimer-repo")
        out={"native_sha256":native_run(a.output,case,profile,a.retimer_repo)}
    else:out=finish(a.output,case)
    print(json.dumps(out,indent=2))


if __name__=="__main__":main()
