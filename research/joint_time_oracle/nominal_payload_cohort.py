# SPDX-License-Identifier: Apache-2.0
"""Register many ORIGINAL-effort Panda tasks under heavier *physical* payloads.

This registry is frozen before native results are inspected. Every payload
updates the robot inertia, not its actuator limits. Source task IDs, splits
and attempted mass points—including gravity-rejected cases—are preserved.
A task that native fails to solve is NOT declared solvable: only a separately
independently validated witness can establish a native failure rescue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from .cohort_retime import ROOT, dynamics
from .payload_parity import Payload, apply_pinocchio_payload, nominal_effort_only
from .torque_active import canonical_hash, torque_utilization
from .geometry_feasibility_core import sha256_file

MASS_KG=(0.,.5,1.,2.,3.,5.,7.)
SCENE_SPLITS=("development","confirmation","holdout")
SEED=20261109
ROOT_OUT=ROOT/"research/joint_time_oracle/results/nominal_payload_hardcases_20261109"
BASELINE=ROOT/"research/joint_time_oracle/results/native_reconstruction_20261106/baseline_validation_states.npz"


def split_id(case_id:str,seed:int=SEED)->str:
    """Stable source-disjoint split: all masses of one base task stay together."""
    d=hashlib.sha256(f"{seed}:{case_id}".encode()).digest()
    return SCENE_SPLITS[int.from_bytes(d[:8],"little")%len(SCENE_SPLITS)]


def frozen_json(path:Path,value)->None:
    text=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n"
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and json.loads(path.read_text())!=value:
        raise ValueError("frozen cohort inventory already differs: "+str(path))
    if not path.exists():path.write_text(text)


def endpoints_gravity(payload:Payload,start,goal,nominal:np.ndarray)->dict:
    model,_,neutral,qi,vi=dynamics()
    apply_pinocchio_payload(model,payload)
    data=model.createData()
    zero=np.zeros(model.nv)
    points=np.stack((np.asarray(start,dtype=float),np.asarray(goal,dtype=float)))
    t=[]
    for x in points:
        q=neutral.copy();q[qi]=x
        t.append(pin.rnea(model,data,q,zero,zero)[vi].copy())
    efforts=np.stack(t)
    return {"joint_gravity_nm":efforts.tolist(),
            "maximum_nominal_effort_utilization":float(
                np.max(torque_utilization(efforts,nominal)))}


def create(source:Path,output:Path,*,masses= MASS_KG,
           seed:int=SEED,endpoint_reserve:float=.90)->dict:
    if not 0<endpoint_reserve<.95:
        raise ValueError("endpoint reserve must fit inside native 0.95 margin")
    tasks=json.loads((source/"tasks.json").read_text())
    if len(tasks)!=len({x["case_id"] for x in tasks}):
        raise ValueError("duplicate generated source tasks")
    if len(tasks)<30:
        raise ValueError("too few source tasks for multi-task hard-case discovery")
    with np.load(BASELINE) as baseline:
        nominal=np.asarray(baseline["torque_bounds"],dtype=float)
    nominal_effort_only(nominal,nominal)
    arm=output/"nominal_payload"
    tasks_by_split={s:[] for s in SCENE_SPLITS}
    admitted=[]
    all_attempts=[]
    masses=tuple(float(x) for x in masses)
    if not masses or any(x<0 or x>15 for x in masses):
        raise ValueError("invalid registered payload masses")
    for original in tasks:
        base_id=original["case_id"]
        split=split_id(base_id,seed)
        for mass in masses:
            payload=Payload(mass).record()
            name=f"{base_id}_m{int(mass*1000):05d}"
            case={k:v for k,v in original.items() if k!="task_sha256"}
            case.update({"case_id":name,"base_case_id":base_id,"source_task_sha256":original["task_sha256"],
                         "payload":payload,"regime":"nominal_effort_actual_payload",
                         "split":split,"dynamic_limit_scale":.95,
                         "source_generator_seed":seed})
            case["task_sha256"]=canonical_hash(case)
            gravity=endpoints_gravity(Payload(mass),case["start"],case["goal"],nominal)
            eligible=gravity["maximum_nominal_effort_utilization"]<=endpoint_reserve
            attempt={"case_id":name,"base_case_id":base_id,"split":split,
                     "payload_mass_kg":mass,"task_sha256":case["task_sha256"],
                     "endpoint_gravity":gravity,
                     "admission":"ADMITTED" if eligible else "ENDPOINT_STATIC_GRAVITY_REJECT"}
            all_attempts.append(attempt)
            if not eligible:
                continue
            profile={
                "schema_version":1,"case_id":name,"source_case_id":name,
                "regime":"nominal_effort_actual_payload","derate":1.,
                "nominal_effort_lower":nominal[0].tolist(),
                "nominal_effort_upper":nominal[1].tolist(),
                "effort_lower":nominal[0].tolist(),
                "effort_upper":nominal[1].tolist(),
                "source_task_sha256":case["task_sha256"],
                "source_torque_sha256":sha256_file(BASELINE),
                "payload":payload,"admitted":True,"rejection_reason":None,
                "static_endpoint_utilization":gravity["maximum_nominal_effort_utilization"],
            }
            profile["profile_sha256"]=canonical_hash(profile)
            frozen_json(arm/name/"torque_profile.json",profile)
            admitted.append(case)
            tasks_by_split[split].append(name)
    frozen_json(arm/"tasks.json",admitted)
    frozen_json(output/"all_attempts.json",all_attempts)
    registration={"schema_version":1,"source_tasks_sha256":sha256_file(source/"tasks.json"),
                  "source_generator":str(source.resolve()),"seed":seed,"payload_masses_kg":list(masses),
                  "endpoint_static_reserve":endpoint_reserve,
                  "nominal_effort_sha256":sha256_file(BASELINE),
                  "admitted_count":len(admitted),"attempted_count":len(all_attempts),
                  "split_case_ids":tasks_by_split,
                  "denominator_note":"all source tasks and mass attempts retained; masses of one base task never cross train/confirm/holdout",
                  "model_gate":"must pass independent native/K107 vs Pinocchio RNEA parity before planner results qualify"}
    frozen_json(output/"cohort_registration.json",registration)
    return registration


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--output",type=Path,default=ROOT_OUT)
    p.add_argument("--seed",type=int,default=SEED)
    p.add_argument("--endpoint-reserve",type=float,default=.90)
    a=p.parse_args()
    print(json.dumps(create(a.source,a.output,seed=a.seed,endpoint_reserve=a.endpoint_reserve),
                     indent=2))


if __name__=="__main__":main()
