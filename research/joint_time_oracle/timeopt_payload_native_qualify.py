# SPDX-License-Identifier: Apache-2.0
"""Independent payload-aware native success qualification on S8193/S32768.

An internal cuRobo native success is NOT a qualified scientific success
until fresh exact-GPU-replayed S8193/S32768 samples pass Pinocchio
inverse-dynamics, signed original physical + Scaled95 limits,
robot/world/self spheres, attached payload world/self collision,
and strict endpoint tests. Run CPU independent audit in pinned Pinocchio
environment, separate from the native CUDA replay process.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cohort_retime import ROOT
from .geometry_feasibility_scene import IndependentGeometryAuditor
from .timeopt_k107_independent_audit import max_bounds,_torques
from .timeopt_payload_collision import payload_clearances
from .timeopt_payload_model import Payload,file_sha256
from .timeopt_payload_tasks import OUT


def _family_limits(task:dict)->dict:
    path=ROOT/"research/joint_time_oracle/results/native_reconstruction_20261106/baseline_validation_states.npz"
    with np.load(path) as source:
        lim={k:np.asarray(source[k+"_bounds"]) for k in
             ("position","velocity","acceleration","jerk")}
    lim["torque"]=np.stack((task["original_effort_lower"],
                           task["original_effort_upper"]))
    return lim


def audit_states(task:dict,values:dict)->dict:
    payload=Payload(**{k:task["payload"][k] for k in (
        "mass_kg","radius_m","native_link","pin_frame")})
    samples=len(values["q"])
    if samples<8193 or any(np.asarray(v).shape!=(samples,7) for k,v in
                          values.items() if k in ("q","qd","qdd","jerk")):
        raise ValueError("native audit needs dense [S,7] state arrays")
    if not all(np.isfinite(v).all() for v in values.values()):
        return {"qualified":False,"status":"NONFINITE_NATIVE_SAMPLED_STATE"}
    auditor=IndependentGeometryAuditor(task["scene"])
    limits=_family_limits(task)
    geometry=auditor.check(values["q"],min_clearance_m=.005)
    payload_geometry=payload_clearances(
        values["q"],payload=payload,auditor=auditor,min_clearance_m=.005)
    tau=_torques(values,payload)
    families={}
    for family,arr in (("position",values["q"]),("velocity",values["qd"]),
                       ("acceleration",values["qdd"]),("jerk",values["jerk"]),
                       ("torque",tau)):
        bounds=limits[family]
        if family=="position":
            physical=scaled=max_bounds(arr,bounds,scaled=False)
        elif family=="torque":
            physical=max_bounds(arr,bounds,scaled=False)
            scaled=max_bounds(arr,bounds*.95,scaled=False)
        else:
            physical=max_bounds(arr,bounds,scaled=False)
            scaled=max_bounds(arr,bounds,scaled=True)
        families[family]={"physical":physical,"scaled95":scaled}
    endpoint_q=max(float(np.max(abs(values["q"][0]-task["start"]))),
                   float(np.max(abs(values["q"][-1]-task["goal"]))))
    endpoint_speed=float(np.max(abs(values["qd"][[0,-1]])))
    passed=bool(geometry["sampled_acceptable"] and
                payload_geometry["sampled_acceptable"] and
                endpoint_q<=1e-3 and endpoint_speed<=1e-4 and
                all(item[mode]["max_violation"]<=1e-6
                    for item in families.values() for mode in ("physical","scaled95")))
    return {"qualified":passed,
            "status":"INDEPENDENT_SAMPLED_PHYSICAL_SCALED95_PASS" if passed else
                     "INDEPENDENT_SAMPLED_FAIL",
            "samples":samples,
            "duration_s":float(values["time"][-1]),
            "families":families,
            "robot_world_clearance_m":geometry["world_clearance_m"],
            "robot_self_clearance_m":geometry["self_clearance_m"],
            "payload_world_clearance_m":payload_geometry["world_clearance_m"],
            "payload_self_clearance_m":payload_geometry["self_clearance_m"],
            "endpoint_position_rad":endpoint_q,
            "endpoint_speed_rad_s":endpoint_speed,
            "scope":"independent Pinocchio sampled+exact native replay, NOT continuous-time certificate"}


def qualify(root:Path,*,case_filter:str|None=None,
            parity_receipt:Path)->dict:
    receipt=json.loads(parity_receipt.read_text())
    if not receipt.get("model_and_collision_parity_pass"):
        raise ValueError("nominal payload model+collision parity missing")
    tasks=json.loads((root/"tasks.json").read_text())
    result={}
    for task in tasks:
        case=task["case_id"]
        if case_filter and case!=case_filter:continue
        if task["payload"]["payload_sha256"] not in receipt["payload_sha256_qualified"]:
            raise ValueError("payload model not independently qualified: "+case)
        folder=root/"native_full"/case
        if not (folder/"native_candidates.json").exists():
            continue
        inventory=json.loads((folder/"native_candidates.json").read_text())
        for tag,record in inventory.items():
            key=case+":"+tag
            if record["status"]!="SUCCESS_NATIVE_UNQUALIFIED":
                continue
            entry={"case_id":case,"tag":tag,"qualified":False}
            try:
                stages={}
                for count in (8193,32768):
                    path=folder/(tag.lower()+
                         ("_dense.npz" if count==8193 else "_s32768.npz"))
                    if not path.exists():
                        raise FileNotFoundError("exact native replay missing "+str(path))
                    digest=file_sha256(path)
                    if count==8193 and digest!=record["dense_sha256"]:
                        raise ValueError("native original S8193 source changed")
                    with np.load(path) as z:
                        values={k:np.asarray(z[k]) for k in z.files}
                    if len(values["q"])<count:
                        raise ValueError("insufficient independently audited samples")
                    stage=audit_states(task,values)
                    stage["source_sha256"]=digest
                    stages["S"+str(count)]=stage
                entry.update({"stages":stages,
                              "qualified":bool(stages["S8193"]["qualified"] and
                                               stages["S32768"]["qualified"]),
                              "duration_s":stages["S32768"]["duration_s"],
                              "source_task_sha256":task["task_sha256"],
                              "payload_sha256":task["payload"]["payload_sha256"]})
            except Exception as exc:
                entry.update({"reason":repr(exc),"status":"UNQUALIFIED_OR_MISSING_EVIDENCE"})
            result[key]=entry
    output={"schema_version":1,
            "model_parity_receipt_sha256":file_sha256(parity_receipt),
            "qualified_count":sum(x["qualified"] for x in result.values()),
            "attempted_native_success_candidates":len(result),
            "candidates":result,
            "scope":"dense/event-aware sampled validation only; native successes not autoqualified"}
    dest=root/"native_full"/"independent_payload_qualification.json"
    if dest.exists() and json.loads(dest.read_text())!=output:
        raise ValueError("native independent validation inventory changed")
    if not dest.exists():dest.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUT)
    p.add_argument("--case")
    p.add_argument("--parity-receipt",type=Path,required=True)
    a=p.parse_args()
    report=qualify(a.root,case_filter=a.case,
                   parity_receipt=a.parity_receipt)
    print(json.dumps({"qualified":report["qualified_count"],
                      "attempted":report["attempted_native_success_candidates"]},indent=2))


if __name__=="__main__":
    main()
