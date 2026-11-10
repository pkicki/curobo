# SPDX-License-Identifier: Apache-2.0
"""Execute research TimeOpt-feedback search with a real pinned JSON retimer.

The caller supplies an independently verified payload-parity receipt
and signed ORIGINAL actuator limits. No fake backend and no silent
IPOPT fallback. Collision includes independent robot spheres and a
payload-world sphere proxy; full payload SELF-collision parity must be
explicitly attested by the separate native collision receipt. Online
results NEVER count as independently physically certified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .cohort_geometry import QuinticGeometry
from .cohort_retime import ROOT
from .payload_geometry_audit import PayloadGeometryAuditor
from .payload_contract import Payload
from .retimer_oracle import Oracle, make_request, run_oracle
from .timeopt_feedback_search import ARMS, BUDGET, run

OUTPUT=ROOT/"research/joint_time_oracle/results/retimer_guided_payload_20261110"


def initial_quintic(task:dict)->np.ndarray:
    """Frozen straight-line seed when native planning produced no usable cubic."""
    a=np.asarray(task["start"],dtype=float)
    b=np.asarray(task["goal"],dtype=float)
    t=np.linspace(0,1,22)
    # The first/last triples guarantee zero endpoint path derivatives.
    base=a[None]+t[:,None]*(b-a)[None]
    base[:3]=a
    base[-3:]=b
    return base


def inputs(task:dict,payload_doc:dict,contract:dict)->tuple[Payload,dict,dict]:
    p=Payload(**payload_doc).validate()
    model=contract["model"]
    limits=contract["limits"]
    if contract["case_id"]!=task["case_id"] or (
        contract["task_sha256"]!=task["task_sha256"]):
        raise ValueError("task/parity contract mismatch")
    if model.get("nominal_effort_scale")!=1.0:
        raise ValueError("source changed manufacturer effort limits")
    if model.get("payload")!=p.serialize():
        raise ValueError("payload definition differs from model-parity receipt")
    if model.get("native_pinocchio_payload_parity_verified") is not True:
        raise ValueError("native/Pinocchio payload RNEA parity NOT verified")
    if model.get("attached_collision_parity_verified") is not True:
        raise ValueError("attached payload self/world collision parity NOT verified")
    return p,model,limits


def execute(root:Path,task:dict,payload_doc:dict,
            contract:dict,engine:Oracle,*,arm:str,budget:int=BUDGET,
            initial:np.ndarray|None=None)->dict:
    p,model,limits=inputs(task,payload_doc,contract)
    if arm not in ARMS:
        raise ValueError("invalid arm")
    base=initial_quintic(task) if initial is None else np.asarray(initial,dtype=float)
    if base.shape!=(22,7):
        raise ValueError("unsupported initial spline representation")
    auditor=PayloadGeometryAuditor(task["scene"],p)
    case=task["case_id"]
    case_dir=Path(root)/case/engine.name/arm
    case_dir.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    stage_times=[]

    def callback(controls:np.ndarray,iteration:int)->dict:
        stamp=time.perf_counter()
        path=QuinticGeometry(controls)
        q=path.evaluate(np.linspace(0,1,129))[0]
        assessment=auditor.check(q,min_clearance_m=.005)
        geometry_ok=bool(assessment["sampled_acceptable"])
        if not geometry_ok:
            result={"status":"INFEASIBLE_HEURISTIC",
                    "infeasibility_score":float(
                          max(0.,.005-(assessment["world_clearance_m"]
                             if assessment["world_clearance_m"] is not None else -.1))+
                          max(0.,.005-(assessment["self_clearance_m"]
                             if assessment["self_clearance_m"] is not None else -.1))),
                    "reason":"sampled geometry or payload-world collision screening",
                    "independent_qualification":"NOT_PERFORMED"}
        else:
            request=make_request(case,controls,source_sha=task["task_sha256"],
                                 model_contract=model,limit_contract=limits,engine=engine)
            result=run_oracle(request,engine,case_dir/f"proposal_{iteration:04d}")
            if result.get("status")=="FEASIBLE_PRELIMINARY":
                result["independent_qualification"]="PENDING_CPU_AND_GPU_AUDIT"
        stage_times.append({"proposal":iteration,"wall_s":time.perf_counter()-stamp,
                            "retimer_called":geometry_ok,
                            "robot_world_clearance_m":assessment["world_clearance_m"],
                            "robot_self_clearance_m":assessment["self_clearance_m"]})
        return result

    search=run(case,base,callback,arm=arm,budget=budget)
    search.update({"wall_s_including_geometry_and_retimer":time.perf_counter()-start,
                   "engine":engine.name,"engine_commit":engine.engine_commit,
                   "model_contract_sha256":hashlib.sha256(json.dumps(model,
                      sort_keys=True).encode()).hexdigest(),
                   "parity_receipt_status":"EXTERNALLY_VERIFIED_BY_AGENT",
                   "per_proposal_costs":stage_times,
                   "task_sha256":task["task_sha256"],
                   "interpretation":"online search feedback only, no qualified witness until full independent audit"})
    outfile=case_dir/"search.json"
    if outfile.exists() and json.loads(outfile.read_text())!=search:
        raise ValueError("scored search output already exists with different data")
    if not outfile.exists():
        outfile.write_text(json.dumps(search,indent=2,allow_nan=False)+"\n")
    return search


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task",type=Path,required=True)
    ap.add_argument("--payload",type=Path,required=True)
    ap.add_argument("--contract",type=Path,required=True)
    ap.add_argument("--root",type=Path,default=OUTPUT)
    ap.add_argument("--arm",choices=ARMS,required=True)
    ap.add_argument("--engine",choices=("toppra_newton_k107","native_timeopt"),required=True)
    ap.add_argument("--engine-commit",required=True)
    ap.add_argument("--command",nargs="+",required=True)
    ap.add_argument("--budget",type=int,default=BUDGET)
    args=ap.parse_args()
    task=json.loads(args.task.read_text())
    payload=json.loads(args.payload.read_text())
    contract=json.loads(args.contract.read_text())
    retimer=Oracle(args.engine,tuple(args.command),args.engine_commit)
    summary=execute(args.root,task,payload,contract,retimer,
                    arm=args.arm,budget=args.budget)
    print(json.dumps({"task":summary["task_id"],"arm":summary["arm"],
                      "evaluations":summary["oracle_evaluations"],
                      "wall_s":summary["wall_s_including_geometry_and_retimer"]}))


if __name__=="__main__":
    main()
