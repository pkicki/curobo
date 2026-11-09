# SPDX-License-Identifier: Apache-2.0
"""Run matched-budget TimeOpt-guided global geometry search with real K107.

Requires qualified payload physics/collision parity, a frozen nominal
actuator task registry, and a persistent actual TOPP-RA+Newton K107
worker. The online K107 screen alone never qualifies a trajectory;
a separate independent Pinocchio dense/event-aware audit with the
same PAYLOAD inertia and nominal effort is mandatory.

No historical feasible W controls are ever used to initialize search.
The three research arms differ only in path-proposal selection/feedback.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .cohort_retime import ROOT
from .geometry_feasibility_scene import IndependentGeometryAuditor
from .timeopt_feedback_search import ARMS,BUDGETS,nominal_path,search
from .timeopt_k107_service import K107Service
from .timeopt_oracle_contract import K107_ROUTE,request
from .timeopt_payload_model import (
    Payload, file_sha256, nominal_effort_bounds,
)
from .timeopt_payload_tasks import OUT
from .timeopt_payload_collision import payload_clearances

RESULTS=ROOT/"research/joint_time_oracle/results/timeopt_feedback_20261109"


def common_limits(task:dict)->dict:
    file=ROOT/("research/joint_time_oracle/results/"
               "native_reconstruction_20261106/baseline_validation_states.npz")
    with np.load(file) as d:
        limits={k:np.asarray(d[k+"_bounds"]).tolist() for k in
                ("velocity","acceleration","jerk")}
    limits["torque"]=np.stack((
        task["original_effort_lower"],task["original_effort_upper"]
    )).tolist()
    return limits


def run(*,source:Path,output:Path,case:str,arm:str,budget:int,
        worker_command:str,parity_receipt:Path,seed:int=20261109,
        max_wall_s:float=60.,hard_case_registry:Path|None=None,
        pilot:bool=False,holdout:bool=False,
        timeopt_sha:str|None=None)->dict:
    if arm not in ARMS or budget not in BUDGETS:
        raise ValueError("unregistered arm or budget")
    tasks=json.loads((source/"tasks.json").read_text())
    task=next((x for x in tasks if x["case_id"]==case),None)
    if task is None:
        raise ValueError("case absent from frozen source inventory")
    if task.get("allowed_for_native_challenge") is not True:
        raise ValueError("trivially inadmissible endpoint is not a planning hard case")
    parity=json.loads(parity_receipt.read_text())
    if not parity.get("model_and_collision_parity_pass") or (
        task["payload"]["payload_sha256"] not in parity.get("payload_sha256_qualified",[])
    ):
        raise ValueError("payload dynamics+collision parity gate not met")
    if not pilot and task["separation_group"]=="holdout":
        if not holdout or hard_case_registry is None or not timeopt_sha:
            raise ValueError("scored holdout requires frozen method and K107 revision")
        from .timeopt_campaign_freeze import verify
        verify(source=source,results=output,registry=hard_case_registry,
               parity=parity_receipt,other_revision=timeopt_sha)
    elif holdout and not pilot:
        raise ValueError("development task may not use --holdout")
    if not pilot:
        if hard_case_registry is None:
            raise ValueError("scored search requires independently qualified hard-case registry")
        hard=json.loads(hard_case_registry.read_text())
        matches=[x for x in hard.get("cases",[]) if x.get("case_id")==case]
        if not matches or matches[0]["class"] not in (
            "NATIVE_FAILED_ALTERNATIVE_QUALIFIED",
            "NATIVE_FEASIBLE_BUT_DURATION_SUBOPTIMAL",
        ):
            raise ValueError("case does not have source-disjoint qualified hard-case evidence")
    mass=Payload(**{k:task["payload"][k] for k in (
        "mass_kg","radius_m","native_link","pin_frame")})
    base=nominal_path(task["start"],task["goal"])
    limits=common_limits(task)
    if not np.array_equal(np.asarray(limits["torque"]),nominal_effort_bounds()):
        raise ValueError("effort changed relative to frozen Panda nominal")
    auditor=IndependentGeometryAuditor(task["scene"])
    folder=output/arm/case
    folder.mkdir(parents=True,exist_ok=True)
    dest=folder/"search_result.json"
    if dest.exists():
        raise ValueError("search output already exists: frozen runs are immutable")
    requests_folder=folder/"oracle"
    geometry_rejects={"robot":0,"payload":0}
    indexed=[]
    with K107Service(worker_command,requests_folder) as worker:
        # Untimed request warms CUDA and verifies real-worker schema. It
        # does not enter the policy archive or influence its proposals.
        warm=request(task,base,request_id="warmup_000",
                     retimer=K107_ROUTE,limit_contract=limits)
        baseline=worker.evaluate(warm)
        if baseline["status"] in ("ERROR","UNSUPPORTED"):
            raise RuntimeError("K107 worker warm-up cannot satisfy contract: "+
                               str(baseline.get("reason")))
        sync_begin=time.perf_counter()
        def oracle(controls:np.ndarray,idx:int)->dict:
            geom=__import__("research.joint_time_oracle.cohort_geometry",
                            fromlist=["QuinticGeometry"]).QuinticGeometry(controls)
            sampled=np.asarray(geom.evaluate(np.linspace(0,1,129))[0])
            pos=auditor.check(sampled,min_clearance_m=.005)
            if not pos["sampled_acceptable"]:
                geometry_rejects["robot"]+=1
                return {"status":"INFEASIBLE_SCREENED",
                        "diagnostics":{"max_normalized_violation":2.,
                                       "diagnostic_valid":False},
                        "reason":"GEOMETRY_OR_COLLISION_PRESCREEN",
                        "k107_called":False}
            payload_check=payload_clearances(sampled,payload=mass,
                                               auditor=auditor,min_clearance_m=.005)
            if not payload_check["sampled_acceptable"]:
                geometry_rejects["payload"]+=1
                return {"status":"INFEASIBLE_SCREENED",
                        "diagnostics":{"max_normalized_violation":2.,
                                       "diagnostic_valid":False},
                        "reason":"PAYLOAD_WORLD_OR_SELF_COLLISION",
                        "payload_clearance":payload_check,"k107_called":False}
            req=request(task,controls,
                        request_id=f"{case}_{arm}_{idx:04d}",
                        retimer=K107_ROUTE,limit_contract=limits)
            row=worker.evaluate(req)
            row["k107_called"]=True
            return row
        result=search(base,oracle,arm=arm,task_id=case,seed=seed,
                      budget=budget,max_wall_s=max_wall_s)
        result["search_wall_s_inside_warm_service"]=time.perf_counter()-sync_begin
        result["service_warmup_response_status"]=baseline["status"]
        result["service_warmup_wall_ms"]=baseline.get("subprocess_wall_ms")
    result.update({
        "source_task_sha256":task["task_sha256"],
        "source_inventory_sha256":file_sha256(source/"tasks.json"),
        "payload_sha256":mass.as_record()["payload_sha256"],
        "parity_receipt_sha256":file_sha256(parity_receipt),
        "hard_case_registry_sha256":file_sha256(hard_case_registry)
            if hard_case_registry is not None else None,
        "source_effort_unmodified":True,
        "model_parity_gate":True,
        "oracle_worker":"pinned K107 TOPP-RA+Newton",
        "pilot_only_not_scored":pilot,
        "geometry_prefilter_rejections":geometry_rejects,
        "qualified_success_count":0,
        "qualification_pending":"payload-aware independent S8193/event S32768 verification",
    })
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=OUT)
    p.add_argument("--output",type=Path,default=RESULTS)
    p.add_argument("--case",required=True)
    p.add_argument("--arm",choices=ARMS,required=True)
    p.add_argument("--budget",choices=BUDGETS,required=True,type=int)
    p.add_argument("--seed",type=int,default=20261109)
    p.add_argument("--max-wall-s",type=float,default=60.)
    p.add_argument("--worker-command",required=True)
    p.add_argument("--parity-receipt",type=Path,required=True)
    p.add_argument("--hard-case-registry",type=Path)
    p.add_argument("--pilot",action="store_true")
    p.add_argument("--holdout",action="store_true")
    p.add_argument("--timeopt-sha",help="full SHA of pinned K107 repository revision")
    a=p.parse_args()
    r=run(source=a.source,output=a.output,case=a.case,arm=a.arm,
          budget=a.budget,seed=a.seed,max_wall_s=a.max_wall_s,
          worker_command=a.worker_command,
          parity_receipt=a.parity_receipt,
          hard_case_registry=a.hard_case_registry,pilot=a.pilot,
          holdout=a.holdout,timeopt_sha=a.timeopt_sha)
    print(json.dumps({k:v for k,v in r.items() if k not in
                      ("ledger","best_controls")},indent=2))


if __name__=="__main__":
    main()
