# SPDX-License-Identifier: Apache-2.0
"""Fail-closed registry of independently proven native hard cases.

A failed native planner request or an oracle numerical return is NEVER
by itself a feasible counterexample. An original-effort hard case needs
a completely observed registered native budget plus an independently
payload-aware collision/dynamics Scaled95-qualified alternative with
exact task/trajectory hashes. A duration-regret witness additionally
requires independent qualification of the native success on the SAME
physical/Scaled95 contract, not its native solver success flag.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .payload_parity_qualify import verify_task_payload_receipt
from .payload_timeopt_contract import (
    OUT,RESTART_BUDGETS,canonical_sha,immutable_json,source_sha,
)

CATEGORIES=("FEASIBLE_RESCUE_NATIVE_BUDGET_FAILURE",
            "NATIVE_SUCCESS_BUT_DURATION_SUBOPTIMAL",
            "NATIVE_SUCCESS_NO_PROVEN_GAP",
            "NATIVE_FAILURE_NOT_PROVEN_SOLVABLE",
            "NATIVE_OR_REFERENCE_ERROR",
            "NOT_FULLY_QUALIFIED",
            "ENDPOINT_GRAVITY_REJECTED")


def verified_witness(task:dict,receipt:dict|None,path:Path|None)->tuple[bool,str]:
    if receipt is None or path is None or not path.is_file():
        return False,"MISSING_INDEPENDENT_WITNESS"
    if (receipt.get("physical") is not True or receipt.get("scaled95") is not True
        or receipt.get("payload_collision_physical") is not True):
        return False,"WITNESS_NOT_PHYSICAL_SCALED95_PAYLOAD_COLLISION_QUALIFIED"
    if receipt.get("task_sha256")!=task["task_sha256"]:
        return False,"WITNESS_TASK_SHA_MISMATCH"
    if receipt.get("trajectory_sha256")!=source_sha(path):
        return False,"WITNESS_TRAJECTORY_SHA_MISMATCH"
    if receipt.get("payload_parity_sha256")!=task["payload_collision_receipt_sha256"]:
        return False,"WITNESS_PAYLOAD_PARITY_SHA_MISMATCH"
    if not isinstance(receipt.get("duration_s"),(float,int)) or receipt["duration_s"]<=0:
        return False,"WITNESS_DURATION_INVALID"
    return True,"INDEPENDENTLY_QUALIFIED"


def native_budget_complete(records:list[dict],*,budget:int)->tuple[bool,str]:
    if budget not in RESTART_BUDGETS or len(records)<budget:
        return False,"NATIVE_RESTART_BUDGET_INCOMPLETE"
    actual=[r["restart"] for r in records[:budget]]
    if actual!=list(range(budget)):
        return False,"NATIVE_RESTART_SEQUENCE_INVALID"
    if any(r.get("status")=="ERROR" for r in records[:budget]):
        return False,"NATIVE_RESTART_ERRORS"
    allowed={"BUDGET_ATTEMPT_FAILED","NATIVE_SUCCESS_UNQUALIFIED"}
    if any(r.get("status") not in allowed for r in records[:budget]):
        return False,"UNRECOGNIZED_NATIVE_RESTART_STATUS"
    return True,"NATIVE_REGISTERED_BUDGET_COMPLETE"


def classify(task:dict,attempts:list[dict],*,budget:int,
             alternative_receipt:dict|None,alternative_path:Path|None,
             native_receipt:dict|None,native_path:Path|None,
             duration_gap_threshold:float=.05)->dict:
    if not 0<duration_gap_threshold<1:
        raise ValueError("invalid duration gap definition")
    verify_task_payload_receipt(task)
    admitted=task.get("admission")=="ENDPOINT_ADMITTED"
    completed,why=native_budget_complete(attempts,budget=budget)
    alternative,reason=verified_witness(task,alternative_receipt,alternative_path)
    native,reason_native=verified_witness(task,native_receipt,native_path)
    native_success=any(r.get("status")=="NATIVE_SUCCESS_UNQUALIFIED"
                       for r in attempts[:budget])
    if not admitted:
        category="ENDPOINT_GRAVITY_REJECTED"
    elif not completed:
        category="NATIVE_OR_REFERENCE_ERROR" if why=="NATIVE_RESTART_ERRORS" else "NOT_FULLY_QUALIFIED"
    elif native_success:
        if not native:
            category="NOT_FULLY_QUALIFIED"
        elif alternative and (
            float(alternative_receipt["duration_s"])<=
            (1-duration_gap_threshold)*float(native_receipt["duration_s"])):
            category="NATIVE_SUCCESS_BUT_DURATION_SUBOPTIMAL"
        else:
            category="NATIVE_SUCCESS_NO_PROVEN_GAP"
    elif alternative:
        category="FEASIBLE_RESCUE_NATIVE_BUDGET_FAILURE"
    else:
        category="NATIVE_FAILURE_NOT_PROVEN_SOLVABLE"
    return {"schema":1,"case_id":task["case_id"],
            "source_task_id":task["source_task_id"],
            "split":task["split"],"payload":task["payload"],
            "category":category,"native_budget":budget,
            "native_budget_completed":completed,
            "native_budget_reason":why,
            "native_success_reported":native_success,
            "native_independently_qualified":native,
            "alternative_independently_qualified":alternative,
            "alternative_reason":reason,
            "native_witness_reason":reason_native,
            "native_qualified_duration_s":native_receipt.get("duration_s") if native else None,
            "alternate_qualified_duration_s":alternative_receipt.get("duration_s") if alternative else None,
            "nominal_effort":True,
            "payload_collision_qualified":True,
            "scope":"finite-restart nominal-effort mass+COM research task; independent sampled physical+Scaled95/robot+payload spheres"}


def build(qualified_root:Path,native_root:Path,
          independent_root:Path,output:Path,*,budget:int=32)->dict:
    if budget not in RESTART_BUDGETS:raise ValueError("invalid budget")
    tasks=json.loads((qualified_root/"qualified_tasks.json").read_text())["tasks"]
    results=[];counts=Counter()
    for task in tasks:
        case=task["case_id"]
        attempts_file=native_root/case/"native_attempts.json"
        attempts=json.loads(attempts_file.read_text()) if attempts_file.exists() else []
        q=independent_root/case
        alt_receipt=q/"alternative_receipt.json"
        native_receipt=q/"native_receipt.json"
        alt_path=q/"alternative_dense.npz"
        native_path=q/"native_dense.npz"
        read=lambda p:json.loads(p.read_text()) if p.exists() else None
        result=classify(task,attempts,budget=budget,
                        alternative_receipt=read(alt_receipt),
                        alternative_path=alt_path if alt_path.exists() else None,
                        native_receipt=read(native_receipt),
                        native_path=native_path if native_path.exists() else None)
        counts[result["category"]]+=1
        results.append(result)
    report={"schema":1,"registered_cases":len(tasks),
            "registered_source_groups":len({t["source_task_id"] for t in tasks}),
            "nominal_actuator_effort":True,
            "native_budget":budget,
            "classification_counts":dict(counts),
            "all_results":results,
            "source_task_registry_sha256":source_sha(qualified_root/"qualified_tasks.json"),
            "source_parity_registry_sha256":source_sha(qualified_root/"qualification_registry.json"),
            "warning":"native planner failure alone is not a feasible hard case; all independent receipts must include payload collision"}
    immutable_json(output/"hardcase_registry.json",report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qualified",type=Path,default=OUT/"qualified")
    p.add_argument("--native",type=Path,default=OUT/"native")
    p.add_argument("--independent",type=Path,default=OUT/"independent")
    p.add_argument("--out",type=Path,default=OUT/"registry")
    p.add_argument("--budget",type=int,default=32)
    args=p.parse_args()
    res=build(args.qualified,args.native,args.independent,args.out,budget=args.budget)
    print(json.dumps(res["classification_counts"],indent=2))


if __name__=="__main__":
    main()
