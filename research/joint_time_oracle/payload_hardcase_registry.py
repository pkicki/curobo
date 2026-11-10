# SPDX-License-Identifier: Apache-2.0
"""Fail-closed hard-case registry for nominal-effort payload planning.

A native failure is NOT a confirmed hard case unless an independently
qualified alternative with the SAME exact scene, start/goal, attached
payload inertia and geometry, original signed actuator limits, endpoint
contract and physical constraints is found. Native suboptimality
requires BOTH independently qualified trajectories and a registered
duration gap. An LP zero is never a proof of feasible timing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def hash_json(data)->str:
    return hashlib.sha256(json.dumps(data,sort_keys=True,
             separators=(",",":"),allow_nan=False).encode()).hexdigest()


def population_split(task_id:str,*,seed:int=20261110)->str:
    # Independent of planner outcomes. Preserve entire task across
    # mass/COM scenarios to avoid source leakage into holdout.
    x=int(hashlib.sha256(f"{task_id}|{seed}".encode()).hexdigest()[:12],16)
    return "confirm" if x%5==0 else "development"


def same_contract(native:dict,alternative:dict)->bool:
    required=("task_sha256","payload_sha256","native_model_sha256",
              "collision_model_sha256","effort_contract_sha256",
              "constraint_contract_sha256","endpoint_contract_sha256")
    return all(k in native and native[k] is not None and
               native[k]==alternative.get(k) for k in required)


def qualified(entry:dict)->bool:
    audit=entry.get("independent_audit") or {}
    return bool(audit.get("physical") is True and
                audit.get("scaled95") is True and
                audit.get("collision") is True and
                audit.get("endpoint") is True and
                audit.get("dense_samples",0)>=8193 and
                audit.get("adaptive_samples",0)>=8193 and
                audit.get("source_trajectory_sha256") and
                audit.get("model_alignment_pass") is True)


def classify(native:dict,alternative:dict|None,*,gap:float=.05)->dict:
    if not 0<gap<1:
        raise ValueError("invalid registered meaningful duration gap")
    profile=native.get("profile")
    if not isinstance(profile,dict) or profile.get("effort_scale")!=1.:
        return {"status":"BLOCKED_NOT_ORIGINAL_ACTUATOR_LIMITS"}
    if not native.get("payload_dynamics_parity_pass"):
        return {"status":"BLOCKED_NATIVE_PAYLOAD_DYNAMICS_MISMATCH"}
    if not native.get("attached_collision_parity_pass"):
        return {"status":"BLOCKED_ATTACHED_OBJECT_COLLISION_UNVERIFIED"}
    if native.get("endpoint_static_torque_admissible") is False:
        return {"status":"ENDPOINT_GRAVITY_INADMISSIBLE_NOT_A_PLANNER_FAILURE"}
    if native.get("status")=="ERROR":
        return {"status":"NATIVE_RUNTIME_ERROR_UNRESOLVED"}
    other=alternative or {}
    if other and not same_contract(native,other):
        return {"status":"INCOMPARABLE_CONSTRAINT_OR_PAYLOAD_CONTRACT"}
    native_ok=qualified(native)
    other_ok=qualified(other) if other else False
    if not native_ok and other_ok:
        if native.get("status") not in ("NO_VERIFIED_SOLUTION","FAILED_WITHIN_BUDGET"):
            return {"status":"NATIVE_RESULT_UNRESOLVED"}
        return {"status":"HARD_NATIVE_FAILURE_WITH_FEASIBLE_ALTERNATIVE",
                "alternative_duration_s":other.get("duration_s")}
    if native_ok and other_ok:
        t0=float(native.get("duration_s",float("nan")))
        t1=float(other.get("duration_s",float("nan")))
        if not (t0>0 and t1>0):
            return {"status":"INVALID_DURATION"}
        improvement=1-t1/t0
        return {"status":("HARD_NATIVE_SUBOPTIMALITY" if improvement>=gap
                          else "NATIVE_SUCCESS_NO_REGISTERED_GAP"),
                "relative_duration_improvement":improvement}
    if native_ok:
        return {"status":"NATIVE_QUALIFIED_NO_BETTER_ALTERNATIVE"}
    return {"status":"NATIVE_FAILURE_NO_INDEPENDENT_EXISTENCE_WITNESS"}


def classify_rows(rows:list[dict])->dict:
    result=[]
    for row in rows:
        task=row["task_id"]
        native=row["native"]
        alternative=row.get("alternative")
        outcome=classify(native,alternative)
        result.append({"task_id":task,"scenario_id":row["scenario_id"],
                       "split":population_split(task),
                       "cohort":row.get("cohort"),
                       "conclusion":outcome,
                       "source":row.get("source")})
    counts={}
    for item in result:
        status=item["conclusion"]["status"]
        counts[status]=counts.get(status,0)+1
    return {"schema_version":1,"counts":counts,"rows":result,
            "note":"classifies only source- and constraint-matched independently audited trajectories; no missing result is inferred feasible"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    rows=json.loads(a.input.read_text())
    report=classify_rows(rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report["counts"],indent=2))


if __name__=="__main__":
    main()
