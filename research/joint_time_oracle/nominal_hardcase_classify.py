# SPDX-License-Identifier: Apache-2.0
"""Strict native-planner failure/suboptimality classification, original effort.

A hard failure case is not certified by a native planner error alone:
- all registered native attempts complete (FAILED or SUCCESS, no errors);
- native has no INDEPENDENTLY physical+Scaled95 qualified trajectory;
- a DIFFERENT geometric trajectory is independently qualified under the
  identical true-payload model/effort/scene/endpoint contract.
A suboptimality case requires independent qualification of BOTH
native and alternative. Keep unresolved cases and all failures in
the denominator, never select holdout results retrospectively.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cohort_torque_report import sampled_acceptance
from .geometry_feasibility_core import sha256_file
from .torque_active import read_profile

FULL_NATIVE_RESTARTS=32
DURATION_GAIN=.05


def native_audit_inventory(cohort:Path)->dict:
    """Create a compatible pilot.jsonl for independent F000..F031 replay."""
    arm=cohort/"nominal_payload"
    native=arm/"native_full"
    tasks=json.loads((arm/"tasks.json").read_text())
    if not (native/"tasks.json").exists():
        raise FileNotFoundError("native planner results unavailable")
    rows=[]
    for task in tasks:
        file=native/task["case_id"]/"native_candidates.json"
        history=json.loads(file.read_text()) if file.exists() else {}
        methods={}
        for restart in range(FULL_NATIVE_RESTARTS):
            tag=f"F{restart:03d}"
            data=history.get(tag)
            if data and data.get("status")=="SUCCESS":
                methods[tag]={"status":"PRELIMINARY",
                              "duration_s":data.get("duration_s"),
                              "verification":{}}
        rows.append({"case_id":task["case_id"],"population":task["population"],
                     "task_sha256":task["task_sha256"],"methods":methods})
    path=native/"pilot.jsonl"
    text="".join(json.dumps(row,sort_keys=True)+"\n" for row in rows)
    if path.exists() and path.read_text()!=text:
        raise ValueError("native qualification inventory changed")
    if not path.exists():path.write_text(text)
    return {"tasks":len(tasks),"native_preliminary":sum(len(x["methods"]) for x in rows)}


def classify(cohort:Path,qualified_alternatives:Path,*,native_audit:Path)->dict:
    arm=cohort/"nominal_payload"
    all_tasks=json.loads((arm/"tasks.json").read_text())
    registration=json.loads((cohort/"cohort_registration.json").read_text())
    native=arm/"native_full"
    independent=json.loads(native_audit.read_text())
    alternatives=json.loads(qualified_alternatives.read_text())
    rows=[]
    for task in all_tasks:
        case=task["case_id"]
        profile=read_profile(arm/case/"torque_profile.json")
        native_file=native/case/"native_candidates.json"
        history=json.loads(native_file.read_text()) if native_file.exists() else {}
        complete=all(f"F{k:03d}" in history and
                     history[f"F{k:03d}"].get("status") in ("SUCCESS","FAILED")
                     for k in range(FULL_NATIVE_RESTARTS))
        candidates=[]
        for restart in range(FULL_NATIVE_RESTARTS):
            tag=f"F{restart:03d}"
            row=history.get(tag,{})
            if row.get("status")!="SUCCESS":
                continue
            diag=independent.get(f"{case}:{tag}")
            valid,reason=sampled_acceptance(diag,profile,scaled=True)
            if valid and diag.get("source_trajectory_sha256")!=row.get("dense_sha256"):
                valid=False;reason="native independently audited trajectory SHA differs"
            if valid:
                candidates.append({"restart":restart,"duration_s":float(row["duration_s"]),
                                   "trajectory_sha256":row["dense_sha256"]})
        native_best=min((x["duration_s"] for x in candidates),default=None)
        proposals=[x for x in alternatives.get("rows",[]) if
                   x.get("source_case_id")==case and x.get("qualified_scaled95") is True
                   and x.get("source_profile_sha256")==profile["profile_sha256"]]
        alternative_best=min((float(x["duration_s"]) for x in proposals),default=None)
        if not complete:state="NATIVE_BUDGET_INCOMPLETE"
        elif native_best is None and alternative_best is not None:
            state="VERIFIED_NATIVE_FAILURE_RESCUED"
        elif native_best is None:
            state="NATIVE_FAILURE_WITHOUT_INDEPENDENT_WITNESS"
        elif alternative_best is None:
            state="NATIVE_QUALIFIED_NO_INDEPENDENT_ALTERNATIVE"
        elif alternative_best<=(1-DURATION_GAIN)*native_best:
            state="VERIFIED_NATIVE_DURATION_SUBOPTIMALITY"
        else:
            state="BOTH_QUALIFIED_NO_MATERIAL_DURATION_GAIN"
        rows.append({"case_id":case,"base_case_id":task["base_case_id"],
                     "payload_mass_kg":task["payload"]["mass_kg"],
                     "split":task["split"],"task_sha256":task["task_sha256"],
                     "profile_sha256":profile["profile_sha256"],
                     "native_completed_restarts":sum(f"F{k:03d}" in history and
                                      history[f"F{k:03d}"].get("status") in ("SUCCESS","FAILED")
                                      for k in range(FULL_NATIVE_RESTARTS)),
                     "native_qualified_count":len(candidates),
                     "native_best_duration_s":native_best,
                     "alternative_qualified_count":len(proposals),
                     "alternative_best_duration_s":alternative_best,
                     "status":state,
                     "claim":"sampled independent physical+Scaled95; no continuous-time proof"})
    classes={k:sum(x["status"]==k for x in rows) for k in sorted({x["status"] for x in rows})}
    return {"schema_version":1,"registration_sha256":sha256_file(cohort/"cohort_registration.json"),
            "native_audit_sha256":sha256_file(native_audit),
            "alternative_report_sha256":sha256_file(qualified_alternatives),
            "counts":classes,"admitted_tasks":len(rows),
            "attempted_mass_cases":registration["attempted_count"],
            "rows":rows,
            "scope":"real mass on unchanged nominal actuator limits; 32 registered native restarts; source-disjoint by base task; selected scenarios not population-certified"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort",type=Path,required=True)
    p.add_argument("--stage",choices=("native-inventory","classify"),required=True)
    p.add_argument("--native-audit",type=Path)
    p.add_argument("--alternatives",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.stage=="native-inventory":
        result=native_audit_inventory(a.cohort)
    else:
        if a.native_audit is None or a.alternatives is None or a.output is None:
            p.error("classification requires native-audit, alternatives, and output")
        result=classify(a.cohort,a.alternatives,native_audit=a.native_audit)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


if __name__=="__main__":main()
