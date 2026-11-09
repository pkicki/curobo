# SPDX-License-Identifier: Apache-2.0
"""Conservative matched nominal-payload hard-case registry.

A native 32-restart failure is evidence of a PLANNER BUDGET FAILURE,
not global infeasibility. To call it a proven hard case, an alternative
path must independently qualify under the SAME task/model/nominal
actuator limits on both S8193 and event-aware S32768. Native successes
must also independently qualify before duration comparisons.

No labels based on native internal success bits alone, and no data
from heldout tasks may tune the algorithm before code freeze. Preserve
all source cells and error counts, including trivial loaded-endpoint
gravity impossibility and inconclusive cases.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .timeopt_payload_model import file_sha256
from .timeopt_payload_tasks import OUT

MIN_SUBOPTIMALITY_GAP=.05


def _qualified_alternatives(task:dict,roots:list[Path])->list[dict]:
    found=[]
    case=task["case_id"]
    for root in roots:
        for arm in ("random_global","duration_rank","timeopt_feedback"):
            folder=root/arm/case
            receipt=folder/"independent_k107_qualification.json"
            if not receipt.exists():continue
            entry=json.loads(receipt.read_text())
            for candidate in entry.get("candidates",[]):
                if not candidate.get("qualified"):
                    continue
                request_file=folder/"oracle"/(
                    next((x.name for x in (folder/"oracle").glob(
                        "*"+candidate["request_id"]+".request.json")), ""))
                if not request_file.is_file():
                    continue
                request=json.loads(request_file.read_text())
                if (request.get("task_sha256")!=task["task_sha256"] or
                    request.get("payload",{}).get("payload_sha256")
                        !=task["payload"]["payload_sha256"]):
                    continue
                if (candidate["S8193"].get("independent_qualified_sampled") is not True
                    or candidate["S32768"].get("independent_qualified_sampled") is not True):
                    continue
                found.append({"method":arm,
                              "duration_s":float(candidate["S32768"]["duration_s"]),
                              "receipt_sha256":file_sha256(receipt),
                              "path_sha256":candidate["path_sha256"],
                              "request_sha256":file_sha256(request_file),
                              "source_root":str(root)})
    return sorted(found,key=lambda x:x["duration_s"])


def assemble(source:Path,output:Path,*,alternative_roots:list[Path],
             native_receipt:Path,require_complete_native:bool=True)->dict:
    tasks=json.loads((source/"tasks.json").read_text())
    successes=json.loads(native_receipt.read_text()).get("candidates",{})
    cases=[]
    counts={}
    for task in tasks:
        case=task["case_id"]
        folder=source/"native_full"/case
        nativefile=folder/"native_candidates.json"
        native=json.loads(nativefile.read_text()) if nativefile.exists() else {}
        indices=[native.get(f"F{i:03d}") for i in range(32)]
        completed=bool(all(isinstance(x,dict) and x.get("status") in
            ("FAIL_NATIVE_BUDGET","SUCCESS_NATIVE_UNQUALIFIED")
            for x in indices))
        any_native_error=any(isinstance(x,dict) and x.get("status")=="ERROR"
                             for x in indices)
        certified=[x for key,x in successes.items() if key.startswith(case+":")
                   and x.get("qualified") is True]
        best_native=min((float(x["duration_s"]) for x in certified),default=None)
        alternate=_qualified_alternatives(task,alternative_roots)
        best_alternative=alternate[0]["duration_s"] if alternate else None
        if task["endpoint"]["category"]=="STATIC_ENDPOINT_IMPOSSIBLE":
            label="TRIVIAL_STATIC_ENDPOINT_IMPOSSIBLE"
        elif not task["allowed_for_native_challenge"]:
            label="ENDPOINT_RESERVE_REJECT"
        elif any_native_error:
            label="NATIVE_RUNTIME_ERRORS_NOT_FAILURES"
        elif not completed:
            label="NATIVE_32_RESTART_BUDGET_INCOMPLETE"
        elif best_native is None:
            if any(x["status"]=="SUCCESS_NATIVE_UNQUALIFIED" for x in indices):
                label="NATIVE_SUCCESS_UNQUALIFIED"
            elif best_alternative is not None:
                label="NATIVE_FAILED_ALTERNATIVE_QUALIFIED"
            else:
                label="NATIVE_FAILED_ALTERNATIVE_UNPROVEN"
        elif best_alternative is None:
            label="NATIVE_FEASIBLE_NO_QUALIFIED_ALTERNATIVE"
        elif best_alternative <(1-MIN_SUBOPTIMALITY_GAP)*best_native:
            label="NATIVE_FEASIBLE_BUT_DURATION_SUBOPTIMAL"
        else:
            label="NATIVE_FEASIBLE_MATCHED_QUALITY"
        counts[label]=counts.get(label,0)+1
        cases.append({
            "case_id":case,"class":label,
            "source_task_sha256":task["task_sha256"],
            "payload_sha256":task["payload"]["payload_sha256"],
            "scene":task["scene"],"source_group":task["separation_group"],
            "payload_mass_kg":task["payload"]["mass_kg"],
            "native_32_restarts_completed":completed,
            "native_error_in_first_32":any_native_error,
            "native_qualified_count":len(certified),
            "native_best_qualified_duration_s":best_native,
            "alternative_qualified_count":len(alternate),
            "alternative_best_qualified_duration_s":best_alternative,
            "alternative_evidence":alternate[:5],
            "native_inventory_sha256":file_sha256(nativefile)
                                      if nativefile.exists() else None,
            "description":"32-restart native budget, not global feasibility certificate"})
    status="INCOMPLETE" if any(r["class"]=="NATIVE_32_RESTART_BUDGET_INCOMPLETE"
                              for r in cases if r["source_group"]=="holdout") else "EVALUATED"
    record={"schema_version":1,"study_status":status,"source_sha256":
            file_sha256(source/"tasks.json"),"native_receipt_sha256":
            file_sha256(native_receipt),"alternative_root_shas":
            {str(x):file_sha256(x/"source_manifest.json")
             if (x/"source_manifest.json").exists() else None for x in alternative_roots},
            "counts":counts,"cases":cases,
            "hard_case_count":sum(r["class"] in (
                "NATIVE_FAILED_ALTERNATIVE_QUALIFIED",
                "NATIVE_FEASIBLE_BUT_DURATION_SUBOPTIMAL") for r in cases),
            "scope":"nominal effort, registered physical spherical payload; 32 finite native restarts and independently audited alternatives"}
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() and json.loads(output.read_text())!=record:
        raise ValueError("hard-case registry previously frozen with different evidence")
    if not output.exists():
        output.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=OUT)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--native-receipt",type=Path,required=True)
    p.add_argument("--alternative-root",type=Path,action="append",default=[])
    a=p.parse_args()
    info=assemble(a.source,a.output,alternative_roots=a.alternative_root,
                  native_receipt=a.native_receipt)
    print(json.dumps({"counts":info["counts"],
                      "hard_case_count":info["hard_case_count"]},indent=2))


if __name__=="__main__":
    main()
