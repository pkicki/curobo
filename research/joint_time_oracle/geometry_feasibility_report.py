# SPDX-License-Identifier: Apache-2.0
"""Fail-closed taxonomy for geometry-dependent torque feasibility experiments.

Three distinct statements:
A. Original fixed geometry: a numerically reliable relaxed torque LP gap.
B. Alternative geometry: independently sampled/adaptive Scaled95 feasible.
C. Native full planner: observed success or a completed finite-budget failure.

A+B demonstrates the motivating *geometric* feasibility distinction even
when native planning C succeeds. A+B+C failure is a stricter algorithmic
counterexample. We never call a runtime exception a native planning failure.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from .geometry_feasibility_core import scenarios
from .geometry_feasibility_discover import OUT, SOURCE
from .torque_active import read_profile

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/"reports/joint_time_oracle/geometry_feasibility"


def classify(witness:dict|None, lp:dict|None, native:dict,
             native_receipts:dict, *,budget:int=32)->dict:
    if budget<1:
        raise ValueError("budget must be >=1")
    witness_good=bool(isinstance(witness,dict) and witness.get("physical") is True
                      and witness.get("scaled95") is True and witness.get("collision") is True
                      and witness.get("endpoint") is True)
    lp_good=bool(isinstance(lp,dict) and
                 lp.get("status")=="NUMERICALLY_STRONG_ORIGINAL_PATH_INFEASIBILITY"
                 and lp.get("numerical_dual_certificate",{}).get(
                     "numerically_strong_infeasibility") is True)
    parents={f"F{k:03d}":native.get(f"F{k:03d}") for k in range(budget)}
    missing=sum(row is None for row in parents.values())
    failed=sum(isinstance(row,dict) and row.get("status")=="FAILED"
               for row in parents.values())
    errors=sum(isinstance(row,dict) and row.get("status")=="ERROR"
               for row in parents.values())
    successes=[(name,row) for name,row in native.items()
               if row.get("status")=="SUCCESS" and isinstance(row.get("restart"),int)
               and 0<=row["restart"]<budget]
    native_ok=[name for name,row in successes if
               native_receipts.get(name,{}).get("physical") is True and
               native_receipts[name].get("torque_profile_sha256")==
               row.get("torque_profile_sha256")]
    if not witness_good:
        category="NO_QUALIFIED_ALTERNATIVE"
    elif not lp_good:
        category="NO_STRONG_ORIGINAL_GEOMETRY_INFEASIBILITY_PROOF"
    elif native_ok:
        category="GEOMETRY_OBSTRUCTION_BUT_NATIVE_REPLANNED_SUCCESSFULLY"
    elif missing:
        category="GEOMETRY_OBSTRUCTION_NATIVE_BUDGET_INCOMPLETE"
    elif errors:
        category="GEOMETRY_OBSTRUCTION_NATIVE_EXECUTION_BLOCKED"
    elif successes:
        category="GEOMETRY_OBSTRUCTION_NATIVE_RETURNED_UNQUALIFIED_TRAJECTORY"
    elif failed==budget:
        category="GEOMETRY_OBSTRUCTION_NATIVE_BUDGET_FAILURE"
    else:
        category="GEOMETRY_OBSTRUCTION_NATIVE_STATUS_INDETERMINATE"
    return {"category":category,
            "original_fixed_geometry_numerically_infeasible":lp_good,
            "alternative_independent_scaled95_feasible":witness_good,
            "native_verified_physical":len(native_ok),
            "native_verified_candidate_ids":native_ok,
            "native_success_candidate_count":len(successes),
            "native_failed_parent_restarts":failed,
            "native_execution_errors":errors,
            "native_missing_parent_restarts":missing,
            "native_budget_restarts":budget,
            "scope":"sampled witness, numerical LP, finite planning budget; not global or continuous-time certificate"}


def collect(root:Path,source:Path,*,budget:int=32)->dict:
    registration=json.loads((root/"discovery_registration.json").read_text())
    if not registration.get("scenarios") or registration["scenarios"]!=scenarios():
        raise ValueError("registered scenario grid mismatch")
    summary=json.loads((root/"discovery_summary.json").read_text())
    outcomes=[]
    for scenario in registration["scenarios"]:
        name=scenario["scenario_id"]
        arm=root/name
        ledger=arm/"attempt_inventory.json"
        if not ledger.exists():
            outcomes.append({"scenario":name,"status":"NOT_SCANNED"})
            continue
        attempts=json.loads(ledger.read_text())
        wfile=arm/"w_adaptive_receipts.json"
        witnesses=json.loads(wfile.read_text()) if wfile.exists() else {}
        nfile=arm/"native_full"/"adaptive_receipts.json"
        nreceipts=json.loads(nfile.read_text()) if nfile.exists() else {}
        for item in attempts:
            case=item["case_id"]
            row={"scenario":name,"case_id":case,
                 "discovery_status":item["status"],
                 "endpoint_admitted":item["endpoint_admitted"],
                 "original_prescreen_lp":item.get("original_relaxed_lp",{}).get(
                     "max_required_torque_bound_expansion"),
                 "candidate_proposals":len(item.get("candidate_attempts",[])),
                 "constructive_candidate_duration_s":item.get("chosen_duration_s")}
            if not item["endpoint_admitted"]:
                row["category"]="ENDPOINT_REJECTED"
            elif item["status"]!="CONSTRUCTIVE_WITNESS_PRESCREEN":
                row["category"]="NO_CONSTRUCTIVE_ALTERNATIVE_DISCOVERED"
            else:
                folder=arm/case
                refined_file=folder/"refined_original_path_lp.json"
                lp=json.loads(refined_file.read_text()) if refined_file.exists() else {}
                candidate_file=arm/"native_full"/case/"native_candidates.json"
                native=json.loads(candidate_file.read_text()) if candidate_file.exists() else {}
                matched={k.removeprefix(case+":"):v for k,v in nreceipts.items()
                         if k.startswith(case+":")}
                res=classify(witnesses.get(case+":W"),lp,native,matched,budget=budget)
                row.update(res)
                row["refined_lp_file"]=str(refined_file) if refined_file.exists() else None
                row["witness_receipt_file"]=str(wfile) if wfile.exists() else None
                row["native_candidate_file"]=str(candidate_file) if candidate_file.exists() else None
            outcomes.append(row)
    counts={}
    for row in outcomes:
        key=row.get("category",row.get("status","UNKNOWN"))
        counts[key]=counts.get(key,0)+1
    return {"schema_version":1,"registration":registration,
            "discovery_summary":summary,"native_budget":budget,
            "counts":counts,"cases":outcomes,
            "note":"Source-case/scenario cells are correlated; adversarial discovery ≠ confirmatory prevalence.",
            "independence":"new source-disjoint tasks and fixed scenario selection required for generalization"}


def publish(data:dict,output:Path):
    output.mkdir(parents=True,exist_ok=True)
    (output/"summary.json").write_text(json.dumps(data,indent=2)+"\n")
    table=[]
    for case in data["cases"]:
        table.append("<tr><td>"+html.escape(str(case.get("case_id","—")))+"</td>"
                     "<td>"+html.escape(case["scenario"])+"</td>"
                     "<td>"+str(case.get("original_prescreen_lp","N/A"))+"</td>"
                     "<td>"+str(case.get("constructive_candidate_duration_s","N/A"))+"</td>"
                     "<td>"+str(case.get("native_failed_parent_restarts","N/A"))+"</td>"
                     "<td>"+html.escape(case.get("category",case.get("status","unknown")))+"</td></tr>")
    page="""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Geometry-dependent torque infeasibility</title>
<style>body{font:15px/1.5 system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}
table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #ddd;padding:.45rem;text-align:left}
th{background:#edf2f7}.note{background:#fff0d8;border-left:4px solid #b36b00;padding:1rem}</style>
<h1>Geometry-dependent torque infeasibility</h1>
<p class="note">A strong geometry witness requires a positive numerically
dual-validated original fixed-path torque LP AND a different independently
sampled/adaptive Scaled95-feasible trajectory. Native full-planner outcomes
are separate: a finite-budget native failure is NOT proof of global
impossibility. The model uses synthetic joint-selective actuator limits,
not the manufacturer's Panda specifications. Dense/adaptive sampled
validation is not continuous-time certification.</p>"""
    counts=data["counts"]
    geom=sum(v for k,v in counts.items() if k.startswith("GEOMETRY_OBSTRUCTION"))
    failed=counts.get("GEOMETRY_OBSTRUCTION_NATIVE_BUDGET_FAILURE",0)
    page+=f"<p><strong>{geom}</strong> qualified geometric distinctions; <strong>{failed}</strong> complete native-budget failures with such a distinction.</p>"
    page+="<h2>Every registered case/scenario outcome</h2><table><thead><tr>"
    page+="<th>Task</th><th>Actuator scenario</th><th>Original LP prescreen</th>"
    page+="<th>Witness duration (s)</th><th>Native failed restarts</th><th>Category</th>"
    page+="</tr></thead><tbody>"+"\n".join(table)+"</tbody></table>"
    page+="<h2>Unfiltered category counts</h2><pre>"+html.escape(json.dumps(counts,indent=2))+"</pre></html>"
    (output/"index.html").write_text(page,encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUT)
    p.add_argument("--source",type=Path,default=SOURCE)
    p.add_argument("--report",type=Path,default=REPORT)
    p.add_argument("--restarts",type=int,default=32)
    a=p.parse_args()
    data=collect(a.root,a.source,budget=a.restarts)
    publish(data,a.report)
    print(json.dumps({"cases":len(data["cases"]),"categories":data["counts"]},indent=2))


if __name__=="__main__":
    main()
