# SPDX-License-Identifier: Apache-2.0
"""Report matched LP-gated versus ungated global candidate screening.

Numerical necessary-LP rejection can save timing work but cannot
turn a zero LP gap into a physical feasibility certificate. Every W is
eligible for the success count ONLY with the independently SHA-linked
physical+Scaled95 dense/adaptive receipts. Compare wall time and total
RNEA calls, including the common offline bank-generation cost separately.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from .cohort_retime import ROOT
from .geometry_feasibility_core import sha256_file
from .torque_global_bank import (
    DEVELOPMENT_CASE, HOLDOUT_CASES, MAX_PROPOSALS, OUTPUT,
)
from .torque_global_screen import ARMS
from .torque_guided_report import qualified

REPORT=ROOT/"reports/joint_time_oracle/torque_global_lp_screening"
BUDGETS=(16,32,64,128)
CASES=(DEVELOPMENT_CASE,*HOLDOUT_CASES)


def collect(root:Path)->dict:
    rows=[]
    common_bank_build_cost={}
    for case in CASES:
        bankfile=root/"bank"/case/"bank.json"
        bank=json.loads(bankfile.read_text()) if bankfile.exists() else None
        common_bank_build_cost[case]=(bank.get("bank_generation_wall_s")
                                      if bank is not None else None)
        for arm_name in ARMS:
            folder=root/arm_name
            result=folder/case/"screen_result.json"
            entry={"case_id":case,"arm":arm_name,
                   "status":"NOT_EXECUTED","qualified":False}
            if result.exists():
                row=json.loads(result.read_text())
                if bank is None or row["bank_sha256"]!=bank["bank_sha256"]:
                    entry["status"]="INVALID_BANK_INTEGRITY"
                else:
                    ok,reasons=qualified(folder,case,{
                        "candidate_witness_prescreen":row["preliminary_witness"],
                        "torque_profile_sha256":row["torque_profile_sha256"],
                        "w_dense_sha256":row.get("w_dense_sha256")
                    })
                    entry.update({
                        "status":"QUALIFIED_RESCUE" if ok else
                                 "PRESCREEN_UNQUALIFIED" if row["preliminary_witness"]
                                 else "NO_WITNESS_WITHIN_BUDGET",
                        "qualified":ok,
                        "qualification_errors":reasons,
                        "search_wall_s":row["total_cpu_search_wall_s"],
                        "first_screen_proposal":row.get("first_screen_proposal"),
                        "first_screen_cpu_s":row["ledger"][-1]["elapsed_s"]
                             if row["preliminary_witness"] else None,
                        "first_qualified_proposal":row.get("first_screen_proposal") if ok else None,
                        "duration_s":row.get("witness_duration_s") if ok else None,
                        "proposals_evaluated":row["proposals_evaluated"],
                        "lp_rejections":row["lp_exclusions"],
                        "lp_errors":row["lp_errors"],
                        "constructive_timing_calls":row["constructive_timing_calls"],
                        "total_rnea_calls":row["total_rnea_calls"],
                        "lp_rnea_calls":row["lp_rnea_calls"],
                        "geometry_and_timing_rnea_calls":row["geometry_and_timing_rnea_calls"],
                        "per_budget_qualification":{str(b):bool(
                          ok and row["first_screen_proposal"]<=b) for b in BUDGETS},
                        "task_sha256":row["task_sha256"],
                        "torque_profile_sha256":row["torque_profile_sha256"],
                        "bank_sha256":row["bank_sha256"],
                        "result_sha256":sha256_file(result),
                        "scope":"independent dense+adaptive sampled, not continuous-time certificate"
                    })
            rows.append(entry)
    alert=[]
    for case in CASES:
        paired={x["arm"]:x for x in rows if x["case_id"]==case}
        plain=paired["no_lp"]
        gated=paired["lp_gate"]
        if plain["qualified"] and not gated["qualified"] and gated["status"]!="NOT_EXECUTED":
            alert.append({"case_id":case,
                          "severity":"REVIEW_NECESSARY_LP_FALSE_REJECTION_OR_RUNTIME_ERROR",
                          "reason":"ungated found W on same bank, gated did not qualify"})
    complete=all(x["status"]!="NOT_EXECUTED" for x in rows)
    qualified_counts={arm:sum(x["qualified"] for x in rows if x["arm"]==arm)
                      for arm in ARMS}
    holdout_freeze=root/"holdout_freeze.json"
    return {
        "schema_version":1,
        "experiment_status":"COMPLETE" if complete else "INCOMPLETE",
        "registered_tasks":list(CASES),
        "arms":list(ARMS),"budgets":list(BUDGETS),
        "qualified_counts":qualified_counts,
        "shared_bank_generation_cpu_wall_s_by_case":common_bank_build_cost,
        "per_budget_success":{arm:{str(b):sum(x.get("per_budget_qualification",{}).get(str(b),False)
                                                 for x in rows if x["arm"]==arm)
                                    for b in BUDGETS} for arm in ARMS},
        "rows":rows,"safety_alerts":alert,
        "holdout_freeze_sha256":sha256_file(holdout_freeze) if holdout_freeze.exists() else None,
        "claim":"same prospective LP-blind proposal bank, selective necessary torque LP gate versus no LP gate",
        "limits":"3 selected synthetic actuator task examples; W qualification is sampled/adaptive only",
        "latency_contract":"search wall times include independent Pinocchio screening, LP and timing but exclude common pre-generated bank construction and offline audit",
        "decision":"BLOCKED_REVIEW_SAFETY_ALERT" if alert else
                   "UNEXECUTED_OR_INCOMPLETE" if not complete else
                   "NO_GO_LP_GATE" if qualified_counts["lp_gate"] < qualified_counts["no_lp"] else
                   "EVALUATE_COMPUTE_PARETO_NOT_SUCCESS_COUNT_ALONE",
    }


def publish(result:dict,path:Path):
    path.mkdir(parents=True,exist_ok=True)
    (path/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    cols=("case_id","arm","status","first_qualified_proposal",
          "search_wall_s","lp_rejections","constructive_timing_calls",
          "total_rnea_calls","duration_s")
    body=[]
    for r in result["rows"]:
        body.append("<tr>"+"".join("<td>"+html.escape(str(r.get(k,"N/A")))+"</td>"
                                      for k in cols)+"</tr>")
    page="""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Global candidate screening: LP gate vs no LP</title>
<style>body{font:15px/1.55 system-ui;max-width:1180px;margin:2rem auto;padding:0 1rem}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #d8dee7;padding:.5rem;text-align:left}
th{background:#e9eff8}.note{background:#fff1d7;border-left:5px solid #b37412;padding:1rem}</style>
<h1>Global path exploration: necessary torque LP gating</h1>
<p class="note">Both arms evaluate the exact SAME frozen LP-blind global proposal bank
in the same order. Only a numerically dual-checked positive necessary torque LP
can exclude a path from expensive timing. LP zero is inconclusive. Only
independently physical+Scaled95 qualified dense/adaptive witnesses count as
success. All tests use synthetic actuator limits, not Panda manufacturer ratings.
No continuous-time torque or collision guarantee is claimed.</p>"""
    page+="<h2>Status</h2><pre>"+html.escape(json.dumps({
        k:result[k] for k in ("experiment_status","decision",
        "qualified_counts","per_budget_success","safety_alerts")},indent=2))+"</pre>"
    page+="<h2>Every arm and case</h2><table><thead><tr>"+"".join(
        "<th>"+html.escape(x)+"</th>" for x in cols)+"</tr></thead><tbody>"
    page+="\n".join(body)+"</tbody></table>"
    page+="<p>Raw trajectory SHA manifests, CPU timing boundaries and independent"
    page+=" receipts are required before interpreting apparent compute savings.</p></html>"
    (path/"index.html").write_text(page,encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUTPUT)
    p.add_argument("--report",type=Path,default=REPORT)
    a=p.parse_args()
    result=collect(a.root)
    publish(result,a.report)
    print(json.dumps({k:result[k] for k in ("experiment_status","decision",
                                           "qualified_counts","safety_alerts")},indent=2))


if __name__=="__main__":
    main()
