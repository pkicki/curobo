# SPDX-License-Identifier: Apache-2.0
"""Hard-case and TimeOpt-feedback study summary from frozen physical evidence.

Only independent Pinocchio S8193 plus event-aware S32768 qualified
trajectories count as planning success. The original unchanged effort
limits and payload model parity must be registered. The method's
online K107 screen is NOT proof of feasibility; native internal success
is NOT an audited baseline. Report every source cell and do not infer
population statistics from selected pilot hard cases.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from statistics import median

from .cohort_retime import ROOT
from .timeopt_feedback_search import ARMS, BUDGETS
from .timeopt_payload_model import file_sha256
from .timeopt_payload_tasks import OUT

RESULTS=ROOT/"research/joint_time_oracle/results/timeopt_feedback_20261109"
REPORT=ROOT/"reports/joint_time_oracle/timeopt_guided_nominal_payload"


def collect(source:Path,search_root:Path,*,registry:Path|None=None,
            parity:Path|None=None)->dict:
    source_file=source/"tasks.json"
    if not source_file.exists():
        return {"schema_version":1,"status":"NOT_EXECUTED",
                "rows":[],"claim":"research implementation only"}
    tasks=json.loads(source_file.read_text())
    classes={x["case_id"]:x for x in
             json.loads(registry.read_text()).get("cases",[])} if registry and registry.exists() else {}
    parity_data=json.loads(parity.read_text()) if parity and parity.exists() else None
    rows=[]
    for task in tasks:
        case=task["case_id"]
        registered=classes.get(case,{})
        for arm in ARMS:
            folder=search_root/arm/case
            result=folder/"search_result.json"
            row={"case_id":case,"arm":arm,"source_group":task["separation_group"],
                 "mass_kg":task["payload"]["mass_kg"],
                 "scene":task["scene"],
                 "source_hard_class":registered.get("class","NOT_YET_CLASSIFIED"),
                 "status":"NOT_EXECUTED","qualified":False}
            if result.exists():
                study=json.loads(result.read_text())
                if (study["source_task_sha256"]!=task["task_sha256"] or
                    study["payload_sha256"]!=task["payload"]["payload_sha256"]):
                    row["status"]="SOURCE_PROVENANCE_MISMATCH"
                elif parity_data is None or not parity_data.get(
                    "model_and_collision_parity_pass"):
                    row["status"]="MODEL_PARITY_MISSING"
                else:
                    receipt_file=folder/"independent_k107_qualification.json"
                    receipts=json.loads(receipt_file.read_text()).get(
                        "candidates",[]) if receipt_file.exists() else []
                    verified={x["path_sha256"]:x for x in receipts
                              if x.get("qualified") is True and
                                 x.get("S8193",{}).get(
                                     "independent_qualified_sampled") is True and
                                 x.get("S32768",{}).get(
                                     "independent_qualified_sampled") is True}
                    possible=[entry for entry in study.get("ledger",[])
                              if entry["path_sha256"] in verified]
                    qualified=bool(possible)
                    row.update({
                        "status":"INDEPENDENT_QUALIFIED" if qualified else
                            "ONLINE_ONLY_UNQUALIFIED" if study.get(
                                "first_preliminary_feasible_proposal") is not None
                                else "NO_K107_SCREENED_FEASIBLE_PATH",
                        "qualified":qualified,
                        "first_qualified_proposal":min((x["proposal"] for x in possible),
                                                        default=None),
                        "best_independent_duration_s":min((
                            float(verified[x["path_sha256"]]["S32768"]["duration_s"])
                            for x in possible),default=None),
                        "proposals_attempted":study["proposals_attempted"],
                        "oracle_requests":study["oracle_requests"],
                        "total_search_wall_s":study["total_wall_s"],
                        "qualified_by_budget":{str(n):any(
                            x["proposal"]<=n for x in possible) for n in BUDGETS},
                        "qualified_in_wall_budget":qualified,
                        "online_status_counts":study.get("status_counts",{}),
                        "research_code_sha256":file_sha256(result),
                        "independent_receipt_sha256":file_sha256(receipt_file)
                            if receipt_file.exists() else None,
                        "source_reference_native_duration_s":
                            registered.get("native_best_qualified_duration_s"),
                        "claim":"S8193 and event-aware S32768 sampled; no continuous-time certification"
                    })
            rows.append(row)
    count_by_arm={}
    for arm in ARMS:
        data=[x for x in rows if x["arm"]==arm]
        values=[x["total_search_wall_s"] for x in data if
                "total_search_wall_s" in x]
        count_by_arm[arm]={
            "registered_task_mass_cells":len(data),
            "attempted":len(values),
            "independently_qualified":sum(x["qualified"] for x in data),
            "median_full_search_wall_s":median(values) if values else None,
            "success_at_proposals":{str(n):sum(
                x.get("qualified_by_budget",{}).get(str(n),False)
                for x in data) for n in BUDGETS},
        }
    groups={}
    for task in tasks:
        group=task["separation_group"]
        groups.setdefault(group,0)
        groups[group]+=1
    status="COMPLETE" if rows and all(x["status"]!="NOT_EXECUTED"
                                      for x in rows) else "INCOMPLETE"
    missing=[x["case_id"]+"/"+x["arm"] for x in rows if x["status"]=="NOT_EXECUTED"]
    return {"schema_version":1,"status":status,
            "source_tasks_sha256":file_sha256(source_file),
            "hard_registry_sha256":file_sha256(registry) if registry and registry.exists() else None,
            "model_parity_sha256":file_sha256(parity) if parity and parity.exists() else None,
            "base_endpoint_groups":groups,
            "registered_task_mass_cells":len(tasks),
            "arms":count_by_arm,
            "rows":rows,"missing_evaluations":missing,
            "decision":"BLOCKED_OR_INCOMPLETE" if status!="COMPLETE" else
                       "EVALUATE_PAIRED_FEEDBACK_VS_UNGUIDED",
            "safety":"only independently model-parity-qualified, sampled S8193/S32768 trajectories may count",
            "limitations":"finite native restart budget, synthetic rigid payload at original actuator limits; no continuous-time safety or global optimality proof"}


def publish(data:dict,output:Path):
    output.mkdir(parents=True,exist_ok=True)
    (output/"summary.json").write_text(
        json.dumps(data,indent=2,allow_nan=False)+"\n")
    cols=("case_id","mass_kg","arm","source_hard_class","status",
          "first_qualified_proposal","best_independent_duration_s",
          "total_search_wall_s")
    header="".join("<th>"+html.escape(key)+"</th>" for key in cols)
    trs=[]
    for row in data.get("rows",[]):
        trs.append("<tr>"+"".join(
            "<td>"+html.escape(str(row.get(key,"N/A")))+"</td>"
            for key in cols)+"</tr>")
    page="""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nominal actuator payload hard cases and TimeOpt-guided path search</title>
<style>body{font:15px/1.5 system-ui;max-width:1250px;margin:2rem auto;padding:1rem}
th,td{border-bottom:1px solid #dce1e8;padding:.4rem;text-align:left}
table{width:100%;border-collapse:collapse}th{background:#eef5ff}
.note{padding:1rem;background:#fff0d7;border-left:5px solid #b77b20}</style>
<h1>TimeOpt feedback for nominal-actuator payload planning</h1>
<p class="note">A native finite-restart failure is not proof of global infeasibility.
Retimer online success is not independent qualification. Only same-payload
Pinocchio physical and Scaled95 S8193/event S32768 checks may be counted.
Heavy payloads are represented by a specified sphere mass/inertia/collision
model at original joint-effort limits. No hardware/continuous-time safety claim.
The full source denominator includes static-endpoint impossibilities and errors.</p>"""
    page+="<h2>Overall status</h2><pre>"+html.escape(json.dumps({
        "status":data.get("status"),"decision":data.get("decision"),
        "task_mass_cells":data.get("registered_task_mass_cells"),
        "arms":data.get("arms"),
    },indent=2))+"</pre>"
    page+="<h2>Case-by-arm evidence</h2><table><thead><tr>"+header+"</tr></thead><tbody>"
    page+="\n".join(trs)+"</tbody></table>"
    page+="<p>Independent cross-repository K107 comparison requires exact payload-model,"
    page+=" endpoint, signed effort and constraint-family parity. See the frozen protocol.</p></html>"
    (output/"index.html").write_text(page,encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=OUT)
    p.add_argument("--results",type=Path,default=RESULTS)
    p.add_argument("--report",type=Path,default=REPORT)
    p.add_argument("--hard-case-registry",type=Path)
    p.add_argument("--parity-receipt",type=Path)
    a=p.parse_args()
    result=collect(a.source,a.results,registry=a.hard_case_registry,
                   parity=a.parity_receipt)
    publish(result,a.report)
    print(json.dumps({"status":result["status"],
                      "decision":result.get("decision"),
                      "arms":result.get("arms")},indent=2))


if __name__=="__main__":
    main()
