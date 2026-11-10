# SPDX-License-Identifier: Apache-2.0
"""Fail-closed HTML+JSON for nominal-effort payload and TimeOpt-feedback study.

Do not claim native failures rescued from a K107 numerical candidate alone.
Only exact task+payload+geometry SHA, independently payload-aware
Pinocchio RNEA and attached-collision sphere Scaled95 sampled/adaptive
receipts promote cases. Three feedback policies share paired source IDs.
Native latency measured separately from a reference K107 subprocess.
"""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
from statistics import median

from .payload_timeopt_contract import ARMS,OUT,immutable_json,source_sha


REPORT=Path(__file__).resolve().parents[2]/"reports/joint_time_oracle/nominal_payload_timeopt_guided"


def read_optional(path:Path):
    return json.loads(path.read_text()) if path.is_file() else None


def collect(root:Path)->dict:
    qualified=root/"qualified"/"qualified_tasks.json"
    if not qualified.is_file():
        return {"schema":1,"status":"NOT_EXECUTED","task_rows":[],
                "message":"no frozen independently qualified payload task inventory"}
    tasks=json.loads(qualified.read_text())["tasks"]
    rows=[];hardcase_counts=Counter();per_arm=Counter()
    for task in tasks:
        case=task["case_id"];split=task["split"]
        stage=read_optional(root/"native"/case/"summary_r32.json")
        result={"case_id":case,"source_task_id":task["source_task_id"],
                "split":split,"population":task["population"],
                "payload":task["payload"],"endpoint_admission":task["admission"],
                "native_status":stage.get("status") if stage else "NOT_EXECUTED",
                "native_restart_budget":stage.get("registered_restart_budget") if stage else None,
                "arms":{}}
        for arm in ARMS:
            folder=root/"search"/arm/case
            outcome=read_optional(folder/"search_result.json")
            # Separate independent *source* receipts; same output
            # expected in independent/<case>/<arm>/final_receipt.json
            receipt_file=root/"independent"/case/arm/"final_receipt.json"
            receipt=read_optional(receipt_file)
            output={"status":"NOT_EXECUTED","qualified":False}
            if outcome is not None:
                output.update({"status":"PRELIMINARY_OR_UNQUALIFIED",
                               "reference_solver_wall_s":outcome.get("total_wall_s"),
                               "attempted_proposals":outcome.get("proposals_attempted"),
                               "oracle_feedback_proposals":outcome.get("timeopt_feedback_proposals"),
                               "bank_sha256":outcome.get("shared_bank_sha256"),
                               "preliminary_best":outcome.get("preliminary_best") is not None})
                if receipt is not None:
                    matched=(receipt.get("case_id")==case and
                             receipt.get("task_sha256")==task["task_sha256"] and
                             receipt.get("payload_parity_sha256")==
                                  task["payload_collision_receipt_sha256"] and
                             receipt.get("physical") is True and
                             receipt.get("scaled95") is True and
                             receipt.get("payload_collision_physical") is True)
                    if matched:
                        output.update({"status":"INDEPENDENTLY_QUALIFIED_SAMPLED",
                                       "qualified":True,
                                       "duration_s":receipt.get("duration_s"),
                                       "qualification_receipt_sha256":source_sha(receipt_file)})
                        per_arm[(split,arm)]+=1
                    else:
                        output["status"]="RECEIPT_FAILED_OR_MISMATCHED"
            result["arms"][arm]=output
        witness=any(x["qualified"] for x in result["arms"].values())
        native_failure=(result["native_status"]=="BUDGET_EXHAUSTED" and
                        result["native_restart_budget"]==32)
        if task["admission"]!="ENDPOINT_ADMITTED":
            label="ENDPOINT_REJECTED"
        elif native_failure and witness:
            label="NATIVE_32_RESTART_FAILURE_INDEPENDENTLY_RESCUED"
        elif native_failure:
            label="NATIVE_FAILURE_WITHOUT_QUALIFIED_WITNESS"
        elif result["native_status"]=="NATIVE_SUCCESS_UNQUALIFIED":
            label="NATIVE_SUCCESS_NOT_YET_INDEPENDENTLY_QUALIFIED"
        elif result["native_status"]=="NOT_EXECUTED":
            label="NATIVE_NOT_EXECUTED"
        else:
            label="INCOMPLETE_OR_OTHER"
        result["classification"]=label
        hardcase_counts[(split,label)]+=1
        rows.append(result)
    paired={}
    for arm in ARMS:
        matched=[x for x in rows if x["split"]=="holdout"
                 and x["arms"][arm]["status"]!="NOT_EXECUTED"]
        paired[arm]={"scored_holdout_cases":len(matched),
                     "qualified_holdout_cases":sum(x["arms"][arm]["qualified"] for x in matched),
                     "median_reference_process_wall_s":(
                         median(x["arms"][arm]["reference_solver_wall_s"] for x in matched
                                if x["arms"][arm]["reference_solver_wall_s"] is not None)
                         if any(x["arms"][arm]["reference_solver_wall_s"] is not None for x in matched)
                         else None),
                     "warning":"path candidates correlated across payload variants; source-level grouping mandatory"}
    return {"schema":1,"status":"PARTIAL_OR_COMPLETE_EVIDENCE",
            "qualified_task_registry_sha256":source_sha(qualified),
            "source_task_groups":len({t["source_task_id"] for t in tasks}),
            "task_scenario_count":len(tasks),
            "native_hardcase_counts_by_split":{"%s:%s"%k:v for k,v in hardcase_counts.items()},
            "feedback_arm_qualifications_by_split":{"%s:%s"%k:v for k,v in per_arm.items()},
            "holdout_arm_summary":paired,"task_rows":rows,
            "claim":"independently qualified witnesses only; nominal-effort payload with parity-qualified collision",
            "no_generalization_without_disjoint_sources":True,
            "no_production_latency_from_K107_subprocess":True,
            "no_continuous_time_collision_or_torque_certificate":True}


def publish(result:dict,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    rows=[]
    for item in result.get("task_rows",[])[:100]:
        rows.append("<tr>"+"".join("<td>"+html.escape(str(field))+"</td>"
            for field in (item["case_id"],item["split"],item["native_status"],
                          item["classification"],
                          *(item["arms"][a]["status"] for a in ARMS)))+"</tr>")
    page="""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Nominal-effort payload TimeOpt-guided path planning</title>
<style>body{font:15px/1.5 system-ui;max-width:1300px;margin:2rem auto;padding:1rem}
.warning{background:#fff1da;padding:1rem;border-left:5px solid #ba801a}
table{border-collapse:collapse;width:100%}td,th{padding:.5rem;border-bottom:1px solid #dde4ec;text-align:left}</style>
<h1>Nominal-effort payload hard cases and TimeOpt-guided global planning</h1>
<p class="warning">Do not confuse native budget exhaustion with global infeasibility,
a TOPP-RA+Newton preliminary candidate with independent feasibility,
or attached-mass dynamics with a payload collision-validated robot.
Only model-parity-qualified samples count; all qualifications remain sampled,
not continuous-time or hardware safety certificates. TOPP subprocess wall time
is not a measured captured-production K107 latency.</p>"""
    page+="<h2>Research summary</h2><pre>"+html.escape(json.dumps({
       k:v for k,v in result.items() if k!="task_rows"},indent=2))+"</pre>"
    page+="<h2>First 100 case records (all cases in JSON)</h2><table><tr>"
    page+="".join("<th>"+x+"</th>" for x in ("Task","Split","Native","Classification",*ARMS))+"</tr>"
    page+="\n".join(rows)+"</table></html>"
    (out/"index.html").write_text(page)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUT)
    p.add_argument("--report",type=Path,default=REPORT)
    a=p.parse_args()
    result=collect(a.root);publish(result,a.report)
    print(json.dumps({k:v for k,v in result.items() if k not in ("task_rows",)},indent=2))


if __name__=="__main__":
    main()
