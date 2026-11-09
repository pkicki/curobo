# SPDX-License-Identifier: Apache-2.0
"""Qualified native-vs-K107 and TimeOpt-guided-vs-global planning report.

All counts use registered original-effort payload task/mass denominators.
A K107 online pass is *not* a rescue, and native failed is not proof
of feasibility. Native successes and K107 alternatives are compared
only after both have identical-model independent physical+Scaled95
dense/adaptive receipts. Report implementation and model blockers.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from .timeopt_feedback_geometry import ARMS
from .geometry_feasibility_core import sha256_file

def read_if(path:Path):
    return json.loads(path.read_text()) if path.exists() else None


def assemble(cohort:Path,search:Path,audits:Path)->dict:
    registry=read_if(cohort/"cohort_registration.json")
    if registry is None:
        return {"schema_version":1,"decision":"NOT_EXECUTED",
                "rows":[],"study":"nominal actuator limits and physical payloads"}
    tasks=json.loads((cohort/"nominal_payload/tasks.json").read_text())
    evidence={}
    for arm in ARMS:
        inventory=read_if(audits/arm/"candidate_inventory.json") or []
        receipts=read_if(audits/arm/"k107_independent_receipts.json") or {}
        qualified={}
        for candidate in inventory:
            case=candidate.get("source_case_id")
            key=candidate.get("audit_case_id")
            receipt=receipts.get(str(key)+":K107")
            if receipt and receipt.get("physical") and receipt.get("scaled95"):
                record=read_if(audits/arm/str(key)/"k107_candidate.json")
                if record is None or record["source_case_id"]!=case:
                    raise ValueError("qualified K107 candidate lacks exact source metadata")
                expected=record["source_response_sha256"]
                row=read_if(search/arm/str(case)/f"c{candidate['proposal']:04d}"/"k107_response.json")
                if row is None or sha256_file(
                    search/arm/str(case)/f"c{candidate['proposal']:04d}"/"k107_response.json")!=expected:
                    raise ValueError("K107 candidate response changed since independent audit")
                duration=row.get("duration_s")
                if duration is None or not np.isfinite(duration):
                    raise ValueError("independently qualified candidate duration missing")
                qualified.setdefault(case,[]).append({
                    "proposal":candidate["proposal"],"duration_s":duration,
                    "audit_case_id":key,"response_sha256":expected,
                    "dense_sha256":receipt["dense_sha256"],
                    "adaptive_sha256":receipt["adaptive_sha256"]})
        evidence[arm]=qualified
    rows=[]
    for task in tasks:
        case=task["case_id"]
        for arm in ARMS:
            record=read_if(search/arm/case/"search_result.json")
            candidates=sorted(evidence[arm].get(case,[]),key=lambda x:x["proposal"])
            first=candidates[0] if candidates else None
            if record is None:
                status="NOT_EXECUTED"
            elif not (audits/arm/"k107_independent_receipts.json").exists():
                status="INDEPENDENT_AUDIT_NOT_EXECUTED"
            elif first is None:
                status="NO_QUALIFIED_K107_CANDIDATE"
            else:
                status="INDEPENDENTLY_QUALIFIED_K107"
            if record and record["task_sha256"]!=task["task_sha256"]:
                raise ValueError("search result task fingerprint mismatch")
            rows.append({"case_id":case,"base_case_id":task["base_case_id"],
                         "split":task["split"],"mass_kg":task["payload"]["mass_kg"],
                         "arm":arm,"status":status,
                         "qualified":bool(first),
                         "first_qualified_proposal":first["proposal"] if first else None,
                         "best_qualified_duration_s":min(x["duration_s"] for x in candidates)
                                if candidates else None,
                         "qualified_candidate_count":len(candidates),
                         "qualified_evidence":candidates,
                         "timeopt_online_candidates":record["online_feasible_candidates"] if record else None,
                         "timeopt_calls":record["oracle_calls"] if record else None,
                         "complete_search_wall_s":record["full_search_wall_s"] if record else None,
                         "source_task_sha256":task["task_sha256"]})
    alternative=[]
    for entry in rows:
        if entry["qualified"]:
            source=entry["case_id"]
            original=next(x for x in tasks if x["case_id"]==source)
            from .torque_active import read_profile
            source_profile=read_profile(cohort/"nominal_payload"/source/"torque_profile.json")
            alternative.append({"source_case_id":source,"arm":entry["arm"],
                                "qualified_scaled95":True,
                                "first_qualified_proposal":entry["first_qualified_proposal"],
                                "duration_s":entry["best_qualified_duration_s"],
                                "source_profile_sha256":source_profile["profile_sha256"],
                                "source_task_sha256":original["task_sha256"]})
    counts={arm:sum(r["qualified"] for r in rows if r["arm"]==arm) for arm in ARMS}
    bysplit={split:{arm:sum(r["qualified"] for r in rows
               if r["split"]==split and r["arm"]==arm) for arm in ARMS}
             for split in ("development","confirmation","holdout")}
    holdout=search/"timeopt_feedback_holdout_freeze.json"
    blocked=any(r["status"] in ("NOT_EXECUTED","INDEPENDENT_AUDIT_NOT_EXECUTED") for r in rows)
    return {"schema_version":1,"decision":"INCOMPLETE_UNQUALIFIED" if blocked else
            "MATCHED_RESULTS_AVAILABLE_REVIEW_COMPUTE_AND_QUALITY",
            "cohort_registration_sha256":sha256_file(cohort/"cohort_registration.json"),
            "method_freeze_sha256":sha256_file(holdout) if holdout.exists() else None,
            "counts_by_arm":counts,"qualified_by_split":bysplit,
            "source_tasks":len(tasks),"mass_attempts":registry["attempted_count"],
            "rows":rows,"alternatives":{"rows":alternative},
            "comparison":"native cuRoboV2-Fast vs same-backend K107 scoring with frozen controls; global/random vs ranking-only vs true feedback-guided modifications",
            "notes":["No conclusions from online K107 screening alone.",
                     "Qualified means independent S8193 + event-aware S32768 physical+Scaled95.",
                     "Original nominal actuator torque limits; payload modifies dynamics.",
                     "One source task across masses is one correlated group; use grouped bootstrap by base_case_id.",
                     "Do not silently pool legacy derated and nominal payload trials.",
                     "Elapsed process runtime may include Python/CUDA initialization; report warm retimer service separately."]}


def publish(document:dict,folder:Path):
    folder.mkdir(parents=True,exist_ok=True)
    summary={k:v for k,v in document.items() if k!="alternatives"}
    (folder/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    (folder/"independently_qualified_alternatives.json").write_text(
        json.dumps(document.get("alternatives",{"rows":[]}),indent=2)+"\n")
    title="<h1>Native cuRoboV2-Fast vs TimeOpt-guided global geometry search</h1>"
    status=html.escape(document["decision"])
    header=("<tr><th>Case</th><th>Payload (kg)</th><th>Split</th><th>Arm</th>"
            "<th>Outcome</th><th>First certified proposal</th>"
            "<th>Duration (s)</th><th>Search wall (s)</th></tr>")
    rows=[]
    for r in document["rows"]:
        v=(r["case_id"],r["mass_kg"],r["split"],r["arm"],r["status"],
           r["first_qualified_proposal"],r["best_qualified_duration_s"],
           r["complete_search_wall_s"])
        rows.append("<tr>"+"".join("<td>"+html.escape(str(x))+"</td>" for x in v)+"</tr>")
    doc="""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TimeOpt-in-the-loop global geometry planning</title>
<style>body{font:15px/1.55 system-ui;margin:2rem auto;max-width:1400px;padding:0 1rem}
table{width:100%;border-collapse:collapse}th,td{padding:.5rem;text-align:left;border-bottom:1px solid #d5dce4}
th{background:#eef4fb}.note{padding:1rem;border-left:5px solid #a66808;background:#fff1d4}</style>"""
    doc+=title+f"<p class='note'><strong>Status:</strong> {status}. "
    doc+="Only independently audited dense/adaptive payload-matched samples count. "
    doc+="No continuous-time certification, broad generalization or native-suboptimality claim is implied.</p>"
    doc+="<h2>Qualified outcomes by arm/split</h2><pre>"+html.escape(
        json.dumps(document.get("qualified_by_split"),indent=2))+"</pre>"
    doc+="<h2>Registered cases</h2><table><thead>"+header+"</thead><tbody>"
    doc+="\n".join(rows)+"</tbody></table>"
    doc+="<p>Every native-failure claim additionally requires the separately independent native audit and complete registered restart budget.</p></html>"
    (folder/"index.html").write_text(doc,encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort",type=Path,required=True)
    p.add_argument("--search-root",type=Path,required=True)
    p.add_argument("--audits",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    a=p.parse_args()
    document=assemble(a.cohort,a.search_root,a.audits)
    publish(document,a.report)
    print(json.dumps({k:v for k,v in document.items()
                      if k in ("decision","source_tasks","counts_by_arm","qualified_by_split")},indent=2))


if __name__=="__main__":main()
