# SPDX-License-Identifier: Apache-2.0
"""Fail-closed compact HTML/JSON aggregation of matched nominal payload study.

All numerator claims require independently qualified real trajectories
with exact task/payload/limit/collision/model hashes. No invented
success rates on unrun cohorts. Broad and gravity-enriched populations
are never pooled as a single unbiased estimate.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from .payload_hardcase_registry import classify_rows

BUDGETS_MS=(100,300,1000,3000)
ARMS=("native","blind","rank_only","feedback")


def collect(rows:list[dict])->dict:
    outcomes=classify_rows(rows)
    matrix={}
    for row in rows:
        kind=row.get("cohort","UNSPECIFIED")
        task=row["task_id"]
        scenario=row["scenario_id"]
        for arm in ARMS:
            key=(kind,arm)
            cell=matrix.setdefault(key,{"attempts":0,"qualified":0,
                                         "qualified_within_budget":{str(x):0 for x in BUDGETS_MS},
                                         "time_ms":[]})
            candidate=row.get("evaluations",{}).get(arm)
            if not isinstance(candidate,dict):
                continue
            cell["attempts"]+=1
            audit=candidate.get("independent_audit",{})
            passed=(audit.get("physical") is True and
                    audit.get("scaled95") is True and
                    audit.get("collision") is True and
                    audit.get("endpoint") is True and
                    audit.get("model_alignment_pass") is True and
                    audit.get("dense_samples",0)>=8193 and
                    audit.get("adaptive_samples",0)>=8193)
            wall=float(candidate.get("complete_request_ms",float("inf")))
            if passed:
                cell["qualified"]+=1
                if wall>0 and wall<float("inf"):
                    cell["time_ms"].append(wall)
                    for budget in BUDGETS_MS:
                        if wall<=budget:
                            cell["qualified_within_budget"][str(budget)]+=1
    by_cohort={}
    for (kind,arm),v in matrix.items():
        by_cohort.setdefault(kind,{})[arm]=v
    return {"schema_version":1,"status":"NO_RESULTS" if not rows else "EXPERIMENTAL_EVIDENCE",
            "case_count":len(rows),"hard_case_counts":outcomes["counts"],
            "hard_case_rows":outcomes["rows"],"matched_planning":by_cohort,
            "limitations":"sampled physical audits; synthetic payload collision sphere; split by source task; no population confidence intervals without independent task bootstrap"}


def render(data:dict,folder:Path):
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"summary.json").write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    summary=html.escape(json.dumps({k:data.get(k) for k in (
        "status","case_count","hard_case_counts","matched_planning")},indent=2))
    page="""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nominal payload TimeOpt-guided planning</title>
<style>body{font:15px/1.55 system-ui;max-width:1200px;margin:2rem auto;padding:1rem}
.note{padding:1rem;background:#fff1db;border-left:5px solid #b87913}
pre{overflow:auto;background:#f2f5f9;padding:1rem}</style>
<h1>Nominal-effort Panda: native TimeOpt versus K107-guided path planning</h1>
<p class="note">A native failure is not a confirmed hard case without an
independently verified feasible alternative. All nominal actuator efforts
remain unchanged; physically attached payload inertia and collisions must
match native and independent models. Source-task-split broad and enriched
cohorts are distinct. Real CPU/GPU experiments are required before any
solver performance claim. Sampled audits are not formal continuous-time
certification.</p>"""
    page+=f"<h2>Scored evidence</h2><pre>{summary}</pre>"
    page+="<p>See TIMEOPT_GUIDED_NOMINAL_PAYLOAD_PROTOCOL_20261110.md for"
    page+=" the source and full experiment gate. Raw hash manifests and"
    page+=" native seed/retimer traces are mandatory for a qualified report.</p>"
    page+="</html>"
    (folder/"index.html").write_text(page)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--report",required=True,type=Path)
    a=p.parse_args()
    data=collect(json.loads(a.input.read_text()))
    render(data,a.report)
    print(json.dumps({"status":data["status"],"case_count":data["case_count"],
                      "hard_case_counts":data["hard_case_counts"]},indent=2))


if __name__=="__main__":
    main()
