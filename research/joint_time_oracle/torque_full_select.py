# SPDX-License-Identifier: Apache-2.0
"""Compare full torque-aware native replanning with B and D at fixed budgets.

Unvalidated native outputs never enter the selected-best result. Sampled
and independently adaptive-qualified comparisons are separate, with missing
receipts failing closed. No implicit continuous-time certification.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import statistics

from .cohort_statistics import bootstrap_median
from .cohort_torque_report import sampled_acceptance
from .torque_active import read_profile

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/"research/joint_time_oracle/results/torque_stress_final_20261107"
BUDGETS=(1,4,16,32)


def best_native(candidates:dict, diagnostics:dict, receipts:dict, profile:dict,
                case:str, budget:int, *, scaled95:bool, strict:bool,
                prefix:str="F") -> dict | None:
    valid=[]
    for method,candidate in sorted(candidates.items()):
        # F000 is the native selected lane; F000L02 is an additional internally
        # accepted lane from the same planner restart. Both consume one restart.
        if not method.startswith(prefix) or len(method)<len(prefix)+3:
            continue
        try:
            restart=int(method[len(prefix):len(prefix)+3])
        except ValueError:
            continue
        if restart>=budget:
            continue
        if not isinstance(candidate,dict) or candidate.get("status")!=("COMPLETE" if prefix=="R" else "SUCCESS"):
            continue
        if candidate.get("restart",restart)!=restart:
            continue
        if candidate.get("torque_profile_sha256")!=profile["profile_sha256"]:
            continue
        duration=candidate.get("duration_s")
        if not isinstance(duration,(int,float)) or duration<=0:
            continue
        diag=diagnostics.get(case+":"+method)
        if not sampled_acceptance(diag,profile,scaled=scaled95)[0]:
            continue
        if strict:
            receipt=receipts.get(case+":"+method)
            if (not isinstance(receipt,dict) or
                receipt.get("torque_profile_sha256")!=profile["profile_sha256"] or
                receipt.get("physical") is not True or
                receipt.get("endpoint") is not True or
                receipt.get("collision") is not True or
                (scaled95 and receipt.get("scaled95") is not True)):
                continue
        valid.append({"method":method,"duration_s":float(duration),
                      "seed":candidate.get("seed"),
                      "planning_wall_s":candidate.get("plan_wall_s")})
    return min(valid,key=lambda x:x["duration_s"]) if valid else None


def summary(values:list[float],attempted:int) -> dict:
    n=len(values)
    return {"paired_n":n,"attempted_n":attempted,
            "median_pct":statistics.median(values) if n else None,
            "mean_pct":statistics.mean(values) if n else None,
            "paired_bootstrap95_median_pct":bootstrap_median(values) if n>=2 else None,
            "improved_count":sum(x>0 for x in values),
            "slower_count":sum(x<0 for x in values),
            "improved_gt_1pct":sum(x>1 for x in values),
            "improved_gt_5pct":sum(x>5 for x in values),
            "improved_gt_10pct":sum(x>10 for x in values)}


def build(arm:Path, *, strict_receipts:Path|None=None) -> dict:
    native=arm/"native_full"
    diagnostics=json.loads((native/"independent_sampled.json").read_text()) if (
        native/"independent_sampled.json").exists() else {}
    receipts=json.loads(strict_receipts.read_text()) if (
        strict_receipts and strict_receipts.exists()) else {}
    existing={}
    for label in ("torque_qualified","torque_sampled"):
        file=arm/(label+".jsonl")
        existing[label]={item["case_id"].split("::")[-1]:item for item in
                         (json.loads(x) for x in file.read_text().splitlines() if x.strip())} if file.exists() else {}
    tasks=json.loads((arm/"tasks.json").read_text())
    data={"source_task_count":len(tasks),"frozen_budgets":list(BUDGETS),
          "qualified_definition":"source D/B independently sampled and adaptive verified; native F independently sampled and matching adaptive receipt",
          "sampled_definition":"independent Pinocchio/sphere 8193 samples; continuous-time safety NOT certified",
          "candidate_selection":"best verified native restart by duration, never solver-only success",
          "results":{}}
    for mode in ("physical","scaled95"):
        for strict in (False,True):
            label=("adaptive_qualified" if strict else "sampled")+":"+mode
            cases={}
            for task in tasks:
                case=task["case_id"]
                profile=read_profile(arm/case/"torque_profile.json")
                candidate_path=native/case/"native_candidates.json"
                candidates=json.loads(candidate_path.read_text()) if candidate_path.exists() else {}
                matched_file=native/case/"matched_retime_candidates.json"
                matched=json.loads(matched_file.read_text()) if matched_file.exists() else {}
                source=existing["torque_qualified" if strict else "torque_sampled"].get(case,{})
                methods=source.get("methods",{})
                relevant={}
                for name in ("B","D"):
                    v=methods.get(name,{})
                    checks=v.get("verification",{})
                    flag=checks.get("scaled95" if mode=="scaled95" else "physical")
                    duration=v.get("duration_s")
                    relevant[name]=float(duration) if flag is True and isinstance(duration,(int,float)) and duration>0 else None
                pair={}
                for budget in BUDGETS:
                    best=best_native(candidates,diagnostics,receipts,profile,case,budget,
                                     scaled95=(mode=="scaled95"),strict=strict)
                    improvement=None
                    baseline=None
                    matched_improvement=None
                    matched_from_B=None
                    retiming_difference=None
                    matched_best=best_native(matched,diagnostics,receipts,profile,case,budget,
                                             scaled95=(mode=="scaled95"),strict=strict,prefix="R")
                    if best and relevant["D"] is not None:
                        improvement=100*(best["duration_s"]-relevant["D"])/best["duration_s"]
                    if best and relevant["B"] is not None:
                        baseline=100*(relevant["B"]-best["duration_s"])/relevant["B"]
                    if matched_best and relevant["D"] is not None:
                        matched_improvement=100*(matched_best["duration_s"]-relevant["D"])/matched_best["duration_s"]
                    if matched_best and relevant["B"] is not None:
                        matched_from_B=100*(relevant["B"]-matched_best["duration_s"])/relevant["B"]
                    if matched_best and best:
                        retiming_difference=100*(best["duration_s"]-matched_best["duration_s"])/best["duration_s"]
                    pair[str(budget)]={"best_native":best,"best_matched":matched_best,
                                       "native_to_D_improvement_pct":improvement,
                                       "B_to_native_improvement_pct":baseline,
                                       "matched_to_D_improvement_pct":matched_improvement,
                                       "B_to_matched_improvement_pct":matched_from_B,
                                       "native_to_matched_improvement_pct":retiming_difference}
                cases[case]={"population":task["population"],"reference":relevant,
                             "budget_comparisons":pair}
            groups={}
            for pop in ("all",*sorted({t["population"] for t in tasks})):
                selected=[r for r in cases.values() if pop=="all" or r["population"]==pop]
                for budget in BUDGETS:
                    records=[r["budget_comparisons"][str(budget)] for r in selected]
                    groups[pop+":"+str(budget)]={
                        "native_to_D":summary([x["native_to_D_improvement_pct"] for x in records
                                               if x["native_to_D_improvement_pct"] is not None],len(records)),
                        "B_to_native":summary([x["B_to_native_improvement_pct"] for x in records
                                               if x["B_to_native_improvement_pct"] is not None],len(records)),
                        "matched_to_D":summary([x["matched_to_D_improvement_pct"] for x in records
                                               if x["matched_to_D_improvement_pct"] is not None],len(records)),
                        "B_to_matched":summary([x["B_to_matched_improvement_pct"] for x in records
                                               if x["B_to_matched_improvement_pct"] is not None],len(records)),
                        "native_to_matched":summary([x["native_to_matched_improvement_pct"] for x in records
                                               if x["native_to_matched_improvement_pct"] is not None],len(records))}
            data["results"][label]={"groups":groups,"cases":cases}
    native.mkdir(parents=True,exist_ok=True)
    (native/"full_planner_comparison.json").write_text(json.dumps(data,indent=2)+"\n")
    sections=[]
    for mode in ("physical","scaled95"):
        for status in ("adaptive_qualified","sampled"):
            title=status+":"+mode
            sections.append("<h2>"+html.escape(title)+"</h2>")
            sections.append("<table><thead><tr><th>Budget (restarts)</th><th>Full/D pairs</th><th>Median Full→D</th><th>95% paired CI</th><th>Median B→Full</th><th>Full-R/D pairs</th><th>Median Full-R→D</th><th>Median B→Full-R</th><th>Median Full→Full-R</th></tr></thead><tbody>")
            groups=data["results"][title]["groups"]
            for budget in BUDGETS:
                result=groups["all:"+str(budget)]
                a,b=result["native_to_D"],result["B_to_native"]
                m,mb,mf=(result["matched_to_D"],result["B_to_matched"],result["native_to_matched"])
                fmt=lambda x: "N/A" if x is None else f"{x:.3f}%"
                ci="N/A" if a["paired_bootstrap95_median_pct"] is None else str(a["paired_bootstrap95_median_pct"])
                sections.append(f"<tr><td>{budget}</td><td>{a['paired_n']}/{a['attempted_n']}</td><td>"+
                                fmt(a["median_pct"])+"</td><td>"+html.escape(ci)+"</td><td>"+
                                fmt(b["median_pct"])+"</td><td>"+str(m["paired_n"])+"/"+str(m["attempted_n"])+
                                "</td><td>"+fmt(m["median_pct"])+
                                "</td><td>"+fmt(mb["median_pct"])+
                                "</td><td>"+fmt(mf["median_pct"])+"</td></tr>")
            sections.append("</tbody></table>")
    page="""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Full native torque-aware planning comparison</title><style>
body{font:16px/1.5 system-ui;max-width:1050px;margin:2rem auto;padding:0 1rem}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:.6rem;text-align:left}
aside{background:#fff0d8;padding:1rem;border-left:5px solid #b96c00}</style>
<h1>Replanning cuRobo under derated torque limits</h1>
<aside>Candidate budgets count independent planner restarts; each restart contains
multiple internal seeds. All results require explicit independent validation.
Sampled evidence is NOT a continuous-time certificate. Full-R is the independent IPOPT retiming of Full geometry, matching B and D. A positive Full→D
percentage means D is faster than the best validated native full-planner candidate
within that budget. Missing results are not treated as zero improvements.</aside>"""
    page+="\n".join(sections)+"</html>"
    (native/"index.html").write_text(page)
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=DEFAULT)
    parser.add_argument("--regime",required=True,choices=("moderate","severe"))
    parser.add_argument("--adaptive-receipts",type=Path)
    args=parser.parse_args()
    data=build(args.root/args.regime,strict_receipts=args.adaptive_receipts)
    print(json.dumps({"source_task_count":data["source_task_count"],
                      "report":str(args.root/args.regime/"native_full"/"index.html")},indent=2))


if __name__=="__main__":
    main()
