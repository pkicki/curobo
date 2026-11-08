# SPDX-License-Identifier: Apache-2.0
"""Report independent success versus proposal and CPU wall-time budgets.

Preliminary W is NOT a result: success requires an independently validated
physical+Scaled95 adaptive receipt with matching source task/profile SHA and
exact analytic replay. The known offline W witnesses are NEVER counted as a
search result or used to initialize either arm.

Only 3 adversarial development tasks are provided. Do not report
statistical generalization or significance from these 3 observations.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from .cohort_retime import ROOT
from .torque_guided_rescue import ARMS, BUDGETS, OUTPUT
from .geometry_feasibility_core import sha256_file


def qualified(arm:Path,case:str,row:dict)->tuple[bool,list[str]]:
    reasons=[]
    receiptfile=arm/"w_adaptive_receipts.json"
    receipt=(json.loads(receiptfile.read_text()).get(case+":W",{})
             if receiptfile.exists() else {})
    if not row.get("candidate_witness_prescreen"):
        reasons.append("no constructive witness generated")
        return False,reasons
    if receipt.get("physical") is not True or receipt.get("scaled95") is not True:
        reasons.append("independent physical+Scaled95 adaptive receipt missing/failed")
    if receipt.get("collision") is not True or receipt.get("endpoint") is not True:
        reasons.append("independent collision/endpoint qualification missing")
    if receipt.get("torque_profile_sha256")!=row.get("torque_profile_sha256"):
        reasons.append("effort-profile mismatch")
    basefile=arm/case/"w_dense.npz"
    geomfile=arm/case/"w_geometry.json"
    if not basefile.exists() or receipt.get("base_sha256")!=row.get(
        "w_dense_sha256") or receipt.get("base_sha256")!=sha256_file(basefile):
        reasons.append("alternative trajectory missing/checksum mismatch")
    if not geomfile.exists() or receipt.get("witness_geometry_sha256")!=(
        sha256_file(geomfile) if geomfile.exists() else None):
        reasons.append("geometric controls missing/checksum mismatch")
    replay_file=arm/case/"w_adaptive_replay.json"
    replay=json.loads(replay_file.read_text()) if replay_file.exists() else {}
    if replay.get("status")!="REPLAYED":
        reasons.append("exact analytic W adaptive reconstruction missing")
    return not reasons,reasons


def summarize(root:Path)->dict:
    first=root/ARMS[0]/"experiment_config.json"
    if not first.exists():
        registry={}
    else:
        conf=json.loads(first.read_text())
        registry=conf.get("source_registry",{})
    cases={}
    for case in registry:
        cases[case]={}
        for arm_name in ARMS:
            arm=root/arm_name
            result_file=arm/case/"search_result.json"
            if not result_file.exists():
                cases[case][arm_name]={"status":"NOT_RUN"}
                continue
            obj=json.loads(result_file.read_text())
            row=obj["report"]
            valid,reasons=qualified(arm,case,row)
            first=row.get("first_witness_proposal")
            timeline=row.get("ledger",[])
            if valid and (not isinstance(first,int) or first>=len(timeline)):
                valid=False;reasons.append("invalid first witness index")
            elapsed_at_first=(timeline[first]["wall_s"] if valid else None)
            at={str(b):bool(valid and first<=b) for b in BUDGETS}
            cases[case][arm_name]={
                "status":"INDEPENDENTLY_QUALIFIED" if valid
                    else "NO_QUALIFIED_RESCUE",
                "failure_reasons":reasons,
                "candidate_prescreen":bool(row.get("candidate_witness_prescreen")),
                "qualified_by_budget":at,
                "first_qualified_proposal":first if valid else None,
                "first_qualified_elapsed_wall_s":elapsed_at_first,
                "duration_s":row.get("witness_duration_s") if valid else None,
                "total_wall_s":row.get("solver_wall_s_including_gradient_and_screen"),
                "proposals_evaluated":row.get("proposals_evaluated"),
                "rnea_calls":row.get("rnea_calls"),
                "initial_path_lp_expansion":row.get("initial_path_lp_expansion"),
                "initial_native_fit":row.get("base_fit"),
                "task_sha256":row.get("task_sha256"),
                "torque_profile_sha256":row.get("torque_profile_sha256"),
                "witness_sha256":row.get("w_dense_sha256") if valid else None,
                "source_original_native_sha256":row.get(
                    "source_native_buffers_sha256"),
            }
    summary={"schema_version":1,"cases":cases,
             "arms":ARMS,"proposal_budgets":BUDGETS,
             "registered_source_registry":registry,
             "counts":{},
             "note":"3 adversarial examples; no population success-rate inference, no continuous-time physical certificate",
             "fairness":"same nominal-path quintic fit, proposal budget and auditor, different directional evaluation cost; wall time/RNEA must also be compared"}
    # Every completed arm must use the same source, torque-profile and
    # witness registry. Different registries make all cross-arm comparisons
    # N/A, even if a few task names happen to overlap.
    summary["registry_consistent"]=bool(registry)
    for arm in ARMS:
        file=root/arm/"experiment_config.json"
        if file.exists():
            selected=json.loads(file.read_text()).get("source_registry")
            if selected!=registry:
                summary["registry_consistent"]=False
        else:
            summary["registry_consistent"]=False
    for arm in ARMS:
        summary["counts"][arm]={
            "attempted":sum(cases[c][arm]["status"]!="NOT_RUN" for c in cases),
            "qualified":sum(cases[c][arm]["status"]=="INDEPENDENTLY_QUALIFIED"
                             for c in cases),
            "qualified_by_proposals":{
                str(b):sum(cases[c][arm].get("qualified_by_budget",{}).get(str(b),False)
                           for c in cases) for b in BUDGETS},
        }
    if not summary["registry_consistent"]:
        summary["warning"]="ARMS MISSING OR SOURCE REGISTRIES DIFFER: no controlled cross-arm comparison qualified"
    return summary


def publish(summary:dict,output:Path):
    output.mkdir(parents=True,exist_ok=True)
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    rows=[]
    for case,groups in summary["cases"].items():
        for arm,data in groups.items():
            rows.append("<tr><td>"+html.escape(case)+"</td><td>"+html.escape(arm)+
                        "</td><td>"+html.escape(data["status"])+
                        "</td><td>"+str(data.get("first_qualified_proposal","N/A"))+
                        "</td><td>"+str(data.get("first_qualified_elapsed_wall_s","N/A"))+
                        "</td><td>"+str(data.get("duration_s","N/A"))+
                        "</td><td>"+str(data.get("rnea_calls","N/A"))+"</td></tr>")
    sections=[]
    for arm,record in summary["counts"].items():
        by=record["qualified_by_proposals"]
        sections.append("<tr><td>"+html.escape(arm)+"</td>"+
                        "".join("<td>"+str(by[str(b)])+"/"+str(record["attempted"])+
                                "</td>" for b in BUDGETS)+"</tr>")
    page="""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Torque-LP-dual guided geometric rescue</title>
<style>body{font:15px/1.55 system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}
table{border-collapse:collapse;width:100%}td,th{padding:.6rem;border-bottom:1px solid #ddd;text-align:left}
th{background:#eff2f8}.note{padding:1rem;background:#fff0d8;border-left:5px solid #ac680a}</style>
<h1>Can torque-LP dual guidance rescue geometrically infeasible paths?</h1>
<p class="note">Only independently reaudited physical+Scaled95 witness trajectories
count as successes. Raw solver screens and replay-only candidates are NOT proof.
The three source tasks were selected through an earlier adversarial discovery;
these results do not estimate population performance. This is NOT continuous-time
feasibility certification. All actuator constraints are synthetic joint-selective
restrictions, not Panda manufacturer's ratings.</p>"""
    page+="<h2>Qualified rescue by proposal budget</h2><table><thead><tr><th>Arm</th>"
    page+="".join(f"<th>≤{b} proposals</th>" for b in BUDGETS)
    page+="</tr></thead><tbody>"+"".join(sections)+"</tbody></table>"
    page+="<h2>Source-task outcomes</h2><table><thead><tr><th>Case</th><th>Search arm</th><th>Qualification</th>"
    page+="<th>First success proposal</th><th>Time at success (s)</th><th>Motion duration (s)</th>"
    page+="<th>RNEA calls</th></tr></thead><tbody>"+"\n".join(rows)+"</tbody></table>"
    page+="<p>Compare BOTH number of proposals and measured wall time/RNEA calls. LP dual finite "
    page+="differences are more costly than random perturbations. All candidates use the same initial "
    page+="nominal-path quintic fit and the same collision/torque screens. Independent native pre-TimeOpt "
    page+="stage traces are separately diagnostic and cannot be used as uninstrumented planner timings.</p>"
    page+="<h2>Unfiltered machine-readable evidence</h2><pre>"+html.escape(
        json.dumps(summary["counts"],indent=2))+"</pre></html>"
    (output/"index.html").write_text(page,encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUTPUT)
    p.add_argument("--output",type=Path,default=ROOT/
                   "reports/joint_time_oracle/torque_lp_guided_rescue")
    a=p.parse_args()
    result=summarize(a.root)
    publish(result,a.output)
    print(json.dumps({"counts":result["counts"]},indent=2))


if __name__=="__main__":
    main()
