# SPDX-License-Identifier: Apache-2.0
"""Independent result classification for margin-aware fixed-geometry retiming.

Source P is NOT Scaled95 qualified; original slow W is. Only the NEW
R with independent sampled+adaptive physical AND Scaled95 receipt
can be promoted. Report optimization results and native numerical
trace as separate evidence, without conflating feasibility, short
duration, hardware safety or a specific root cause of NaN.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from .cohort_retime import ROOT
from .torque_margin_core import (
    CASE_IDS, DILATION_FACTORS, TIGHTENING_FACTORS, MIN_DURATION_IMPROVEMENT,
    sha256_file,
)
from .torque_margin_prepare import OUTPUT

REPORT=ROOT/"reports/joint_time_oracle/torque_margin_retiming"


def arms()->tuple[str,...]:
    return ("dilation",)+tuple(f"tight_b{round(b*1000):03d}" for b in TIGHTENING_FACTORS)


def inspect_native_trace(native_arm:Path|None)->dict:
    if native_arm is None:
        return {"status":"NOT_EXECUTED"}
    if not native_arm.exists():
        return {"status":"NATIVE_TRACE_ROOT_MISSING"}
    out={}
    for case in CASE_IDS:
        traces=sorted((native_arm/case).glob("*_timeopt_phase_trace.json"))
        reports=[]
        for file in traces:
            try:
                content=json.loads(file.read_text())
                reports.append({"sha256":sha256_file(file),
                                "file":str(file),
                                "status":content.get("status"),
                                "first_observed_problem_per_lane":[
                                    lane.get("first_observed_problem") for lane in
                                    content.get("per_lane",[])]})
            except Exception as exc:
                reports.append({"file":str(file),"status":"TRACE_READ_ERROR",
                                "reason":repr(exc)})
        out[case]={"captured_snapshots":len(reports),"traces":reports}
    return {"status":"EXAMINED_OBSERVATIONAL_TRACES","cases":out,
            "scope":"observational post-finalizer buffers only; no causal isolation without independent controls"}


def collect(root:Path,*,native_arm:Path|None=None)->dict:
    registration=root/"margin_registration.json"
    if not registration.exists():
        return {"schema_version":1,"status":"NOT_EXECUTED","cases":[],
                "classification_counts":{},"native_trace":inspect_native_trace(native_arm)}
    manifest=json.loads(registration.read_text())
    source={row["case_id"]:row for row in manifest["sources"]}
    rows=[]
    for case in CASE_IDS:
        old=source[case]
        for arm_name in arms():
            folder=root/arm_name/case
            result_file=folder/"screen_result.json"
            row={"case_id":case,"arm":arm_name,"status":"NOT_EXECUTED",
                 "source_W_duration_s":old["source_W_duration_s"],
                 "source_P_duration_s":old["source_P_duration_s"],
                 "qualified_scaled95":False}
            if result_file.exists():
                result=json.loads(result_file.read_text())
                row["source_result_sha256"]=sha256_file(result_file)
                row["method_result"]=result
                receiptfile=root/arm_name/"r_adaptive_receipts.json"
                receipts=json.loads(receiptfile.read_text()) if receiptfile.exists() else {}
                receipt=receipts.get(case+":R",{})
                registration_file=folder/"r_registration.json"
                if registration_file.exists():
                    frozen=json.loads(registration_file.read_text())
                    row["R_duration_s"]=frozen["r_duration_s"]
                    row["R_sha256"]=frozen["r_sha256"]
                    row["source_geometry_sha256"]=frozen["source_files_sha256"]["w_geometry.json"]
                    if (frozen["profile_sha256"]!=old["profile_sha256"] or
                        frozen["source_case_sha256"]!=old["task_sha256"]):
                        row["status"]="FROZEN_SOURCE_MISMATCH"
                    elif (not (folder/"r_dense.npz").exists() or
                          sha256_file(folder/"r_dense.npz")!=frozen["r_sha256"]):
                        row["status"]="R_SOURCE_SHA_MISMATCH"
                    elif receipt.get("r_sha256")!=frozen["r_sha256"]:
                        row["status"]="NO_MATCHING_INDEPENDENT_R_RECEIPT"
                    elif receipt.get("physical") is True and receipt.get("scaled95") is True:
                        row["qualified_scaled95"]=True
                        faster=frozen["r_duration_s"]<(
                            1-MIN_DURATION_IMPROVEMENT)*old["source_W_duration_s"]
                        row["duration_improvement_vs_W_pct"]=100*(
                            1-frozen["r_duration_s"]/old["source_W_duration_s"])
                        row["status"]=("QUALIFIED_SCALED95_SHORTER"
                                       if faster else "QUALIFIED_SCALED95_NO_MATERIAL_GAIN")
                    elif receipt:
                        row["status"]="INDEPENDENT_R_UNQUALIFIED"
                    else:
                        row["status"]="R_PRELIMINARY_UNAUDITED"
                else:
                    row["status"]="NO_CANDIDATE" if result.get("reason") else "UNQUALIFIED_SOLVER"
            rows.append(row)
    counts={}
    for row in rows:
        counts[row["status"]]=counts.get(row["status"],0)+1
    successes=sum(row["status"]=="QUALIFIED_SCALED95_SHORTER" for row in rows)
    return {"schema_version":1,"status":"REGISTERED_RESULT_SUMMARY",
            "registration_sha256":sha256_file(registration),
            "registered_cases":list(CASE_IDS),"registered_arms":list(arms()),
            "qualification":"physical+Scaled95 independent 8193+adaptive Pinocchio sampled",
            "material_duration_gain_threshold_pct":100*MIN_DURATION_IMPROVEMENT,
            "rows":rows,"classification_counts":counts,
            "qualified_shorter_count":successes,
            "native_trace":inspect_native_trace(native_arm),
            "decision":"NOT_READY_UNAUDITED" if any(x["status"] in (
                  "NOT_EXECUTED","R_PRELIMINARY_UNAUDITED","NO_MATCHING_INDEPENDENT_R_RECEIPT")
                  for x in rows) else
                  "RESEARCH_GO_MARGIN_RECOVERY" if successes else
                  "NO_GO_MARGIN_RECOVERY",
            "note":"all three tasks intentionally selected; no population inference, continuous-time torque/collision certificate or hardware safety claim"}


def publish(data:dict,path:Path):
    path.mkdir(parents=True,exist_ok=True)
    (path/"summary.json").write_text(json.dumps(data,indent=2)+"\n")
    cols=("case_id","arm","status","source_W_duration_s",
          "source_P_duration_s","R_duration_s","duration_improvement_vs_W_pct")
    tab="<table><thead><tr>"+"".join("<th>"+x+"</th>" for x in cols)+"</tr></thead><tbody>"
    for row in data.get("rows",[]):
        tab+="<tr>"+"".join("<td>"+html.escape(str(row.get(k,"—")))+"</td>" for k in cols)+"</tr>"
    tab+="</tbody></table>"
    page="""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Margin-aware time-optimal retiming and native TimeOpt NaN provenance</title>
<style>body{font:15px/1.55 system-ui;max-width:1200px;margin:2rem auto;padding:1rem}
.note{background:#fff0da;border-left:5px solid #b77918;padding:1rem}
table{border-collapse:collapse;width:100%}
th,td{padding:.5rem;border-bottom:1px solid #dce3e8;text-align:left}
th{background:#edf2f9}</style>
<h1>Fixed-geometry Scaled95 margin recovery</h1>
<p class="note">Source W is independently Scaled95 feasible; original IPOPT P
is only independently physical feasible. A shorter R is only promoted if
independent dense+adaptive physical AND Scaled95 checks pass. The trajectories
use synthetic actuator derating. No continuous-time torque/collision
certificate or native-TimeOpt causal diagnosis is implied.</p>"""
    page+="<h2>Status and classification</h2><pre>"+html.escape(json.dumps({
        k:data.get(k) for k in ("status","decision","classification_counts")
    },indent=2))+"</pre>"
    page+=tab
    page+="<h2>Native TimeOpt numerical trace (separate)</h2><pre>"
    page+=html.escape(json.dumps(data.get("native_trace"),indent=2))+"</pre>"
    page+="<p>Compare duration, margin, wall time and independent validation. IPOPT"
    page+=" convergence and raw NaN flags alone are not physical qualification or"
    page+=" exclusive root-cause evidence.</p></html>"
    (path/"index.html").write_text(page,encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUTPUT)
    p.add_argument("--report",type=Path,default=REPORT)
    p.add_argument("--native-arm",type=Path)
    a=p.parse_args()
    result=collect(a.root,native_arm=a.native_arm)
    publish(result,a.report)
    print(json.dumps({"status":result["status"],"decision":result.get("decision"),
                      "counts":result.get("classification_counts")},indent=2))


if __name__=="__main__":
    main()
