# SPDX-License-Identifier: Apache-2.0
"""Publish descriptive torque-component and independently-qualified native-baseline evidence."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/"research/joint_time_oracle/results/torque_stress_final_20261107"
REPORT=ROOT/"reports/joint_time_oracle/torque_mechanisms"


def _fmt(value,unit="",digits=3):
    return "N/A" if value is None else f"{value:.{digits}f}{unit}"


def publish(source:Path,output:Path) -> dict:
    output.mkdir(parents=True,exist_ok=True)
    summary={"status":"EXPLORATORY_OFFLINE_ANALYSIS",
             "source_root":str(source),"regimes":{}}
    sections=[]
    for regime in ("moderate","severe"):
        arm=source/regime
        mechanisms=arm/"mechanisms"/"summary.json"
        full=arm/"native_full"/"full_planner_comparison.json"
        items=json.loads(mechanisms.read_text())["cases"] if mechanisms.exists() else []
        comparison=json.loads(full.read_text()) if full.exists() else None
        completed=[x for x in items if x.get("status")=="COMPLETE"]
        completed.sort(key=lambda x:x["B_to_D_improvement_pct"],reverse=True)
        summary["regimes"][regime]={"decomposition_attempted":len(items),
                                     "decomposition_complete":len(completed),
                                     "native_full_comparison":full.exists()}
        sections.append(f"<section><h2>{html.escape(regime.title())} torque stress</h2>")
        sections.append(f"<p>{len(completed)}/{len(items)} B/D cases decomposed; "
                        "RMS contributions are descriptive and not causal attributions.</p>")
        sections.append("<table><thead><tr><th>Case</th><th>B→D (%)</th>"
                        "<th>Joint path length Δ (%)</th><th>Gravity RMS Δ (N·m)</th>"
                        "<th>Inertia RMS Δ (N·m)</th><th>Velocity-dependent RMS Δ (N·m)</th>"
                        "</tr></thead><tbody>")
        for x in completed:
            sections.append("<tr><td>"+html.escape(x["case_id"])+"</td>"+
                            "".join("<td>"+_fmt(x.get(key))+"</td>" for key in (
                                "B_to_D_improvement_pct","path_length_increase_pct",
                                "gravity_rms_delta_Nm","inertia_rms_delta_Nm",
                                "velocity_bias_rms_delta_Nm"))+"</tr>")
        sections.append("</tbody></table>")
        if comparison:
            sections.append("<h3>Full native replanning versus D</h3>")
            for qualification in ("adaptive_qualified:physical","sampled:physical"):
                groups=comparison["results"][qualification]["groups"]
                sections.append("<h4>"+qualification+"</h4><table><thead><tr>"
                                "<th>Restarts</th><th>Verified pairs</th>"
                                "<th>Median Full→D (%)</th><th>Median B→Full (%)</th>"
                                "<th>Matched Full-R/D pairs</th><th>Matched Full-R→D (%)</th>"
                                "<th>Median Full→Full-R timing (%)</th></tr></thead><tbody>")
                for budget in (1,4,16,32):
                    result=groups["all:"+str(budget)]
                    a,b=result["native_to_D"],result["B_to_native"]
                    m,mf=result["matched_to_D"],result["native_to_matched"]
                    sections.append(f"<tr><td>{budget}</td><td>{a['paired_n']}/{a['attempted_n']}</td>"
                                    "<td>"+_fmt(a["median_pct"])+"</td>"
                                    "<td>"+_fmt(b["median_pct"])+"</td>"
                                    "<td>"+str(m["paired_n"])+"/"+str(m["attempted_n"])+"</td>"
                                    "<td>"+_fmt(m["median_pct"])+"</td>"
                                    "<td>"+_fmt(mf["median_pct"])+"</td></tr>")
                sections.append("</tbody></table>")
        else:
            sections.append("<p><strong>Full native rerun: NOT EXECUTED or results not published.</strong></p>")
        sections.append("</section>")
    page="""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Torque mechanism and full cuRobo replanning study</title><style>
body{font:16px/1.5 system-ui;max-width:1150px;margin:2rem auto;padding:0 1rem;color:#1d2938}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:.55rem;text-align:left}
.note{background:#fff0d8;border-left:5px solid #b86c00;padding:1rem}
</style><h1>Why geometry helps with active torque constraints</h1>
<p class="note"><strong>Qualification.</strong> RNEA decomposition describes the
gravity, inertial and velocity-dependent terms along different trajectories.
Differences are not causal attribution. Native replanning comparisons use
the BEST independently validated trajectory at each fixed restart budget.
Missing adaptive verification receipts do not qualify candidates.
Dense/adaptive sample checks are not continuous-time certificates.</p>
<p>Source problems and synthetic derated actuator constraints are frozen;
no real payload, global optimality, or real-time latency claim is made.</p>"""
    page+="\n".join(sections)+"</html>"
    (output/"index.html").write_text(page,encoding="utf-8")
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=DEFAULT)
    parser.add_argument("--report",type=Path,default=REPORT)
    a=parser.parse_args()
    print(json.dumps(publish(a.source,a.report),indent=2))


if __name__=="__main__":
    main()
