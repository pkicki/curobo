# SPDX-License-Identifier: Apache-2.0
"""Publish torque-stress HTML summaries from frozen raw evidence.

Re-runnable offline. No solver runs, no hidden filtering, and no fabricated
durations. Run cohort_torque_report for each regime before publishing.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
STRESS=ROOT/"research/joint_time_oracle/results/torque_stress_20261107"
REPORT=ROOT/"reports/joint_time_oracle/torque_active"


def fmt(x):
    return "N/A" if x is None else f"{x:.3f}%"


def publish(source: Path, destination: Path) -> dict:
    destination.mkdir(parents=True,exist_ok=True)
    rows=[];summary={"status":"NO QUALIFIED DATA UNTIL INDEPENDENT ADAPTIVE RECEIPTS ARE PRESENT",
                     "regimes":{}}
    for regime in ("moderate","severe"):
        arm=source/regime
        path=arm/"torque_summary.json"
        if not path.exists():
            rows.append(f"<tr><td>{regime}</td><td colspan='7'>Not executed / no summary</td></tr>")
            summary["regimes"][regime]={"status":"NOT_EXECUTED"}
            continue
        record=json.loads(path.read_text())
        target=destination/regime
        target.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(arm/"index.html",target/"index.html")
        shutil.copyfile(path,target/"torque_summary.json")
        strict=record["adaptive_qualified"]["groups"]["all"]["comparisons"]
        sampled=record["sampled"]["groups"]["all"]["comparisons"]
        primary=strict["physical:B->D"]; exploratory=sampled["physical:B->D"]
        activity=[x for x in record["torque_utilization"] if x["method"]=="B"]
        active=sum(x["torque_ge_85pct"] for x in activity)
        out={"attempted":record["source_case_count"],
             "B_D_qualified_pairs":primary["eligible_pairs"],
             "B_D_qualified_median_pct":primary["median_improvement_pct"],
             "B_D_sampled_pairs":exploratory["eligible_pairs"],
             "B_D_sampled_median_pct":exploratory["median_improvement_pct"],
             "B_torque_ge_85pct":active,"B_torque_reported":len(activity)}
        summary["regimes"][regime]=out
        rows.append(
            "<tr><td><a href='"+regime+"/index.html'>"+regime+"</a></td>"+
            f"<td>{out['attempted']}</td><td>{out['B_D_qualified_pairs']}</td>"+
            f"<td>{fmt(out['B_D_qualified_median_pct'])}</td>"+
            f"<td>{out['B_D_sampled_pairs']}</td>"+
            f"<td>{fmt(out['B_D_sampled_median_pct'])}</td>"+
            f"<td>{active}/{len(activity)}</td></tr>")
    page="""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Joint-time oracle — torque-active study</title><style>
body{font:16px/1.5 system-ui;margin:2rem auto;max-width:1150px;padding:0 1rem}
table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #ddd;padding:.6rem;text-align:left}
.note{background:#fff1e4;border-left:5px solid #bd6e14;padding:1rem}
</style><h1>Torque-active geometric motion planning</h1>
<p class="note"><strong>Experimental and local-solution status.</strong> The
torque derating in this study simulates a modified actuator, NOT a real Panda
payload or manufacturer effort rating. Strict duration comparisons require
independent adaptive verification receipts. Separately shown sampled-only
observations are not continuous-time certificates. A B-to-D reduction is
an observed local geometric benefit, not a global-optimality certificate.</p>
<p>Each source task can appear in both regimes; these paired variants are NOT
independent observations. Do not combine their sample counts for a naive CI.</p>
<h2>Primary geometry contrast: B to D</h2>
<table><thead><tr><th>Regime</th><th>Tasks</th><th>Qualified pairs</th>
<th>Qualified median</th><th>Sampled pairs</th><th>Sampled median</th>
<th>B torque >=85%</th></tr></thead><tbody>"""+ "\n".join(rows) + """</tbody></table>
<p>See per-regime pages for all contrasts A10→A40, A40→B, B→C and B→D,
scene strata, failures and individual torque utilizations.</p>
<p>The baseline here is fixed-geometry replay under the changed torque limits,
not a freshly geometry-optimized full cuRobo motion planner.</p></html>"""
    (destination/"index.html").write_text(page,encoding="utf-8")
    (destination/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=STRESS)
    p.add_argument("--report",type=Path,default=REPORT)
    a=p.parse_args()
    print(json.dumps(publish(a.source,a.report),indent=2))


if __name__=="__main__":
    main()
