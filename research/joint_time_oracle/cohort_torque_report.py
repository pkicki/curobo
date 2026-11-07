# SPDX-License-Identifier: Apache-2.0
"""Fail-closed torque-stress aggregation and offline HTML report.

Independent sampled checks are exploratory. Qualification requires an
additional explicit per-case adaptive verification receipt: missing receipts
cannot pass. Never count multiple deratings of one source path as independent
tasks in the same confidence interval.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from .cohort_statistics import analyze, html_report
from .torque_active import read_profile, torque_utilization

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "research/joint_time_oracle/results/torque_stress_20261107"
METHODS = ("A10", "A40", "B", "C", "D")
TOLERANCES = {"position":1e-5, "velocity":1e-4, "acceleration":1e-3,
              "jerk":2e-2, "torque":1e-3}
ENDPOINT_TOL = {"position_start":1e-5, "position_goal":1e-5,
                "velocity":1e-4, "acceleration":1e-3}


def sampled_acceptance(diag: dict | None, profile: dict, *, scaled: bool) -> tuple[bool,str]:
    if not isinstance(diag, dict) or diag.get("samples", 0) < 8193:
        return False, "absent or insufficiently dense independent sampled validation"
    if diag.get("torque_profile_sha256") != profile["profile_sha256"]:
        return False, "torque profile checksum mismatch"
    world = diag.get("world_min_clearance_m")
    self_clear = diag.get("self_min_clearance_m")
    if (world is not None and world < -1e-7) or self_clear is None or self_clear < -1e-7:
        return False, "sampled world/self collision clearance violation"
    for key,tolerance in ENDPOINT_TOL.items():
        if not np.isfinite(diag.get("endpoints",{}).get(key, float("nan"))) or diag["endpoints"][key]>tolerance:
            return False, f"endpoint: {key}"
    regime="scaled95" if scaled else "physical"
    for family,tolerance in TOLERANCES.items():
        value=diag.get("families",{}).get(family,{}).get(regime,{}).get("violation")
        if value is None or not np.isfinite(value) or value>tolerance:
            return False, f"{regime} {family} violation or missing value"
    return True, "independently sampled only; no continuous-time certificate"


def latest_duration(arm: Path, case: str, method: str, native: dict) -> float | None:
    if method in ("A10", "A40"):
        value=native.get(method,{}).get("duration_s")
        return float(value) if value is not None else None
    file=arm/case/(method.lower()+"_dense.npz")
    if not file.exists():
        return None
    with np.load(file) as data:
        t=np.asarray(data["time"],dtype=float)
    return float(t[-1]) if len(t)>1 and np.isfinite(t[-1]) and t[-1]>0 else None


def torque_ratio(arm: Path, case: str, method: str, profile: dict) -> float | None:
    file=arm/case/(method.lower()+"_pin_torque.npz")
    if not file.exists():
        return None
    with np.load(file) as data:
        tau=np.asarray(data["torque"])
    bound=np.stack((profile["effort_lower"],profile["effort_upper"]))
    return float(np.max(torque_utilization(tau,bound)))


def report(arm: Path, *, adaptive_receipts: Path | None = None) -> dict:
    tasks=json.loads((arm/"tasks.json").read_text())
    diag_path=arm/"independent_sampled.json"
    diagnostics=json.loads(diag_path.read_text()) if diag_path.exists() else {}
    native_path=arm/"pilot.jsonl"
    native={r["case_id"]:r["methods"] for r in
            (json.loads(line) for line in native_path.read_text().splitlines())} if native_path.exists() else {}
    receipts=json.loads(adaptive_receipts.read_text()) if adaptive_receipts and adaptive_receipts.exists() else {}
    strict_rows=[]; sample_rows=[]; utilization=[]; detailed=[]
    for task in tasks:
        case=task["case_id"];p=read_profile(arm/case/"torque_profile.json")
        row={"case_id":p["case_id"],"cohort":"pilot","population":task["population"],
             "source_case_id":case,"regime":p["regime"],
             "task_sha256":task["task_sha256"],"methods":{}}
        sample_row=dict(row,methods={})
        details={"case_id":case,"regime":p["regime"],"derate":p["derate"],
                 "expected_reference_utilization":p["expected_reference_utilization"],
                 "methods":{}}
        for method in METHODS:
            duration=latest_duration(arm,case,method,native.get(case,{}))
            diag=diagnostics.get(case+":"+method)
            phys,reason=sampled_acceptance(diag,p,scaled=False)
            scaled,scaled_reason=sampled_acceptance(diag,p,scaled=True)
            adaptive=receipts.get(case+":"+method)
            adaptive_phys=(isinstance(adaptive,dict) and adaptive.get("physical") is True
                           and adaptive.get("collision") is True
                           and adaptive.get("endpoint") is True
                           and adaptive.get("torque_profile_sha256")==p["profile_sha256"])
            adaptive_scaled=(adaptive_phys and adaptive.get("scaled95") is True)
            solver_info={}
            info=arm/case/(method.lower()+"_attempt.json")
            if info.exists():
                solver_info=json.loads(info.read_text())
            entry={"duration_s":duration,
                   "solver_converged":bool(solver_info.get("solver_converged",False)),
                   "control_bound_active":bool(solver_info.get("lower_control_bound_active",False)),
                   "verification":{"physical":bool(phys and adaptive_phys),
                                   "scaled95":bool(scaled and adaptive_scaled),
                                   "endpoint":bool(phys and adaptive_phys),
                                   "dense":bool(phys and adaptive_phys),
                                   "collision_independent":bool(phys and adaptive_phys),
                                   "dynamics_independent":bool(phys and adaptive_phys)},
                   "note":"Sampled diagnostics only unless a matching adaptive receipt exists"}
            sample_entry={**entry,"verification":{
                "physical":bool(phys),"scaled95":bool(scaled),
                "endpoint":bool(phys),"dense":bool(phys),
                "collision_independent":bool(phys),
                "dynamics_independent":bool(phys)}}
            row["methods"][method]=entry
            sample_row["methods"][method]=sample_entry
            use=torque_ratio(arm,case,method,p)
            details["methods"][method]={"duration_s":duration,
                                        "torque_utilization":use,
                                        "sampled_physical":phys,
                                        "sampled_scaled95":scaled,
                                        "adaptive_qualified":bool(adaptive_phys),
                                        "sampled_reason":reason,
                                        "scaled_reason":scaled_reason}
            if use is not None:
                utilization.append({"case_id":case,"regime":p["regime"],
                                    "method":method,"max_torque_utilization":use,
                                    "torque_ge_85pct":use>=.85,
                                    "torque_ge_95pct":use>=.95})
        strict_rows.append(row);sample_rows.append(sample_row);detailed.append(details)
    summary={"qualification":"adaptive receipts required; continuous-time certification not established",
             "source_case_count":len(tasks),
             "sampled":analyze(sample_rows),
             "adaptive_qualified":analyze(strict_rows),
             "torque_utilization":utilization,
             "case_details":detailed,
             "warning":"Do not pool regime variants sharing the same source trajectories."}
    (arm/"torque_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (arm/"torque_qualified.jsonl").write_text(
        "".join(json.dumps(x)+"\n" for x in strict_rows))
    (arm/"torque_sampled.jsonl").write_text(
        "".join(json.dumps(x)+"\n" for x in sample_rows))
    page=html_report(strict_rows,summary["adaptive_qualified"])
    extra=["<section><h2>Exploratory independently sampled diagnostics — NOT QUALIFIED</h2>",
           "<p>Only this section reports physical-time sampled results without the adaptive "
           "qualification receipt. All durations are local feasible incumbents, never global "
           "minimum-time guarantees.</p>"]
    for contrast,values in summary["sampled"]["groups"]["all"]["comparisons"].items():
        if contrast.split(":")[1] not in ("A10->A40","A40->B","B->D","B->C"):
            continue
        extra.append("<p>"+html.escape(contrast)+" — "+
                     str(values["eligible_pairs"])+"/"+str(len(tasks))+
                     " pairs; median "+str(values["median_improvement_pct"])+"% (sampled only)</p>")
    extra += ["<h2>Torque activity by source task</h2>",
              "<table><thead><tr><th>Case</th><th>Method</th>"
              "<th>Max utilization of derated torque limit</th></tr></thead><tbody>"]
    for item in utilization:
        extra.append("<tr><td>"+html.escape(item["case_id"])+"</td><td>"+item["method"]+
                     "</td><td>"+f"{100*item['max_torque_utilization']:.2f}%"+
                     "</td></tr>")
    extra.append("</tbody></table></section>")
    page=page.replace("</main></html>","\n".join(extra)+"</main></html>")
    (arm/"index.html").write_text(page,encoding="utf-8")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime",required=True,choices=("moderate","severe"))
    parser.add_argument("--output",type=Path,default=DEFAULT)
    parser.add_argument("--adaptive-receipts",type=Path)
    args=parser.parse_args()
    data=report(args.output/args.regime,adaptive_receipts=args.adaptive_receipts)
    print(args.regime,":",data["source_case_count"],"tasks; paired validation and HTML updated")


if __name__=="__main__":
    main()
