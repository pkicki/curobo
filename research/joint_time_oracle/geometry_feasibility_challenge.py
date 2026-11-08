# SPDX-License-Identifier: Apache-2.0
"""Freeze qualified counterexample candidates before running full native planning.

The plan contains ONLY independently qualified alternative witnesses with
numerically strong original-path torque-LP obstruction. Planner outcome
is never used for inclusion. CUDA --run uses the already vetted
torque_full_planner.execute with the full derated robot model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .geometry_feasibility_core import sha256_file
from .geometry_feasibility_discover import OUT


def select(root:Path,*,max_cases:int|None=None)->list[dict]:
    registration=json.loads((root/"discovery_registration.json").read_text())
    eligible=[]
    for scenario in registration["scenarios"]:
        name=scenario["scenario_id"];arm=root/name
        wfile=arm/"w_adaptive_receipts.json"
        lfile=arm/"refined_lp_summary.json"
        if not wfile.exists() or not lfile.exists():
            continue
        witnesses=json.loads(wfile.read_text())
        lps=json.loads(lfile.read_text())
        for task in json.loads((arm/"tasks.json").read_text()):
            case=task["case_id"]
            w=witnesses.get(case+":W",{})
            lp=lps.get(case,{})
            if (w.get("physical") is True and w.get("scaled95") is True
                and w.get("collision") is True and w.get("endpoint") is True
                and lp.get("status")=="NUMERICALLY_STRONG_ORIGINAL_PATH_INFEASIBILITY"):
                eligible.append({
                    "scenario":name,"case_id":case,
                    "task_sha256":task["task_sha256"],
                    "torque_profile_sha256":w.get("torque_profile_sha256"),
                    "witness_base_sha256":w.get("base_sha256"),
                    "witness_adaptive_sha256":w.get("adaptive_sha256"),
                    "original_lp_dual_lower_bound":lp.get(
                        "numerical_dual_certificate",{}).get("dual_lower_bound"),
                    "reference_witness_receipts_sha256":sha256_file(wfile),
                    "reference_lp_summary_sha256":sha256_file(lfile),
                })
    eligible.sort(key=lambda d:(d["case_id"],d["scenario"]))
    if max_cases is not None:
        if max_cases<1:raise ValueError("max_cases must be positive")
        eligible=eligible[:max_cases]
    return eligible


def plan(root:Path,*,max_cases:int|None=None)->dict:
    output=root/"native_challenge_plan.json"
    selected=select(root,max_cases=max_cases)
    config={"schema_version":1,"cases":selected,
            "search_budget_restarts":[1,4,16,32],
            "lanes_per_restart":4,"native_max_attempts":5,
            "warmup_iterations":2,"cuda_graph":False,
            "selection":"frozen exact-spline/Pinocchio Scaled95 witness + strong LP BEFORE native outcomes",
            "note":"discovery cohort is adversarial, not an unbiased prevalence estimate"}
    if output.exists():
        if json.loads(output.read_text())!=config:
            raise ValueError("native challenge previously frozen; do not revise selected tasks")
    else:
        output.write_text(json.dumps(config,indent=2)+"\n")
    return config


def run(root:Path,*,max_restarts:int,max_cases:int|None=None):
    from .torque_full_planner import execute
    config=plan(root,max_cases=max_cases)
    if max_restarts not in config["search_budget_restarts"]:
        raise ValueError("use only registered nested 1/4/16/32 restart budget")
    outcomes={}
    for entry in config["cases"]:
        arm=root/entry["scenario"]
        result=execute(arm,case_filter=entry["case_id"],
                       max_restarts=max_restarts,
                       lanes=config["lanes_per_restart"],
                       max_attempts=config["native_max_attempts"],
                       warmup=config["warmup_iterations"],
                       use_cuda_graph=config["cuda_graph"])
        outcomes[entry["scenario"]+"::"+entry["case_id"]]={
            "native_candidates":len(result.get(entry["case_id"],{})),
            "budget":max_restarts}
    return outcomes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUT)
    p.add_argument("--max-cases",type=int)
    p.add_argument("--restarts",type=int,default=1)
    p.add_argument("--run",action="store_true",
                   help="Actually run full native CUDA planning; default only freezes plan")
    a=p.parse_args()
    info=run(a.root,max_restarts=a.restarts,max_cases=a.max_cases) if a.run else plan(
        a.root,max_cases=a.max_cases)
    print(json.dumps(info,indent=2))


if __name__=="__main__":
    main()
