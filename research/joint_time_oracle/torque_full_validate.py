# SPDX-License-Identifier: Apache-2.0
"""Validate every saved full-planner restart with the existing independent Pinocchio harness.

Run this in the Pinocchio environment, not the CUDA planning environment.
The output is sampled evidence; adaptive qualification is a separate gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
STRESS=ROOT/"research/joint_time_oracle/results/torque_stress_final_20261107"


def prepare(arm:Path) -> tuple[Path,int]:
    source=arm/"native_full"
    tasks=json.loads((arm/"tasks.json").read_text())
    (source/"tasks.json").write_text(json.dumps(tasks,indent=2)+"\n")
    rows=[]
    all_restart_ids=set()
    for task in tasks:
        case=task["case_id"]
        info=source/case/"native_candidates.json"
        candidates=json.loads(info.read_text()) if info.exists() else {}
        methods={}
        for key,record in candidates.items():
            if record.get("status")=="SUCCESS":
                path=source/case/(key.lower()+"_dense.npz")
                if not path.exists():
                    raise ValueError("missing candidate archive " + str(path))
                methods[key]={"duration_s":record.get("duration_s"),
                              "verification":{}}
                all_restart_ids.add(key)
        rows.append({"case_id":case,"cohort":"pilot","population":task["population"],
                     "task_sha256":task["task_sha256"],"methods":methods})
    (source/"pilot.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    return source, (max(int(x[1:]) for x in all_restart_ids)+1) if all_restart_ids else 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=STRESS)
    parser.add_argument("--regime",required=True,choices=("moderate","severe"))
    parser.add_argument("--adaptive",action="store_true",help="audit Fxxx_adaptive_dense.npz")
    args=parser.parse_args()
    arm=args.root/args.regime
    source,count=prepare(arm)
    if not count:
        print("NO_CANDIDATES: no native full-planner trajectories to validate")
        return
    methods=",".join(f"F{k:03d}"+("_adaptive" if args.adaptive else "")
                     for k in range(count))
    command=[sys.executable,"-m","research.joint_time_oracle.cohort_validate",
             "--input",str(source),"--extra",str(source),"--methods",methods,
             "--torque-profile-dir",str(arm),
             "--report","independent_adaptive.json" if args.adaptive else "independent_sampled.json"]
    subprocess.run(command,cwd=ROOT,check=True)
    print(f"Validated full-planner candidate labels: {methods}")


if __name__=="__main__":
    main()
