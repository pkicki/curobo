# SPDX-License-Identifier: Apache-2.0
"""Run frozen torque-stress cases, keeping every failure and exact source IDs.

Requires two compatible Python environments: a CUDA cuRobo environment and a
Pinocchio/cyipopt environment. Do not use for latency benchmarking: process
launch and verification are included in wall-clock times.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .torque_active import read_profile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "research/joint_time_oracle/results/torque_stress_20261107"


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_command(command: list[str], env: dict, timeout: int, log: Path) -> dict:
    stamp=time.perf_counter()
    try:
        proc=subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,text=True,timeout=timeout)
        output=proc.stdout
        result={"status":"completed" if proc.returncode==0 else "failed",
                "returncode":proc.returncode}
    except subprocess.TimeoutExpired as exc:
        output=str(exc.stdout or b"")
        result={"status":"timeout","returncode":None}
    log.write_text(output)
    result.update(wall_s=time.perf_counter()-stamp,
                  command=command,log=log.name)
    return result


def execute(args):
    root=args.output; arm=root/args.regime
    tasks=json.loads((arm/"tasks.json").read_text())
    if args.case:
        tasks=[t for t in tasks if t["case_id"]==args.case]
        if not tasks:
            raise ValueError("case is not admitted in frozen profile")
    if args.max_cases:
        tasks=tasks[:args.max_cases]
    git=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    dirty=subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,text=True)
    if dirty.strip():
        raise RuntimeError("commit changes and use a clean tracked worktree before execution")
    existing={}
    report=arm/"torque_execution.jsonl"
    if report.exists():
        existing={row["case_id"]:row for row in
                  (json.loads(line) for line in report.read_text().splitlines() if line.strip())}
    manifest={"implementation_commit":git,"regime":args.regime,
              "task_inventory_sha256":checksum(arm/"tasks.json"),
              "profile_inventory_sha256":checksum(arm/"all_profiles.json"),
              "native_geometry_source":"cohort_pilot_isolated_20261106",
              "torque_model":"same Panda model; explicit derated asymmetric torque limits",
              "limits":"0.95 optimizer scale; physical independent audit required",
              "C":{"resolutions":args.resolutions,"attempts":args.attempts,
                   "trust_radius":args.trust_radius,"seed_noise":args.seed_noise,
                   "maxiter":args.c_maxiter},
              "B_D":{"grid":args.grid,"maxiter":args.ipopt_maxiter},
              "wall_clock_not_latency":True}
    if (arm/"execution_manifest.json").exists():
        prior=json.loads((arm/"execution_manifest.json").read_text())
        if prior!=manifest:
            raise ValueError("execution manifest changed; use a NEW output directory")
    else:
        (arm/"execution_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    for task in tasks:
        case=task["case_id"]; dest=arm/case
        profile=read_profile(dest/"torque_profile.json")
        if profile["source_case_id"]!=case or profile["regime"]!=args.regime:
            raise ValueError("task/profile mismatch")
        row=existing.get(case,{"case_id":case,"regime":args.regime,
                              "source_task_sha256":task["task_sha256"],
                              "profile_sha256":profile["profile_sha256"],
                              "stages":{}})
        env=os.environ.copy()
        env.update({"ORACLE_COHORT_RESULTS":str(arm),
                    "ORACLE_TORQUE_PROFILE":str(dest/"torque_profile.json"),
                    "OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1"})
        if args.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"]=args.cuda_visible_devices
        if args.shim_dir:
            env["PYTHONPATH"]=str(args.shim_dir)+os.pathsep+str(ROOT)
        stages=(
            ("native",[args.cuda_python,"-m","research.joint_time_oracle.cohort_torque_native",
                      "--regime",args.regime,"--output",str(root),"--case",case]),
            ("B",[args.pin_python,"-m","research.joint_time_oracle.cohort_retime",
                  case,"B","--grid",str(args.grid),"--maxiter",str(args.ipopt_maxiter)]),
            ("C",[args.cuda_python,"-m","research.joint_time_oracle.cohort_joint",
                  case,"--maxiter",str(args.c_maxiter),"--resolutions",args.resolutions,
                  "--attempts",str(args.attempts),"--trust-radius",str(args.trust_radius),
                  "--seed-noise",str(args.seed_noise)]),
            ("D",[args.pin_python,"-m","research.joint_time_oracle.cohort_retime",
                  case,"D","--grid",str(args.grid),"--maxiter",str(args.ipopt_maxiter)]),
        )
        for name, cmd in stages:
            if args.resume and row["stages"].get(name,{}).get("status")=="completed":
                continue
            if name=="D" and not (dest/"c_best.json").exists():
                row["stages"]["D"]={"status":"not_run","reason":"C produced no valid geometry"}
                continue
            row["stages"][name]=run_command(
                cmd,env,args.timeout,dest/(name.lower()+"_torque_process.log"))
            existing[case]=row
            report.write_text("".join(json.dumps(existing[k])+"\n" for k in sorted(existing)))
            print(args.regime,case,name,row["stages"][name]["status"],flush=True)
    return existing


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime",choices=("moderate","severe"),required=True)
    parser.add_argument("--output",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--cuda-python",default=sys.executable)
    parser.add_argument("--pin-python",default=sys.executable)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--shim-dir",type=Path)
    parser.add_argument("--case")
    parser.add_argument("--max-cases",type=int)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--timeout",type=int,default=1800)
    parser.add_argument("--grid",type=int,default=513)
    parser.add_argument("--ipopt-maxiter",type=int,default=500)
    parser.add_argument("--c-maxiter",type=int,default=300)
    parser.add_argument("--resolutions",default="14,22,34")
    parser.add_argument("--attempts",type=int,default=4)
    parser.add_argument("--trust-radius",type=float,default=1.5)
    parser.add_argument("--seed-noise",type=float,default=0.2)
    execute(parser.parse_args())


if __name__=="__main__":
    main()
