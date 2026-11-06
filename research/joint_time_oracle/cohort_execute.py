# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute a fixed-budget B/C/D pilot using previously captured native trajectories."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PILOT=ROOT/'research/joint_time_oracle/results/cohort_pilot_isolated_20261106'
OUT=ROOT/'research/joint_time_oracle/results/cohort_joint_final_20261106'


def main() -> None:
    """Run all frozen tasks and record every subprocess, including failures."""
    OUT.mkdir(parents=True,exist_ok=True)
    git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
    if dirty:raise RuntimeError('freeze implementation in a clean commit before final pilot execution')
    config={'implementation_commit':git,'pilot_tasks_sha256':hashlib.sha256((PILOT/'tasks.json').read_bytes()).hexdigest(),
            'methods':{'B':{'grid':257,'maxiter':150},'C':{'controls':[10,14],'attempts_per_resolution':2,'maxiter':40},
                       'D':{'grid':257,'maxiter':150}},'per_process_timeout_s':300,
            'gpu':'NVIDIA GeForce RTX 3090','qualification':'exploratory sampled, not global or continuous certification',
            'native_compilation':'GPU kernels compiled during representative setup; no cold costs used as latency comparisons'}
    (OUT/'frozen_config.json').write_text(json.dumps(config,indent=2)+'\n')
    tasks=json.loads((PILOT/'tasks.json').read_text());results=[]
    for task in tasks:
        case=task['case_id'];dest=OUT/case;dest.mkdir(exist_ok=True);entry={'case_id':case,'stages':{}}
        for method in ('B','C','D'):
            if method=='D' and not (dest/'c_best.json').exists():
                entry['stages'][method]={'status':'not_run','reason':'no C candidate'};continue
            env=os.environ.copy();env['ORACLE_COHORT_RESULTS']=str(OUT);env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
            if method=='C':
                exe='/home/pkicki/anaconda3/envs/curobo-timeopt/bin/python';module='research.joint_time_oracle.cohort_joint'
                env['CUDA_VISIBLE_DEVICES']='1';env['PYTHONPATH']='/tmp/joint_oracle_compat:'+str(ROOT)
                cmd=[exe,'-m',module,case,'--maxiter','40']
            else:
                exe='/tmp/oracle-ipopt-env/bin/python';module='research.joint_time_oracle.cohort_retime'
                cmd=[exe,'-m',module,case,method,'--grid','257','--maxiter','150']
            start=time.perf_counter()
            try:
                proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=300)
                status={'returncode':proc.returncode,'wall_s':time.perf_counter()-start,
                        'status':'completed' if proc.returncode==0 else 'failed'};log=proc.stdout
            except subprocess.TimeoutExpired as exc:
                status={'status':'timeout','wall_s':time.perf_counter()-start};log=str(exc.stdout or '')
            (dest/(method.lower()+'_process.log')).write_text(log)
            entry['stages'][method]=status
            print(case,method,status,flush=True)
        results.append(entry)
        (OUT/'execution.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in results))


if __name__=='__main__':main()
