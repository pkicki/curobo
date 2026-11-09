# SPDX-License-Identifier: Apache-2.0
"""Execute one matched TimeOpt-guided or unguided global-search arm.

Requires an independently checked *same-payload* native/K107/Pinocchio
RNEA parity receipt. The script does not fabricate one. Each K107 call
is an actual GPU TOPP-RA+Newton solve through the pinned second repo.
Every response and candidate geometry is saved; the online screen is
not a physical qualification. Execute CPU collision checks separately
from the pinned CUDA retimer environment if mixing their PYTHONPATHs
causes test teardown errors; preserve any subprocess overhead in
reported complete-request latency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .cohort_geometry import QuinticGeometry
from .cohort_retime import ROOT
from .geometry_feasibility_core import sha256_file
from .geometry_feasibility_scene import IndependentGeometryAuditor
from .k107_oracle_bridge import prepare, run
from .timeopt_feedback_geometry import ARMS, BUDGET, search
from .torque_active import read_profile


def parity_verified(file:Path,task:dict,model_files:tuple[Path,...]):
    if not file.exists():
        raise ValueError("independent native/K107/Pinocchio RNEA parity receipt missing")
    data=json.loads(file.read_text())
    if data.get("status")!="PASS" or data.get("samples",0)<10 or not data.get("passed"):
        raise ValueError("unqualified model-alignment receipt")
    if data.get("task_sha256")!=task["task_sha256"] or data.get("payload")!=task.get("payload"):
        raise ValueError("model-alignment receipt refers to another task/payload")
    expected={str(p.resolve()):sha256_file(p) for p in model_files}
    if data.get("model_inputs_sha256")!=expected:
        raise ValueError("changed native/K107/Pinocchio model inputs")


def run_case(cohort:Path,output:Path,case_id:str,arm:str,
             *,retimer_repo:Path,retimer_python:str,retimer_sha:str,
             parity_file:Path)->dict:
    if arm not in ARMS:
        raise ValueError("unknown registered search arm")
    parent=cohort/"nominal_payload"
    tasks=json.loads((parent/"tasks.json").read_text())
    task=next((x for x in tasks if x["case_id"]==case_id),None)
    if task is None:
        raise ValueError("source case absent/not admitted in nominal-payload registry")
    profile_file=parent/case_id/"torque_profile.json"
    profile=read_profile(profile_file)
    if profile["source_task_sha256"]!=task["task_sha256"]:
        raise ValueError("source task/profile checksum mismatch")
    model_files=(
        ROOT/"curobo/content/assets/robot/franka_description/franka_panda.urdf",
        ROOT/"curobo/content/configs/robot/franka.yml",
        ROOT/"research/joint_time_oracle/results/native_reconstruction_20261106/baseline_validation_states.npz"
    )
    parity_verified(parity_file,task,model_files)
    auditor=IndependentGeometryAuditor(task["scene"],payload=task["payload"])
    destination=output/arm/case_id
    if (destination/"search_result.json").exists():
        raise ValueError("scored experiment immutable; choose a new output root")
    def cheap(controls):
        spline=QuinticGeometry(controls)
        samples=spline.evaluate(np.linspace(0,1,129))[0]
        check=auditor.check(samples,min_clearance_m=.005)
        return {k:v for k,v in check.items() if k!="gravity_torques"}
    def evaluator(controls,index):
        folder=destination/f"c{index+1:04d}"
        sourceid=hashlib.sha256(np.ascontiguousarray(controls).tobytes()).hexdigest()
        prepare(folder,task,profile_file,controls,
                source_geometry_id=f"{case_id}/candidate/{index+1}",
                source_geometry_sha256=sourceid,
                model_files=model_files,retimer_commit=retimer_sha)
        return run(folder,retimer_repo,retimer_python,retimer_sha,
                   iterations=16,grid=2049)
    result=search(task,arm=arm,evaluate=evaluator,cheap_screen=cheap)
    result.update({"torque_profile_sha256":profile["profile_sha256"],
                   "payload":task["payload"],
                   "independent_model_parity_receipt_sha256":sha256_file(parity_file),
                   "retimer_repository_commit_sha256":retimer_sha,
                   "retimer_interface":"real TOPPRA+Newton K107; C3 no torque rate",
                   "geometry_screen":"independent position/world/self collisions only; no LP/local gradient",
                   "qualification":"NOT QUALIFIED; downstream dense+adaptive independent Pinocchio is mandatory"})
    destination.mkdir(parents=True,exist_ok=True)
    (destination/"search_result.json").write_text(json.dumps(
        result,allow_nan=False,indent=2)+"\n")
    return {k:v for k,v in result.items() if k!="ledger"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--case",required=True)
    p.add_argument("--arm",required=True,choices=ARMS)
    p.add_argument("--retimer-repo",required=True,type=Path)
    p.add_argument("--retimer-python",required=True)
    p.add_argument("--retimer-commit",required=True)
    p.add_argument("--parity-receipt",required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(run_case(a.cohort,a.output,a.case,a.arm,
         retimer_repo=a.retimer_repo,retimer_python=a.retimer_python,
         retimer_sha=a.retimer_commit,parity_file=a.parity_receipt),indent=2))


if __name__=="__main__":main()
