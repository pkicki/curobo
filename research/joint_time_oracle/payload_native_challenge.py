# SPDX-License-Identifier: Apache-2.0
"""Finite-budget *nominal-effort* cuRoboV2-Fast payload challenges.

Each restart builds a fresh private native model with attached payload
inertia, NEVER changes actuator effort, and warms outside the request
timer. Independent Pinocchio/RNEA parity must be separately qualified
before interpreting source outcomes. Returned successful paths still
require full independent physical+payload collision qualification.

Native failure is BUDGET EXHAUSTED, not global geometric infeasibility.
A positive fixed-path LP bound excludes only that exact geometry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from curobo.motion_planner import MotionPlanner,MotionPlannerCfg
from curobo.types import JointState

from .payload_model_parity import native_payload_robot
from .payload_timeopt_contract import (
    OUT,Payload,RESTART_BUDGETS,exact_effort_contract,immutable_json,source_sha,
)
from .run_study import read_plan


def native_attempt(task:dict,*,restart:int,lanes:int=4,
                   max_attempts:int=5,warmup:int=2,
                   capture:bool=False,folder:Path|None=None)->dict:
    payload=Payload(**task["payload"])
    robot,installed=native_payload_robot(payload)
    seed=int(task["seed"]+291337*restart)
    torch.manual_seed(seed);np.random.seed(seed%(2**32))
    t0=time.perf_counter()
    planner=None
    record={"schema":1,"case_id":task["case_id"],"restart":restart,
            "seed":seed,"status":"ERROR",
            "payload":task["payload"],
            "payload_collision_qualified":False,
            "native_mass_injection":installed,
            "original_actuator_effort":True}
    try:
        cfg=MotionPlannerCfg.create(
            robot=robot,scene_model=task["scene"],graph_planner_algorithm="rrt",
            enable_timeopt=True,use_cuda_graph=False,random_seed=seed,
            num_trajopt_seeds=lanes,max_batch_size=1)
        planner=MotionPlanner(cfg)
        native_effort=(planner.trajopt_solver.kinematics.config
                       .kinematics_config.joint_limits.effort.detach().cpu().numpy())
        expected=np.asarray(installed["effort_parity"]["nominal_signed_effort"])
        exact_effort_contract(native_effort,expected)
        planner.warmup(enable_graph=False,num_warmup_iterations=warmup)
        torch.cuda.synchronize()
        record["setup_and_warmup_wall_s"]=time.perf_counter()-t0
        device=planner.default_joint_state.position.device
        names=[f"panda_joint{i}" for i in range(1,8)]
        start=JointState.from_position(
            torch.as_tensor(task["start"],device=device,dtype=torch.float32)[None],
            joint_names=names)
        goal=JointState.from_position(
            torch.as_tensor(task["goal"],device=device,dtype=torch.float32)[None],
            joint_names=names)
        planner.reset_seed()
        torch.cuda.synchronize()
        t=time.perf_counter()
        if capture:
            if folder is None:raise ValueError("capture requires output folder")
            from .native_pretimeopt_probe import NativePreTimeOptProbe
            with NativePreTimeOptProbe(planner,folder,f"F{restart:03d}",
                                       trace_phases=True):
                result=planner.plan_cspace(goal,start,max_attempts=max_attempts)
        else:
            result=planner.plan_cspace(goal,start,max_attempts=max_attempts)
        torch.cuda.synchronize()
        record["request_wall_s"]=time.perf_counter()-t
        record["planner_attempts"]=int(planner.last_planning_attempts)
        if result is None or not bool(result.success.any()):
            record["status"]="BUDGET_ATTEMPT_FAILED"
            return record
        interpolated=read_plan(result,names)
        if interpolated is None:
            record["status"]="NATIVE_SUCCESS_NO_INDEPENDENT_PLAN"
            return record
        # Record unqualified native plan: do not invent a physical pass.
        if folder is not None:
            file=folder/f"f{restart:03d}_native_success.npz"
            if file.exists():raise ValueError("native successful plan already frozen")
            np.savez_compressed(file,**interpolated)
            record["native_plan_sha256"]=source_sha(file)
        record["status"]="NATIVE_SUCCESS_UNQUALIFIED"
        record["native_duration_s"]=float(interpolated["time"][-1])
        return record
    except Exception as exc:
        record["status"]="ERROR"
        record["reason"]=repr(exc)
        record["traceback"]=traceback.format_exc(limit=8)
        return record
    finally:
        if planner is not None:
            planner.destroy()


def run(task:dict,*,root:Path,budget:int=32,lanes:int=4,
        max_attempts:int=5,capture:bool=False)->dict:
    if budget not in RESTART_BUDGETS:
        raise ValueError("only registered restart budgets allowed")
    if task.get("joint_limit_scale")!=1.0 or not task.get("nominal_actuator_limits"):
        raise ValueError("not an original actuator-effort source task")
    if task.get("admission")!="ENDPOINT_ADMITTED":
        return {"case_id":task["case_id"],"status":"ENDPOINT_GRAVITY_REJECTED",
                "restarts":0}
    folder=root/task["case_id"]
    folder.mkdir(parents=True,exist_ok=True)
    input_file=folder/"task.json"
    immutable_json(input_file,task)
    records_file=folder/"native_attempts.json"
    records=json.loads(records_file.read_text()) if records_file.exists() else []
    if len(records)>budget:
        raise ValueError("existing inventory exceeds requested restart budget")
    for restart in range(len(records),budget):
        row=native_attempt(task,restart=restart,lanes=lanes,
                           max_attempts=max_attempts,
                           capture=capture,folder=folder)
        records.append(row)
        records_file.write_text(json.dumps(records,indent=2,allow_nan=False)+"\n")
    statuses=[x["status"] for x in records]
    status=("NATIVE_SUCCESS_UNQUALIFIED" if "NATIVE_SUCCESS_UNQUALIFIED" in statuses else
            "NATIVE_ERRORS_PRESENT" if any(x=="ERROR" for x in statuses) else
            "BUDGET_EXHAUSTED")
    summary={"schema":1,"case_id":task["case_id"],"status":status,
             "completed_restarts":len(records),"registered_restart_budget":budget,
             "nominal_actuator_effort":True,"payload":task["payload"],
             "payload_collision_qualified":False,
             "attempt_inventory_sha256":source_sha(records_file),
             "interpretation":"native finite budget and unqualified trajectories only; do not infer global infeasibility",
             "native_success_count":sum(x=="NATIVE_SUCCESS_UNQUALIFIED" for x in statuses),
             "native_error_count":sum(x=="ERROR" for x in statuses)}
    # Budget stages may increase; only stage-root one immutable summary.
    immutable_json(folder/f"summary_r{budget:02d}.json",summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",type=Path,required=True)
    p.add_argument("--out",type=Path,default=OUT/"native")
    p.add_argument("--restarts",type=int,default=32,choices=RESTART_BUDGETS)
    p.add_argument("--lanes",type=int,default=4)
    p.add_argument("--max-attempts",type=int,default=5)
    p.add_argument("--capture",action="store_true")
    args=p.parse_args()
    task=json.loads(args.task.read_text())
    print(json.dumps(run(task,root=args.out,budget=args.restarts,
                         lanes=args.lanes,max_attempts=args.max_attempts,
                         capture=args.capture),indent=2))


if __name__=="__main__":
    main()
