# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan the frozen wider-excursion tasks and capture native TimeOpt inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from curobo._src.motion.motion_trajectory_timeopt import TrajectoryTimeOpt
from curobo._src.motion.motion_trajectory_timeopt_cfg import TrajectoryTimeOptCfg
from curobo.types import JointState
from .cohort_explore import planner_for, sample
from .curobo_adapter import CuroboOracleModel
from .run_study import read_plan

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/"research/joint_time_oracle/results/torque_wide_source_20261107"


def execute(source: Path, cases: list[str] | None=None) -> dict:
    """Run one native geometric plan and preserve exact A10/A40 spline inputs."""
    tasks=json.loads((source/"tasks.json").read_text())
    selected=set(cases or [])
    if selected:tasks=[t for t in tasks if t["case_id"] in selected]
    rows={r["case_id"]:r for r in (json.loads(line) for line in (source/"pilot.jsonl").read_text().splitlines())} if (source/"pilot.jsonl").exists() else {}
    for scene in sorted({task["scene"] for task in tasks},key=lambda x:str(x)):
        planner=planner_for(scene);planner.warmup(enable_graph=True,num_warmup_iterations=5)
        model=CuroboOracleModel(planner)
        for task in (t for t in tasks if t["scene"]==scene):
            case=task["case_id"];target=source/case;target.mkdir(parents=True,exist_ok=True)
            row={"case_id":case,"cohort":"confirm","population":task["population"],
                 "task_sha256":task["task_sha256"],"methods":{},"errors":[]}
            try:
                torch.manual_seed(int(task["seed"]));np.random.seed(int(task["seed"]))
                start=JointState.from_position(torch.tensor(task["start"],device=model.device,dtype=torch.float32)[None],joint_names=model.joint_names)
                goal=JointState.from_position(torch.tensor(task["goal"],device=model.device,dtype=torch.float32)[None],joint_names=model.joint_names)
                planner.reset_seed();torch.cuda.synchronize();stamp=time.perf_counter()
                result=planner.plan_cspace(goal,start,max_attempts=int(task["max_planner_attempts"]))
                torch.cuda.synchronize();row["planner_wall_s"]=time.perf_counter()-stamp
                if result is None or not bool(result.success.any()):
                    row["errors"].append({"stage":"geometry","reason":"planner_failed"})
                else:
                    base=read_plan(result,model.joint_names)
                    native=planner.trajopt_solver._result_processor.get_interpolation_input()
                    inputs={key:getattr(native,key).detach().cpu().numpy().copy() for key in ("knot","knot_dt","position")}
                    retimer=planner.trajopt_solver._timeopt.optimizer;count=retimer.batch_size
                    buffers={key:getattr(retimer,attr).detach().cpu().numpy().copy() for key,attr in
                             (("controls","_controls"),("start_boundary","_start_boundary"),("goal","_goal"),
                              ("timing","_timing"),("integrated_times","_times"),("success","_success"),("stationary","_stationary"))}
                    cfg=TrajectoryTimeOptCfg(robot=planner.kinematics.config,batch_size=count,
                        n_knots=inputs["knot"].shape[1],use_cuda_graph=False)
                    replay=TrajectoryTimeOpt(cfg);replay._setup_path(reference=True)
                    for key,attr in (("controls","_controls"),("start_boundary","_start_boundary"),("goal","_goal"),
                                     ("timing","_timing"),("integrated_times","_times"),("success","_success"),("stationary","_stationary")):
                        getattr(replay,attr).copy_(torch.as_tensor(buffers[key],device=model.device))
                    replay._has_result=True
                    duration_guess=(len(base["q"])-1)*base["dt"]
                    probe=torch.linspace(0,duration_guess,len(base["q"]),device=model.device).repeat(count,1).contiguous()
                    q_native=replay.sample(probe).position.detach().cpu().numpy().copy()
                    score=np.max(np.abs(q_native-base["q"][None]),axis=(1,2));lane=int(np.argmin(score))
                    if score[lane]>2e-5:raise RuntimeError(f"TimeOpt replay mismatch: {score[lane]}")
                    np.savez_compressed(target/"a10_interpolated.npz",**base)
                    np.savez_compressed(target/"native_input.npz",**inputs)
                    np.savez_compressed(target/"a10_buffers.npz",**buffers)
                    duration=float(buffers["integrated_times"][lane,-1])
                    np.savez_compressed(target/"a10_dense.npz",**sample(replay,lane,duration,8193))
                    row["methods"]["A10"]={"duration_s":duration,"native_success":bool(buffers["success"][lane]),
                        "solver_converged":False,"lane":lane,"lane_match_max_error_rad":float(score[lane]),"verification":{}}
                    cfg40=TrajectoryTimeOptCfg(robot=planner.kinematics.config,batch_size=count,
                        n_knots=inputs["knot"].shape[1],max_iterations=40,use_cuda_graph=False)
                    fast=TrajectoryTimeOpt(cfg40);tensor=lambda a:torch.as_tensor(a,device=model.device,dtype=torch.float32).contiguous()
                    torch.cuda.synchronize();stamp=time.perf_counter()
                    out=fast.timeopt(tensor(inputs["knot"]),tensor(buffers["start_boundary"][:,0]),
                                     tensor(buffers["goal"]),tensor(inputs["knot_dt"]))
                    torch.cuda.synchronize();elapsed=time.perf_counter()-stamp
                    error=max(float(np.max(np.abs(getattr(fast,attr).detach().cpu().numpy()-buffers[key])))
                              for key,attr in (("controls","_controls"),("start_boundary","_start_boundary"),("goal","_goal")))
                    if error>1e-6:raise RuntimeError(f"A40 geometry changed: {error}")
                    duration40=float(out.duration[lane].item());success=bool(out.success[lane].item())
                    np.savez_compressed(target/"a40_buffers.npz",timing=out.timing_controls.detach().cpu().numpy().copy(),
                        times=out.time.detach().cpu().numpy().copy(),success=out.success.detach().cpu().numpy().copy(),status=out.status.detach().cpu().numpy().copy())
                    row["methods"]["A40"]={"duration_s":duration40 if success else None,"native_success":success,
                        "native_status":int(out.status[lane].item()),"solver_converged":False,"solver_wall_s":elapsed,
                        "geometry_error_rad":error,"verification":{}}
                    if success:np.savez_compressed(target/"a40_dense.npz",**sample(fast,lane,duration40,8193))
                    else:row["errors"].append({"stage":"A40","reason":"native_retimer_failed"})
            except Exception as exc:
                row["errors"].append({"stage":"execution","exception":repr(exc),"traceback":traceback.format_exc()})
            rows[case]=row;(source/"pilot.jsonl").write_text("".join(json.dumps(rows[k])+"\n" for k in sorted(rows)))
            print(case,row["methods"],row["errors"],flush=True)
        planner.destroy()
    manifest={"task_inventory_sha256":hashlib.sha256((source/"tasks.json").read_bytes()).hexdigest(),
              "method":"native cuRoboV2-Fast C-space planner; fixed geometry; A10/A40 TimeOpt",
              "hardware":torch.cuda.get_device_name(0),"attempted":len(tasks),
              "planner_success":sum(bool(r.get("methods",{}).get("A10",{}).get("native_success")) for r in rows.values())}
    (source/"planning_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--source",type=Path,default=DEFAULT);parser.add_argument("--cases")
    args=parser.parse_args();print(json.dumps(execute(args.source,args.cases.split(",") if args.cases else None),indent=2))


if __name__=="__main__":main()
