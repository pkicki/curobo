# SPDX-License-Identifier: Apache-2.0
"""FULL native cuRobo C-space planning under task-specific derated effort limits.

Unlike A10/A40, effort limits are injected into RobotCfg BEFORE MotionPlanner
construction. Every independent restart re-runs geometric TrajOpt/graph seeds
and native TimeOpt. All candidate trajectories are retained; no duration
comparison is qualified until independent Pinocchio/collision audits run.

This pilot is intentionally offline. Rebuilding planners and warmups are
separately recorded and not an online latency benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState
from curobo._src.types.robot import RobotCfg
from curobo._src.util.config_io import resolve_config, join_path
from curobo.content import get_robot_configs_path
from curobo._src.motion.motion_trajectory_timeopt import TrajectoryTimeOpt
from curobo._src.motion.motion_trajectory_timeopt_cfg import TrajectoryTimeOptCfg

from .run_study import read_plan
from .torque_active import read_profile

ROOT=Path(__file__).resolve().parents[2]
STRESS=ROOT/"research/joint_time_oracle/results/torque_stress_final_20261107"
BUDGETS=(1,4,16,32)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def robot_with_derated_effort(profile: dict) -> RobotCfg:
    """Do not mutate the global robot config or reuse a planner across profiles."""
    robot_data=resolve_config(join_path(get_robot_configs_path(),"franka.yml"))
    robot=RobotCfg.create(robot_data)
    bounds=robot.kinematics.kinematics_config.joint_limits
    nominal=np.stack((profile["nominal_effort_lower"],profile["nominal_effort_upper"]))
    actual=bounds.effort.detach().cpu().numpy()
    if actual.shape!=nominal.shape or not np.allclose(actual,nominal,rtol=1e-5,atol=1e-5):
        raise ValueError("robot effort bounds do not match frozen nominal profile")
    modified=np.stack((profile["effort_lower"],profile["effort_upper"]))
    if modified.shape!=actual.shape or not np.all(np.isfinite(modified)):
        raise ValueError("invalid derated effort shape")
    bounds.effort.copy_(torch.as_tensor(modified,device=bounds.effort.device,
                                        dtype=bounds.effort.dtype))
    if not np.allclose(bounds.effort.detach().cpu().numpy(),modified,atol=1e-6):
        raise RuntimeError("modified effort limits were not applied")
    return robot


def make_planner(profile: dict, scene: str | None, seed: int,
                 *, lanes: int=4, use_cuda_graph: bool=False) -> MotionPlanner:
    robot=robot_with_derated_effort(profile)
    cfg=MotionPlannerCfg.create(
        robot=robot,scene_model=scene,graph_planner_algorithm="rrt",
        enable_timeopt=True,use_cuda_graph=use_cuda_graph,random_seed=seed,
        num_trajopt_seeds=lanes,max_batch_size=1)
    planner=MotionPlanner(cfg)
    # Verify that BOTH geometric planning and TimeOpt received this configuration.
    target=np.stack((profile["effort_lower"],profile["effort_upper"]))
    active=planner.trajopt_solver.kinematics.config.kinematics_config.joint_limits.effort
    if not np.allclose(active.detach().cpu().numpy(),target,atol=1e-6):
        planner.destroy()
        raise RuntimeError("TrajOpt model did not inherit derated limits")
    return planner


def captured_candidate(planner: MotionPlanner, result, task: dict,
                       profile: dict) -> tuple[dict,dict,dict]:
    """Exact selected native TimeOpt replay; no interpolation of sparse archives."""
    names=[f"panda_joint{i}" for i in range(1,8)]
    interpolation=read_plan(result,names)
    native=planner.trajopt_solver._result_processor.get_interpolation_input()
    inputs={k:getattr(native,k).detach().cpu().numpy().copy() for k in ("knot","knot_dt","position")}
    retimer=planner.trajopt_solver._timeopt.optimizer
    buffers={key:getattr(retimer,attr).detach().cpu().numpy().copy() for key,attr in
             (("controls","_controls"),("start_boundary","_start_boundary"),("goal","_goal"),
              ("timing","_timing"),("integrated_times","_times"),
              ("success","_success"),("stationary","_stationary"))}
    count=int(retimer.batch_size)
    cfg=TrajectoryTimeOptCfg(
        robot=planner.kinematics.config,batch_size=count,
        n_knots=inputs["knot"].shape[1],use_cuda_graph=False,
        torque_lower=torch.as_tensor(profile["effort_lower"],device=planner.default_joint_state.position.device),
        torque_upper=torch.as_tensor(profile["effort_upper"],device=planner.default_joint_state.position.device),
        limit_scale=.95,jerk_limit_scale=.95)
    replay=TrajectoryTimeOpt(cfg)
    replay._setup_path(reference=True)
    for key,attr in (("controls","_controls"),("start_boundary","_start_boundary"),
                     ("goal","_goal"),("timing","_timing"),
                     ("integrated_times","_times"),("success","_success"),
                     ("stationary","_stationary")):
        target=getattr(replay,attr)
        target.copy_(torch.as_tensor(buffers[key],device=target.device,dtype=target.dtype))
    replay._has_result=True
    guess=(len(interpolation["q"])-1)*interpolation["dt"]
    times=torch.linspace(0,guess,len(interpolation["q"]),
                         device=planner.default_joint_state.position.device).repeat(count,1).contiguous()
    positions=replay.sample(times).position.detach().cpu().numpy().copy()
    error=np.max(np.abs(positions-interpolation["q"][None]),axis=(1,2))
    lane=int(np.argmin(error))
    if float(error[lane])>2e-5:
        raise ValueError(f"native selected-lane mismatch: {float(error[lane])}")
    # The independent replay is reused only after clearing its sampled-shape graph.
    replay.reset_cuda_graph()
    duration=float(buffers["integrated_times"][lane,-1])
    if not np.isfinite(duration) or duration<=0 or not bool(buffers["success"][lane]):
        raise ValueError("selected retimed lane is invalid")
    # Replay every internally valid native candidate, not just the selected
    # lane. A lane must pass the native collision/path/dynamic validation masks;
    # the independent Pinocchio/sphere audit remains a separate hard gate.
    validity=getattr(planner.trajopt_solver._timeopt,"validation_masks",None)
    mask_available=isinstance(validity,dict) and all(
        key in validity for key in ("path_valid","collision_valid","limits_and_dynamics_valid"))
    accepted=np.asarray(buffers["success"],dtype=bool).reshape(-1).copy()
    if mask_available:
        for key in ("path_valid","collision_valid","limits_and_dynamics_valid"):
            v=validity[key].detach().cpu().numpy().reshape(-1)
            if len(v)!=count:
                raise ValueError("native seed mask shape mismatch")
            accepted &= v.astype(bool)
    else:
        accepted[:]=False
        accepted[lane]=True
    durations=np.asarray(buffers["integrated_times"][:,-1],dtype=float)
    stable=np.where(np.isfinite(durations)&(durations>0),durations,duration)
    unit=torch.linspace(0,1,8193,device=planner.default_joint_state.position.device)
    time_grid=(torch.as_tensor(stable,device=unit.device,dtype=unit.dtype)[:,None]*unit[None,:]).contiguous()
    state=replay.sample(time_grid)
    times=time_grid.detach().cpu().numpy()
    def lane_states(idx):
        return dict(time=times[idx].copy(),
                    q=state.position[idx].detach().cpu().numpy().copy(),
                    qd=state.velocity[idx].detach().cpu().numpy().copy(),
                    qdd=state.acceleration[idx].detach().cpu().numpy().copy(),
                    jerk=state.jerk[idx].detach().cpu().numpy().copy())
    dense=lane_states(lane)
    alternates=[]
    for internal in range(count):
        if internal==lane or not accepted[internal] or not np.isfinite(durations[internal]) or durations[internal]<=0:
            continue
        alternates.append({"lane":internal,"duration_s":float(durations[internal]),
                           "trajectory":lane_states(internal),
                           "native_validation_masks":{key:bool(validity[key].reshape(-1)[internal].item())
                                                      for key in ("path_valid","collision_valid","limits_and_dynamics_valid")}})
    metadata={"duration_s":duration,"lane":lane,
              "alternate_accepted_lanes":len(alternates),
              "native_mask_available":mask_available,
              "lane_position_replay_error_rad":float(error[lane]),
              "native_success":bool(buffers["success"][lane]),
              "num_native_lanes":count,
              "effective_effort_lower":profile["effort_lower"],
              "effective_effort_upper":profile["effort_upper"],
              "task_sha256":task["task_sha256"],
              "torque_profile_sha256":profile["profile_sha256"]}
    return dense,metadata,{"native":inputs,"retimer":buffers,"alternates":alternates}


def execute(arm: Path, *, case_filter: str | None, max_restarts: int,
            lanes: int, max_attempts: int, warmup: int,
            use_cuda_graph: bool=False) -> dict:
    if not (1<=max_restarts<=32 and lanes>=1 and max_attempts>=1):
        raise ValueError("invalid native full-planner budget")
    tasks=json.loads((arm/"tasks.json").read_text())
    if case_filter:
        tasks=[t for t in tasks if t["case_id"]==case_filter]
        if not tasks:
            raise ValueError("case not in admitted source inventory")
    out=arm/"native_full"
    out.mkdir(parents=True,exist_ok=True)
    source_hash=sha(arm/"tasks.json")
    contract={"schema":1,"source_tasks_sha256":source_hash,
              "maximum_restarts_allowed":32,"lanes_per_restart":lanes,
              "max_planner_attempts":max_attempts,"warmup_iterations":warmup,
              "use_cuda_graph":use_cuda_graph,
              "role":"full native replanning under frozen derated effort limits",
              "budget_definition":"independent planner restarts, not solver Newton iterations",
              "num_requested_lanes_per_restart":lanes,
              "notes":"All candidate trajectories require independent physical validation."}
    manifest=out/"configuration.json"
    if manifest.exists() and json.loads(manifest.read_text())!=contract:
        raise ValueError("frozen native full-planner configuration mismatch")
    manifest.write_text(json.dumps(contract,indent=2)+"\n")
    # Copy only frozen task metadata; do not overwrite existing source artifacts.
    (out/"tasks.json").write_text((arm/"tasks.json").read_text())
    inventory={}
    for task in tasks:
        case=task["case_id"]
        folder=out/case
        folder.mkdir(parents=True,exist_ok=True)
        profile=read_profile(arm/case/"torque_profile.json")
        if profile["source_case_id"]!=case:
            raise ValueError("profile/source mismatch")
        frozen=folder/"native_candidates.json"
        rows=json.loads(frozen.read_text()) if frozen.exists() else {}
        for restart in range(max_restarts):
            tag=f"F{restart:03d}"
            if tag in rows and rows[tag].get("status") in ("SUCCESS","FAILED","ERROR"):
                continue
            seed=int(task["seed"]+291337*restart)
            np.random.seed(seed%(2**32))
            torch.manual_seed(seed)
            planner=None
            stamp=time.perf_counter()
            record={"name":tag,"restart":restart,"seed":seed,
                    "profile_sha256":profile["profile_sha256"],
                    "status":"ERROR"}
            try:
                planner=make_planner(profile,task["scene"],seed,lanes=lanes,
                                     use_cuda_graph=use_cuda_graph)
                planner.warmup(enable_graph=use_cuda_graph,
                               num_warmup_iterations=warmup)
                torch.cuda.synchronize()
                record["setup_and_warmup_wall_s"]=time.perf_counter()-stamp
                device=planner.default_joint_state.position.device
                names=[f"panda_joint{i}" for i in range(1,8)]
                start=JointState.from_position(
                    torch.tensor(task["start"],device=device,dtype=torch.float32)[None],
                    joint_names=names)
                goal=JointState.from_position(
                    torch.tensor(task["goal"],device=device,dtype=torch.float32)[None],
                    joint_names=names)
                planner.reset_seed()
                torch.cuda.synchronize()
                run_stamp=time.perf_counter()
                result=planner.plan_cspace(goal,start,max_attempts=max_attempts)
                torch.cuda.synchronize()
                record["plan_wall_s"]=time.perf_counter()-run_stamp
                record["planning_attempts"]=planner.last_planning_attempts
                if result is None or not bool(result.success.any()):
                    record["status"]="FAILED"
                    record["reason"]="native full planner returned no successful trajectory"
                else:
                    dense,metadata,buffers=captured_candidate(planner,result,task,profile)
                    np.savez_compressed(folder/(tag.lower()+"_dense.npz"),**dense)
                    np.savez_compressed(folder/(tag.lower()+"_native_input.npz"),**buffers["native"])
                    np.savez_compressed(folder/(tag.lower()+"_buffers.npz"),**buffers["retimer"])
                    record.update(metadata)
                    record["status"]="SUCCESS"
                    record["dense_sha256"]=sha(folder/(tag.lower()+"_dense.npz"))
                    record["trajectory_source"]=getattr(result,"trajectory_source",None)
                    for extra in buffers["alternates"]:
                        other_tag=f"{tag}L{extra['lane']:02d}"
                        other_path=folder/(other_tag.lower()+"_dense.npz")
                        np.savez_compressed(other_path,**extra["trajectory"])
                        rows[other_tag]={
                            "name":other_tag,"restart":restart,"seed":seed,
                            "status":"SUCCESS","internal_candidate":True,
                            "lane":extra["lane"],"duration_s":extra["duration_s"],
                            "torque_profile_sha256":profile["profile_sha256"],
                            "task_sha256":task["task_sha256"],
                            "dense_sha256":sha(other_path),
                            "native_validation_masks":extra["native_validation_masks"],
                            "parent_restart_tag":tag}

            except Exception as exc:
                record["reason"]=repr(exc)
                record["traceback"]=traceback.format_exc()
            finally:
                if planner is not None:
                    planner.destroy()
            rows[tag]=record
            frozen.write_text(json.dumps(rows,indent=2,default=str)+"\n")
            print(case,tag,record["status"],record.get("duration_s"),flush=True)
        inventory[case]=rows
    return inventory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=STRESS)
    parser.add_argument("--regime",required=True,choices=("moderate","severe"))
    parser.add_argument("--case")
    parser.add_argument("--max-restarts",type=int,default=4)
    parser.add_argument("--lanes",type=int,default=4)
    parser.add_argument("--max-attempts",type=int,default=5)
    parser.add_argument("--warmup",type=int,default=2)
    parser.add_argument("--use-cuda-graph",action="store_true")
    args=parser.parse_args()
    data=execute(args.root/args.regime,case_filter=args.case,
                 max_restarts=args.max_restarts,lanes=args.lanes,
                 max_attempts=args.max_attempts,warmup=args.warmup,
                 use_cuda_graph=args.use_cuda_graph)
    print(json.dumps({"cases":len(data),
                      "successful_restarts":sum(v.get("status")=="SUCCESS"
                          for rows in data.values() for v in rows.values())},indent=2))


if __name__=="__main__":
    main()
