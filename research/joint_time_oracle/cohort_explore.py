# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze and explore native A10/A40 on a deterministic, explicitly unqualified pilot.

This exploratory runner does not implement B, C, D, or independent qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState
from curobo._src.motion.motion_trajectory_timeopt import TrajectoryTimeOpt
from curobo._src.motion.motion_trajectory_timeopt_cfg import TrajectoryTimeOptCfg
from .curobo_adapter import CuroboOracleModel
from .run_study import as_numpy, read_plan

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research/joint_time_oracle/results/cohort_pilot_exploratory_20261106'
SCENES = [('free', None, 6), ('table', 'collision_test.yml', 8),
          ('cluttered', 'collision_primitives_3d.yml', 6)]


def digest(value: object) -> str:
    """Hash canonical JSON task definitions."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def planner_for(scene: str | None) -> MotionPlanner:
    """Construct a reusable Franka planner for one frozen scene."""
    return MotionPlanner(MotionPlannerCfg.create(robot='franka.yml', scene_model=scene,
                         graph_planner_algorithm='rrt', enable_timeopt=True, use_cuda_graph=True))


def prepare() -> None:
    """Generate task endpoints, logging all rejected endpoint proposals."""
    OUT.mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(771214)
    tasks=[]; proposals=[]
    for population, scene, count in SCENES:
        planner=planner_for(scene); model=CuroboOracleModel(planner)
        limits=model.limits(); home=as_numpy(planner.default_joint_state.position)
        lo, hi=limits.position
        for index in range(count):
            for attempt in range(1000):
                start=np.clip(home+rng.uniform(-.32,.32,7),lo+.12,hi-.12)
                goal=np.clip(home+rng.uniform(-.65,.65,7),lo+.12,hi-.12)
                q=torch.as_tensor(np.stack([start,goal]),device=model.device,dtype=torch.float32)
                with torch.no_grad():
                    costs=[float(c.detach().max().item()) for c in model.collision_components(q)]
                accepted=all(np.isfinite(costs)) and max(costs,default=0)<=1e-7
                proposals.append({'population':population,'index':index,'proposal':attempt,
                                  'start':start.tolist(),'goal':goal.tolist(),'collision_costs':costs,
                                  'accepted':accepted})
                if accepted:
                    task={'case_id':f'pilot_{population}_{index:02d}','cohort':'pilot',
                          'population':population,'robot':'franka.yml','scene':scene,
                          'start':start.tolist(),'goal':goal.tolist(), 'joint_names':model.joint_names,
                          'initial_velocity':[0.]*7,'initial_acceleration':[0.]*7,
                          'terminal_velocity':[0.]*7,'terminal_acceleration':[0.]*7,
                          'payload':None,'dynamic_limit_scale':.95,'seed':771214+len(tasks),
                          'max_planner_attempts':5, 'endpoint_native_costs':costs}
                    task['task_sha256']=digest(task);tasks.append(task);break
            else:
                raise RuntimeError(f'could not find collision-free endpoint for {population}/{index}')
        planner.destroy()
    (OUT/'tasks.json').write_text(json.dumps(tasks,indent=2)+'\n')
    (OUT/'endpoint_proposals.json').write_text(json.dumps(proposals,indent=2)+'\n')
    print(f'frozen {len(tasks)} tasks; {len(proposals)} endpoint proposals',flush=True)


def sample(retimer: TrajectoryTimeOpt, lane: int, duration: float, count: int) -> dict:
    """Sample the native time law in physical time and copy borrowed buffers."""
    retimer.reset_cuda_graph()
    times=torch.linspace(0,duration,count,device=retimer.device).repeat(retimer.batch_size,1).contiguous()
    state=retimer.sample(times)
    result={key:getattr(state,attr)[lane].detach().cpu().numpy().copy() for key,attr in
            [('q','position'),('qd','velocity'),('qdd','acceleration'),('jerk','jerk')]}
    result['time']=times[lane].cpu().numpy().copy()
    return result


def run() -> None:
    """Run A10 and A40, saving original geometry, time laws and all failures."""
    tasks=json.loads((OUT/'tasks.json').read_text()); rows=[]
    for population,scene,_ in SCENES:
        planner=planner_for(scene);planner.warmup(enable_graph=True,num_warmup_iterations=5)
        model=CuroboOracleModel(planner)
        for task in (item for item in tasks if item['population']==population):
            name=task['case_id']; row={key:task[key] for key in ('case_id','cohort','population','task_sha256')}
            row['methods']={}; row['errors']=[]; target=OUT/name;target.mkdir(exist_ok=True)
            try:
                torch.manual_seed(task['seed']); np.random.seed(task['seed'])
                start=JointState.from_position(torch.tensor(task['start'],device=model.device,dtype=torch.float32)[None],joint_names=model.joint_names)
                goal=JointState.from_position(torch.tensor(task['goal'],device=model.device,dtype=torch.float32)[None],joint_names=model.joint_names)
                planner.reset_seed()
                torch.cuda.synchronize();stamp=time.perf_counter()
                result=planner.plan_cspace(goal,start,max_attempts=task['max_planner_attempts'])
                torch.cuda.synchronize();row['planner_wall_s']=time.perf_counter()-stamp
                if result is None or not bool(result.success.any()):
                    row['errors'].append({'stage':'A10','reason':'planner_failed'})
                else:
                    base=read_plan(result,model.joint_names)
                    np.savez_compressed(target/'a10_interpolated.npz',**base)
                    native=planner.trajopt_solver._result_processor.get_interpolation_input()
                    inputs={key:getattr(native,key).detach().cpu().numpy().copy() for key in ('knot','knot_dt','position')}
                    retimer=planner.trajopt_solver._timeopt.optimizer; count=retimer.batch_size
                    duration=(len(base['q'])-1)*base['dt']
                    buffers={key:getattr(retimer,attr).detach().cpu().numpy().copy() for key,attr in
                             [('controls','_controls'),('start_boundary','_start_boundary'),('goal','_goal'),
                              ('timing','_timing'),('integrated_times','_times'),('success','_success'),('stationary','_stationary')]}
                    # Never reset the planner-owned sampler: its outer graph borrows its buffers.
                    cfg=TrajectoryTimeOptCfg(robot=planner.kinematics.config,batch_size=count,
                              n_knots=inputs['knot'].shape[1],use_cuda_graph=False)
                    replay=TrajectoryTimeOpt(cfg);replay._setup_path(reference=True)
                    for key,attr in [('controls','_controls'),('start_boundary','_start_boundary'),('goal','_goal'),
                                     ('timing','_timing'),('integrated_times','_times'),('success','_success'),('stationary','_stationary')]:
                        dest=getattr(replay,attr);dest.copy_(torch.as_tensor(buffers[key],device=dest.device))
                    replay._has_result=True
                    times=torch.linspace(0,duration,len(base['q']),device=model.device).repeat(count,1).contiguous()
                    positions=replay.sample(times).position.detach().cpu().numpy().copy()
                    score=np.max(np.abs(positions-base['q'][None]),axis=(1,2));lane=int(np.argmin(score))
                    if score[lane]>2e-5:
                        raise RuntimeError(f'selected TimeOpt lane mismatch {score[lane]}')
                    np.savez_compressed(target/'native_input.npz',**inputs)
                    np.savez_compressed(target/'a10_buffers.npz',**buffers)
                    actual=float(buffers['integrated_times'][lane,-1])
                    np.savez_compressed(target/'a10_dense.npz',**sample(replay,lane,actual,8193))
                    row['methods']['A10']={'duration_s':actual,'solver_converged':False,
                        'verification':{},'native_success':bool(buffers['success'][lane]),
                        'lane':lane,'lane_match_max_error_rad':float(score[lane])}
                    cfg=TrajectoryTimeOptCfg(robot=planner.kinematics.config,batch_size=count,
                              n_knots=inputs['knot'].shape[1],max_iterations=40,use_cuda_graph=False)
                    fast=TrajectoryTimeOpt(cfg)
                    tensor=lambda a: torch.as_tensor(a,device=model.device,dtype=torch.float32).contiguous()
                    start40=tensor(buffers['start_boundary'][:,0]);goal40=tensor(buffers['goal'])
                    torch.cuda.synchronize();stamp=time.perf_counter()
                    out=fast.timeopt(tensor(inputs['knot']),start40,goal40,tensor(inputs['knot_dt']))
                    torch.cuda.synchronize();elapsed=time.perf_counter()-stamp
                    actual40=float(out.duration[lane].item());success=bool(out.success[lane].item())
                    geometry_error=max(float(np.max(np.abs(getattr(fast,attr).detach().cpu().numpy()-buffers[key])))
                        for key,attr in [('controls','_controls'),('start_boundary','_start_boundary'),('goal','_goal')])
                    if geometry_error>1e-6:
                        raise RuntimeError(f'A40 geometry differs from A10: {geometry_error}')
                    arrays={'timing':out.timing_controls.detach().cpu().numpy().copy(),
                            'integrated_times':out.time.detach().cpu().numpy().copy(),
                            'success':out.success.detach().cpu().numpy().copy(),
                            'status':out.status.detach().cpu().numpy().copy()}
                    np.savez_compressed(target/'a40_buffers.npz',**arrays)
                    row['methods']['A40']={'native_success':success,'native_status':int(out.status[lane].item()),
                                           'solver_converged':False,'verification':{},'solver_wall_s':elapsed,
                                           'geometry_control_max_error_rad':geometry_error}
                    if success and np.isfinite(actual40) and actual40>0:
                        np.savez_compressed(target/'a40_dense.npz',**sample(fast,lane,actual40,8193))
                        row['methods']['A40']['duration_s']=actual40
                    else:
                        row['errors'].append({'stage':'A40','reason':'native_retimer_failed'})
            except Exception as exc:
                row['errors'].append({'stage':'execution','reason':repr(exc),'traceback':traceback.format_exc()})
            rows.append(row)
            (OUT/'pilot.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            print(name, list(row['methods']), row['errors'],flush=True)
        planner.destroy()
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='checksums.json']
    (OUT/'checksums.json').write_text(json.dumps({str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--frozen-tasks',type=Path)
    options=parser.parse_args()
    OUT=options.output
    if options.frozen_tasks is None:
        prepare()
    else:
        OUT.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(options.frozen_tasks,OUT/'tasks.json')
    run()
