# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay native TimeOpt close to dynamic peaks and geometric spline knots."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import torch
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo._src.motion.motion_trajectory_timeopt import TrajectoryTimeOpt
from curobo._src.motion.motion_trajectory_timeopt_cfg import TrajectoryTimeOptCfg

ROOT=Path(__file__).resolve().parents[2]
PILOT=ROOT/'research/joint_time_oracle/results/cohort_pilot_isolated_20261106'
RESULTS=Path(os.environ.get('ORACLE_COHORT_RESULTS',ROOT/'research/joint_time_oracle/results/cohort_joint_final_20261106'))


def main() -> None:
    """Reconstruct A10/A40 and save exact native states around suspected extrema."""
    rows=[json.loads(line) for line in (PILOT/'pilot.jsonl').read_text().splitlines()]
    diagnostics=json.loads((RESULTS/'independent_native.json').read_text())
    planner=MotionPlanner(MotionPlannerCfg.create(robot='franka.yml',scene_model=None,
                          graph_planner_algorithm='rrt',enable_timeopt=True,use_cuda_graph=True))
    for row in rows:
        case=row['case_id'];baseline=np.load(PILOT/case/'a10_buffers.npz');lane=row['methods']['A10']['lane']
        for method in ('A10','A40'):
            key=case+':'+method
            if key not in diagnostics:continue
            later=np.load(PILOT/case/'a40_buffers.npz') if method=='A40' else None
            count=len(baseline['controls']);cfg=TrajectoryTimeOptCfg(robot=planner.kinematics.config,
                   batch_size=count,n_knots=baseline['controls'].shape[1],use_cuda_graph=False)
            replay=TrajectoryTimeOpt(cfg);replay._setup_path(reference=True)
            for name,attr in [('controls','_controls'),('start_boundary','_start_boundary'),('goal','_goal'),
                              ('timing','_timing'),('integrated_times','_times'),('success','_success'),('stationary','_stationary')]:
                target=getattr(replay,attr)
                src=later[name] if later is not None and name in later.files else baseline[name]
                target.copy_(torch.as_tensor(src,device=target.device))
            replay._has_result=True;duration=float(replay._times[lane,-1].item())
            times=torch.linspace(0,duration,8193,device=replay.device).repeat(count,1).contiguous()
            replay.sample(times);progress=replay._sample_progress[lane].detach().cpu().numpy().copy()
            physical=times[lane].cpu().numpy();n=baseline['controls'].shape[1]
            knots=np.arange(1,n+3)/(n+3);knot_progress=(knots*(n+3)-1)/(n+2)
            knot_progress=knot_progress[(knot_progress>0)&(knot_progress<1)]
            centers=list(np.interp(knot_progress,progress,physical))
            for family in diagnostics[key]['families'].values():
                centers.extend([family['physical']['time_s'],family['scaled95']['time_s']])
            centers.extend(x for x in [diagnostics[key]['world_min_time_s'],diagnostics[key]['self_min_time_s']] if x is not None)
            step=duration/8192
            query=np.unique(np.clip(np.r_[(np.asarray(centers)[:,None]+np.array([-1.,-.25,0.,.25,1.])*step).ravel(),0.,duration],0,duration)).astype(np.float32)
            replay.reset_cuda_graph();t=torch.as_tensor(query,device=replay.device).repeat(count,1).contiguous();state=replay.sample(t)
            arrays={'time':query}
            for name,attr in [('q','position'),('qd','velocity'),('qdd','acceleration'),('jerk','jerk')]:
                arrays[name]=getattr(state,attr)[lane].detach().cpu().numpy().copy()
            dest=RESULTS/case;dest.mkdir(exist_ok=True);np.savez_compressed(dest/f'{method.lower()}_adaptive_dense.npz',**arrays)
        print(case,'native adaptive states',flush=True)
    planner.destroy()


if __name__=='__main__':main()
