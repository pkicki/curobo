# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay captured TimeOpt buffers without rerunning planning or retiming."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo._src.motion.motion_trajectory_timeopt import TrajectoryTimeOpt
from curobo._src.motion.motion_trajectory_timeopt_cfg import TrajectoryTimeOptCfg


def main() -> None:
    root=Path(__file__).resolve().parents[2]
    out=root/'research/joint_time_oracle/results/native_reconstruction_20261106'
    data=np.load(out/'native_buffers.npz')
    info=json.loads((out/'reconstruction.json').read_text()); lane=info['selected_lane_by_position_match']
    planner=MotionPlanner(MotionPlannerCfg.create(robot='franka.yml',scene_model='collision_table.yml',
                          graph_planner_algorithm='rrt',enable_timeopt=True,use_cuda_graph=True))
    retimer=TrajectoryTimeOpt(TrajectoryTimeOptCfg(robot=planner.kinematics.config,
                          batch_size=len(data['controls']), n_knots=data['controls'].shape[1]))
    retimer._setup_path(reference=True)
    for key,attr in [('controls','_controls'),('start_boundary','_start_boundary'),('goal','_goal'),
                     ('timing','_timing'),('integrated_times','_times'),('success','_success'),
                     ('stationary','_stationary')]:
        target=getattr(retimer,attr); target.copy_(torch.as_tensor(data[key],device=target.device))
    retimer._has_result=True
    device=retimer.device; duration=info['rerun_duration_s']; n=16385
    times=torch.linspace(0,duration,n,device=device).repeat(retimer.batch_size,1).contiguous()
    states=retimer.sample(times)
    keys=[('q','position'),('qd','velocity'),('qdd','acceleration'),('jerk','jerk')]
    arrays={key:getattr(states,attr)[lane].detach().cpu().numpy().copy() for key,attr in keys}
    arrays['time']=times[lane].cpu().numpy(); arrays['progress']=retimer._sample_progress[lane].cpu().numpy()
    reference=np.load(out/'native_8193.npz')
    residual={key:float(np.max(np.abs(arrays[key][::2]-reference[key]))) for key,_ in keys}
    np.savez_compressed(out/'native_16385_replay.npz',**arrays)
    knots=np.arange(1,data['controls'].shape[1]+3)/(data['controls'].shape[1]+3)
    progress=(knots*(data['controls'].shape[1]+3)-1)/(data['controls'].shape[1]+2)
    progress=progress[(progress>0)&(progress<1)]
    at=np.interp(progress,arrays['progress'],arrays['time'])
    adaptive=np.unique(np.clip(np.r_[at-1e-6,at,at+1e-6],0,duration)).astype(np.float32)
    retimer.reset_cuda_graph()
    query=torch.tensor(adaptive,device=device).repeat(retimer.batch_size,1).contiguous()
    near=retimer.sample(query)
    np.savez_compressed(out/'native_adaptive_knots.npz',time=adaptive,
        **{key:getattr(near,attr)[lane].detach().cpu().numpy().copy() for key,attr in keys})
    (out/'replay.json').write_text(json.dumps({'vs_8193_max_abs':residual,'samples':n,
                                              'duration_s':duration},indent=2)+'\n')
    print(residual)
    planner.destroy()


if __name__=='__main__':
    main()
