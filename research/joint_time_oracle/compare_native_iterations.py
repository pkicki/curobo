# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rerun native fixed-path TimeOpt on frozen inputs at several iteration budgets."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo._src.motion.motion_trajectory_timeopt import TrajectoryTimeOpt
from curobo._src.motion.motion_trajectory_timeopt_cfg import TrajectoryTimeOptCfg


def main() -> None:
    """Save iteration-budget outputs without changing planner defaults."""
    out = Path(__file__).resolve().parent/'results/native_reconstruction_20261106'
    inputs = np.load(out/'native_input.npz'); saved = np.load(out/'native_buffers.npz')
    lane = json.loads((out/'reconstruction.json').read_text())['selected_lane_by_position_match']
    planner=MotionPlanner(MotionPlannerCfg.create(robot='franka.yml',scene_model='collision_table.yml',
                           graph_planner_algorithm='rrt',enable_timeopt=True,use_cuda_graph=True))
    device=planner.kinematics.device_cfg.device
    tensor=lambda value: torch.as_tensor(value,device=device,dtype=torch.float32).contiguous()
    controls=tensor(inputs['knot']); start=tensor(saved['start_boundary'][:,0]); goal=tensor(saved['goal']); dt=tensor(inputs['knot_dt'])
    result=[]; arrays={}
    for iterations in (5,10,20,40,80):
        cfg=TrajectoryTimeOptCfg(robot=planner.kinematics.config,batch_size=len(controls),n_knots=controls.shape[1],
                                 max_iterations=iterations,use_cuda_graph=False)
        retimer=TrajectoryTimeOpt(cfg)
        res=retimer.timeopt(controls,start,goal,dt)
        row={'iterations':iterations,'duration_s':float(res.duration[lane].item()),'success':bool(res.success[lane].item()),
             'status':int(res.status[lane].item()),'optimization_ok':bool(res.optimization_ok[lane].item()),
             'uniform_seed_duration_s':float(res.uniform_duration[lane].item())}
        arrays[str(iterations)]=res.timing_controls[lane].detach().cpu().numpy().copy()
        result.append(row)
    np.savez_compressed(out/'native_iteration_controls.npz',**arrays)
    (out/'native_iterations.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2)); planner.destroy()


if __name__=='__main__': main()
