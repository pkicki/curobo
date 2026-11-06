# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Save FK for the independently validated comparison samples."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from .curobo_adapter import CuroboOracleModel


def main() -> None:
    root=Path(__file__).resolve().parents[2]
    out=root/'research/joint_time_oracle/results/native_reconstruction_20261106'
    planner=MotionPlanner(MotionPlannerCfg.create(robot='franka.yml',scene_model='collision_table.yml',
                          graph_planner_algorithm='rrt',enable_timeopt=True,use_cuda_graph=True))
    model=CuroboOracleModel(planner); arrays={}
    for name in ('baseline','fixed_path','candidate'):
        data=np.load(out/f'{name}_validation_states.npz')
        indices=np.linspace(0,len(data['q'])-1,1001).round().astype(int)
        q=torch.tensor(data['q'][indices],device=model.device,dtype=torch.float32)
        with torch.no_grad():
            state=model.kinematics.compute_kinematics(model._state(q))
            arrays[name]=state.tool_poses.get_link_pose('panda_hand').position.cpu().numpy().reshape(-1,3)
            arrays[name+'_q']=q.cpu().numpy(); arrays[name+'_time']=data['time'][indices]
    np.savez_compressed(out/'ee_paths.npz',**arrays)
    planner.destroy()


if __name__=='__main__':
    main()
