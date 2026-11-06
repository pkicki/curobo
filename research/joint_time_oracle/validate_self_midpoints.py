# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check self collision at a denser independently reconstructed grid."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import BSpline
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from .curobo_adapter import CuroboOracleModel


def main() -> None:
    root=Path(__file__).resolve().parents[2]
    out=root/'research/joint_time_oracle/results/native_reconstruction_20261106'
    candidate=json.loads((root/'research/joint_time_oracle/results/diagnosis_20261106/recovery.json').read_text())['best']
    spline=BSpline(np.r_[np.zeros(5),np.linspace(0,1,6),np.ones(5)],np.asarray(candidate['coefficients']),5)
    u=np.unique(np.r_[np.linspace(0,1,16385),np.arange(1,5)/5-1e-8,np.arange(1,5)/5,np.arange(1,5)/5+1e-8])
    baseline=np.load(out/'native_16385_replay.npz'); fixed=np.load(out/'fixed_dense.npz')
    cases={'baseline':(baseline['time'],baseline['q']),
           'fixed_path':(fixed['time'],fixed['q']),
           'candidate':(u*candidate['duration'],spline(u))}
    planner=MotionPlanner(MotionPlannerCfg.create(robot='franka.yml',scene_model='collision_table.yml',
                          graph_planner_algorithm='rrt',enable_timeopt=True,use_cuda_graph=True))
    model=CuroboOracleModel(planner); result={}
    for name,(times,q) in cases.items():
        with torch.no_grad():
            cost=model.collision_components(torch.as_tensor(q,device=model.device,dtype=torch.float32))[1]
            values=cost.reshape(len(q),-1).cpu().numpy().copy()
        i,j=np.unravel_index(np.argmax(values),values.shape)
        result[name]={'samples':len(q),'max_self_collision_cost':float(values[i,j]),
                      'time_s':float(times[i]),'element':int(j),
                      'note':'Point samples, including intermediate points; no continuous self-collision certificate.'}
    (out/'self_midpoints.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2)); planner.destroy()


if __name__=='__main__':
    main()
