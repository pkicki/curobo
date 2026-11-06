# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identify active samples behind a representative finite-difference discrepancy."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from .curobo_adapter import CuroboOracleModel
from .oracle import Boundary,JointTimeOracle,OracleConfig


def main() -> None:
    root=Path(__file__).resolve().parents[2];out=root/'research/joint_time_oracle/results/native_reconstruction_20261106'
    old=root/'research/joint_time_oracle/results/debug_long_20261106';diag=root/'research/joint_time_oracle/results/diagnosis_20261106'
    base=np.load(old/'task_0000_fast.npz');z=np.load(diag/'states.npz')['seed']
    planner=MotionPlanner(MotionPlannerCfg.create(robot='franka.yml',scene_model='collision_table.yml',graph_planner_algorithm='rrt',enable_timeopt=True,use_cuda_graph=True))
    model=CuroboOracleModel(planner); bc=Boundary(base['q'][0],base['q'][-1],base['qd'][0],base['qd'][-1],base['qdd'][0],base['qdd'][-1])
    solver=JointTimeOracle(model.limits(),bc,model.torque,model.collision,config=OracleConfig(controls=10,samples_per_span=3,attempts=1),device=str(model.device))
    rows=[]
    for step in (.001,.003,.01):
        for sign in (-1,0,1):
            zz=z.copy();zz[19]+=sign*step
            with torch.no_grad():
                T,(_,v,_,_,_)=solver._states(torch.tensor(zz,device=model.device,dtype=torch.float32),solver.basis)
                v=v.cpu().numpy();lo,hi=model.limits().velocity;lo*=.95;hi*=.95
                margin=np.minimum((v-lo)/(hi-lo)*2,(hi-v)/(hi-lo)*2)[:,5]
                i=int(np.argmin(margin))
            rows.append({'step':step,'sign':sign,'active_sample':i,'normalized_time':float(solver.basis.points[i]),
                         'time_s':float(solver.basis.points[i]*float(T)), 'margin':float(margin[i]),'value':float(v[i,5])})
    (out/'active_sample_diagnosis.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2));planner.destroy()


if __name__=='__main__':main()
