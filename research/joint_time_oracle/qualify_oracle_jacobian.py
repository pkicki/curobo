# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check complete constraint Jacobians used by SLSQP at two saved trajectories."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from .curobo_adapter import CuroboOracleModel
from .oracle import Boundary, JointTimeOracle, OracleConfig


def main() -> None:
    root=Path(__file__).resolve().parents[2]
    out=root/'research/joint_time_oracle/results/native_reconstruction_20261106'
    old=root/'research/joint_time_oracle/results/debug_long_20261106'
    diag=root/'research/joint_time_oracle/results/diagnosis_20261106'
    base=np.load(old/'task_0000_fast.npz'); zs=np.load(diag/'states.npz')
    planner=MotionPlanner(MotionPlannerCfg.create(robot='franka.yml',scene_model='collision_table.yml',
                          graph_planner_algorithm='rrt',enable_timeopt=True,use_cuda_graph=True))
    model=CuroboOracleModel(planner)
    boundary=Boundary(base['q'][0],base['q'][-1],base['qd'][0],base['qd'][-1],base['qdd'][0],base['qdd'][-1])
    solver=JointTimeOracle(model.limits(),boundary,model.torque,model.collision,
             config=OracleConfig(controls=10,samples_per_span=3,attempts=1),device=str(model.device))
    steps=(.01,.003,.001); entries=[]
    for case in ('seed','final'):
        z=zs[case]; analytic=solver._evaluate(z,True).copy(); center=solver._evaluate(z,False).copy()
        for col in range(len(z)):
            for step in steps:
                delta=np.zeros_like(z);delta[col]=step
                plus=solver._evaluate(z+delta,False).copy();minus=solver._evaluate(z-delta,False).copy()
                fd=(plus-minus)/(2*step); fwd=(plus-center)/step; back=(center-minus)/step
                for row in range(len(center)):
                    entries.append({'case':case,'row':row,'column':col,'step':step,
                        'analytic':float(analytic[row,col]),'fd':float(fd[row]),
                        'abs_error':float(abs(analytic[row,col]-fd[row])),
                        'one_sided_difference':float(abs(fwd[row]-back[row])),
                        'possible_active_set_change':bool(abs(fwd[row]-back[row])>.01+.05*max(abs(fwd[row]),abs(back[row])))})
    result={'entries':entries,'steps':steps,'rows':'7 rows each position, velocity, acceleration, jerk, torque; final row collision',
            'note':'Nonsmooth minima and max collision can change active samples under perturbation.'}
    (out/'oracle_jacobian_sweep.json').write_text(json.dumps(result,indent=2)+'\n')
    print('entries',len(entries),'max error',max(e['abs_error'] for e in entries),
          'possible active changes',sum(e['possible_active_set_change'] for e in entries))
    planner.destroy()


if __name__=='__main__':
    main()
