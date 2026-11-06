# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep native CUDA RNEA Jacobians and representative collision gradients."""
from __future__ import annotations

import json
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
    model=CuroboOracleModel(planner); rng=np.random.default_rng(9011)
    dense=np.load(out/'baseline_validation_states.npz')
    samples=[(dense['q'][2000],dense['qd'][2000],dense['qdd'][2000]),
             (dense['q'][6000],dense['qd'][6000],dense['qdd'][6000]),
             (rng.uniform(*model.limits().position).astype(np.float32),
              rng.uniform(-1,1,7).astype(np.float32),rng.uniform(-4,4,7).astype(np.float32))]
    steps=(.01,.003,.001); entries=[]
    for case, inputs in enumerate(samples):
        inputs=[a.astype(np.float32)[None,:] for a in inputs]
        t=[torch.tensor(x,device=model.device,requires_grad=True) for x in inputs]
        result=model.torque(*t)
        gradients=[]
        for output in range(7):
            g=torch.autograd.grad(result[0,output],t,retain_graph=output<6)
            gradients.append([v.detach().cpu().numpy().copy() for v in g])
        for arg in range(3):
            for joint in range(7):
                for step in steps:
                    x=[a.copy() for a in inputs]; x[arg][0,joint]+=step
                    plus=model.torque(*[torch.as_tensor(a,device=model.device) for a in x]).detach().cpu().numpy().copy()
                    x[arg][0,joint]-=2*step
                    minus=model.torque(*[torch.as_tensor(a,device=model.device) for a in x]).detach().cpu().numpy().copy()
                    for output in range(7):
                        analytic=float(gradients[output][arg][0,joint]); fd=float((plus[0,output]-minus[0,output])/(2*step))
                        entries.append({'family':'rnea','case':case,'arg':arg,'joint':joint,'output':output,
                            'step':step,'analytic':analytic,'fd':fd,'abs_error':abs(analytic-fd)})
    lo,hi=model.limits().position
    random=rng.uniform(lo,hi,size=(1500,7)).astype(np.float32)
    with torch.no_grad():
        cost=model.collision_components(torch.tensor(random,device=model.device))
        costs=[x.reshape(len(random),-1).max(dim=1).values.cpu().numpy() for x in cost]
    cases={}
    for family,kind in [('world',0),('self',1)]:
        vals=costs[kind]; positive=np.flatnonzero(vals>1e-5)
        if len(positive)==0:
            cases[family]={'status':'no positive case found'}; continue
        nearest=positive[np.argmin(vals[positive])]; deepest=int(np.argmax(vals))
        cases[family]={'near_index':int(nearest),'near_cost':float(vals[nearest]),
                       'deep_index':deepest,'deep_cost':float(vals[deepest])}
        for label,pos in [('free',dense['q'][2000]),('near',random[nearest]),('penetrating',random[deepest])]:
            x=torch.tensor(pos[None],device=model.device,dtype=torch.float32,requires_grad=True)
            value=model.collision_components(x)[kind].max()*.37
            analytic=torch.autograd.grad(value,x)[0].detach().cpu().numpy().copy()[0]
            center=float(value.item())
            for joint in range(7):
                for step in steps:
                    shifted=pos.astype(np.float32).copy(); shifted[joint]+=step
                    plus=float((model.collision_components(torch.tensor(shifted[None],device=model.device))[kind].max()*.37).item())
                    shifted[joint]-=2*step
                    minus=float((model.collision_components(torch.tensor(shifted[None],device=model.device))[kind].max()*.37).item())
                    fd=(plus-minus)/(2*step); fwd=(plus-center)/step; bwd=(center-minus)/step
                    entries.append({'family':family,'case':label,'joint':joint,'step':step,
                        'analytic':float(analytic[joint]),'fd':fd,'abs_error':abs(float(analytic[joint])-fd),
                        'center':center,'plus':plus,'minus':minus,'one_sided_difference':abs(fwd-bwd),
                        'possible_transition':bool(min(plus,minus,center)<=1e-9 < max(plus,minus,center) or
                            abs(fwd-bwd)>.005+0.05*max(abs(fwd),abs(bwd)))})
    result={'steps':steps,'collision_cases':cases,'entries':entries,
            'note':'RNEA full 7x21 Jacobian at two baseline states and one deterministic random state; collision uses maxima and can be nonsmooth.'}
    (out/'cuda_derivative_sweep.json').write_text(json.dumps(result,indent=2)+'\n')
    for family in ('rnea','world','self'):
        a=[x for x in entries if x['family']==family]
        print(family,'entries',len(a),'max abs error',max((x['abs_error'] for x in a),default=None),
              'possible transitions',sum(x.get('possible_transition',False) for x in a))
    print(cases)
    planner.destroy()


if __name__=='__main__':
    main()
