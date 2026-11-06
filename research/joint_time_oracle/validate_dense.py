# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply identical sampled cuRobo checks to the native baseline and quintic candidate."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import BSpline
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from .curobo_adapter import CuroboOracleModel


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root/'research/joint_time_oracle/results/native_reconstruction_20261106'
    prior = root/'research/joint_time_oracle/results/diagnosis_20261106'
    old = root/'research/joint_time_oracle/results/debug_long_20261106'
    assert json.loads((source/'reconstruction.json').read_text())['historical_reconstruction_valid']
    candidate = json.loads((prior/'recovery.json').read_text())['best']
    c = np.asarray(candidate['coefficients'])
    spline = BSpline(np.r_[np.zeros(5), np.linspace(0,1,6), np.ones(5)], c, 5)
    original = np.load(old/'task_0000_fast.npz')
    planner = MotionPlanner(MotionPlannerCfg.create(robot='franka.yml', scene_model='collision_table.yml',
                            graph_planner_algorithm='rrt', enable_timeopt=True, use_cuda_graph=True))
    model = CuroboOracleModel(planner)
    limits = model.limits()
    result = {}
    for n in (641, 4097, 8193, 16385):
        baseline = np.load(source/('native_16385_replay.npz' if n == 16385 else f'native_{n}.npz'))
        duration = candidate['duration']
        u = np.linspace(0,1,n)
        cases = {'baseline': {k: baseline[k] for k in ('q','qd','qdd','jerk','time')},
                 'candidate': {'time': u*duration, **{key: (spline(u,nu=i)/duration**i).astype(np.float32)
                        for i,key in enumerate(('q','qd','qdd','jerk'))}}}
        if n == 8193 and (source/'fixed_dense.npz').exists():
            adaptive = np.load(source/'native_adaptive_knots.npz')
            old_dense = json.loads((source/'dense_validation.json').read_text())['candidate_8193']
            peaks = [old_dense[f]['physical']['time_s']/duration for f in
                     ('position', 'velocity', 'acceleration', 'jerk', 'torque')]
            knots = np.arange(1,5)/5
            local = np.asarray(peaks+knots.tolist())
            local = np.unique(np.clip(np.r_[local-1e-7,local,local+1e-7],0,1))
            cases['candidate_adaptive'] = {'time':local*duration, **{
                key:(spline(local,nu=i)/duration**i).astype(np.float32)
                for i,key in enumerate(('q','qd','qdd','jerk'))}}
            cases['baseline_adaptive'] = {k: adaptive[k] for k in ('q', 'qd', 'qdd', 'jerk', 'time')}
            fixed = np.load(source/'fixed_dense.npz')
            cases['fixed_path'] = {k: fixed[k] for k in ('q', 'qd', 'qdd', 'jerk', 'time')}
        for name, arr in cases.items():
            n_actual = len(arr['q'])
            time = arr['time']
            q, qd, qdd = [torch.as_tensor(arr[k],device=model.device,dtype=torch.float32) for k in ('q','qd','qdd')]
            with torch.no_grad():
                tau = model.torque(q,qd,qdd).cpu().numpy().copy()
                kin = model.kinematics.compute_kinematics(model._state(q))
                buffer = CollisionBuffer.from_shape(kin.robot_spheres.shape, model.device_cfg)
                swept = model.scene.get_swept_sphere_distance(kin, buffer, model.world_weight,
                    model.activation, torch.tensor([float(time[-1])/(n_actual-1)],device=model.device),
                    sum_distance=False).detach().cpu().numpy().copy()
                selfcost = model.collision_components(q)[1].detach().cpu().numpy().copy()
            row = {'last_sample_time_s': float(time[-1]), 'samples': n_actual}
            for family, key in [('position','q'),('velocity','qd'),('acceleration','qdd'),('jerk','jerk'),('torque','tau')]:
                value = tau if key=='tau' else arr[key]
                low,high = getattr(limits,family)
                row[family] = {}
                for label,scale in [('physical',1.),('scaled_95',1. if family=='position' else .95)]:
                    mid=(low+high)/2; lower=mid+scale*(low-mid); upper=mid+scale*(high-mid)
                    residual=np.maximum(lower-value,value-upper)
                    i,j=np.unravel_index(np.argmax(residual),residual.shape)
                    row[family][label]={'max_violation':float(max(0.,residual[i,j])),
                        'signed_residual':float(residual[i,j]),'time_s':float(time[i]),
                        'joint':model.joint_names[j],'observed':float(value[i,j]),
                        'lower':float(lower[j]),'upper':float(upper[j])}
            for label, value in [('world_swept',swept),('self_discrete',selfcost)]:
                value=value.reshape(n_actual,-1); i,j=np.unravel_index(np.argmax(value),value.shape)
                row[label]={'max_cost':float(value[i,j]),'time_s':float(time[i]),'element':int(j)}
            if 'adaptive' not in name:
                row['endpoint_residuals'] = {label:float(np.max(np.abs(arr[key][index]-original[key][index])))
                    for label,key,index in [('q0','q',0),('q1','q',-1),('v0','qd',0),('v1','qd',-1),
                                        ('a0','qdd',0),('a1','qdd',-1)]}
            if n == 8193:
                np.savez_compressed(source/f'{name}_validation_states.npz', **arr, tau_curobo=tau,
                                    **{k+'_bounds':v for k,v in vars(limits).items()})
            result[f'{name}_{n}']=row
    (source/'dense_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    planner.destroy()


if __name__ == '__main__':
    main()
