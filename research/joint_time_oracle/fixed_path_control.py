# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independently retime the captured native cubic geometry with SLSQP and Pinocchio."""
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import pinocchio as pin
from scipy.interpolate import BSpline
from scipy.optimize import minimize


def path_at(s: np.ndarray, buffers: dict, lane: int) -> tuple[np.ndarray, ...]:
    """Evaluate the native uniform cubic path and three path derivatives."""
    controls = buffers['controls'][lane]
    k = len(controls)
    support = np.concatenate([buffers['start_boundary'][lane], controls[:-1],
                              np.tile(buffers['goal'][lane], (3, 1))])
    coordinate = s * (k+3)
    segment = np.minimum(coordinate.astype(int), k+2)
    t = coordinate-segment
    weights = [np.stack([(1-t)**3, 3*t**3-6*t*t+4, -3*t**3+3*t*t+3*t+1, t**3],1)/6,
               np.stack([-3*(1-t)**2, 9*t*t-12*t, -9*t*t+6*t+3, 3*t*t],1)/6*(k+3),
               np.stack([6*(1-t), 18*t-12, -18*t+6, 6*t],1)/6*(k+3)**2,
               np.tile(np.array([-1.,3.,-3.,1.]),(len(s),1))*(k+3)**3]
    selected=support[segment[:,None]+np.arange(4)]
    return tuple(np.einsum('ni,nij->nj',w,selected) for w in weights)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root/'research/joint_time_oracle/results/native_reconstruction_20261106'
    info=json.loads((out/'reconstruction.json').read_text()); lane=info['selected_lane_by_position_match']
    buffers=np.load(out/'native_buffers.npz'); reference=np.load(out/'native_8193.npz')
    k=buffers['controls'].shape[1]
    # Verify the reconstructed geometry against the native sampling interface.
    mask=(reference['progress']>0)&(reference['progress']<1)
    sr=(reference['progress'][mask]*(k+2)+1)/(k+3)
    q=path_at(sr,buffers,lane)[0]
    error=float(np.max(np.abs(q-reference['q'][mask])))
    if error>2e-5:
        raise RuntimeError(f'native path reproduction failed: {error}')
    knot=np.arange(1,k+3)/(k+3)
    grid=np.unique(np.r_[np.linspace(0,1,257),knot-1e-8,knot,knot+1e-8])
    path=path_at(grid,buffers,lane)
    model=pin.buildModelFromUrdf(str(root/'curobo/content/assets/robot/franka_description/franka_panda.urdf'))
    synthetic=pin.Inertia(.01,np.zeros(3),np.eye(3)*.0001)
    frame=model.frames[model.getFrameId('right_gripper',pin.BODY)]
    model.inertias[frame.parentJoint]-=frame.placement.act(synthetic)
    for name in ('panda_link8','panda_leftfinger','panda_rightfinger','panda_hand'):
        frame=model.frames[model.getFrameId(name,pin.BODY)]
        model.inertias[frame.parentJoint]+=frame.placement.act(synthetic)
    state=pin.neutral(model); data=model.createData(); vel=np.zeros(model.nv); acc=np.zeros(model.nv)
    names=[f'panda_joint{i}' for i in range(1,8)]
    qi=[model.joints[model.getJointId(n)].idx_q for n in names]
    vi=[model.joints[model.getJointId(n)].idx_v for n in names]
    for n in ('panda_finger_joint1','panda_finger_joint2'):
        state[model.joints[model.getJointId(n)].idx_q]=.04
    coeff=np.empty((3,len(grid),7))
    for i in range(len(grid)):
        state[qi]=path[0][i]; vel[:]=0; acc[:]=0
        grav=pin.rnea(model,data,state,vel,acc)[vi].copy()
        acc[vi]=path[1][i]
        accel=pin.rnea(model,data,state,vel,acc)[vi].copy()-grav
        vel[vi]=path[1][i]; acc[vi]=path[2][i]
        moving=pin.rnea(model,data,state,vel,acc)[vi].copy()-grav
        coeff[:,i]=accel,moving,grav
    knots=np.r_[np.zeros(3),np.linspace(0,1,30),np.ones(3)]
    basis=BSpline(knots,np.eye(32),3)
    B=[basis(grid,nu=i) for i in range(3)]
    nodes,weights=np.polynomial.legendre.leggauss(6)
    qgrid=(np.arange(511)[:,None]+(nodes[None,:]+1)/2)/511
    Q=basis(qgrid.ravel()); qw=np.tile(weights/2/511,511)
    original=buffers['timing'][lane].astype(float)
    scale=float(np.median(original)); x0=original/scale
    lim=np.load(out/'baseline_validation_states.npz')
    bounds={name:lim[name+'_bounds'] for name in ('velocity','acceleration','jerk','torque')}
    def objective(x: np.ndarray) -> float:
        z=Q@(x*scale)
        return float(np.sum(qw/np.sqrt(np.maximum(z,1e-14))))
    def constraints(x: np.ndarray) -> np.ndarray:
        x0v,x1v,x2v=[mat@(x*scale) for mat in B]
        speed=np.sqrt(np.maximum(x0v,1e-14))[:,None]
        values={'velocity':path[1]*speed,
                'acceleration':path[2]*x0v[:,None]+.5*path[1]*x1v[:,None],
                'jerk':speed*(path[3]*x0v[:,None]+1.5*path[2]*x1v[:,None]+.5*path[1]*x2v[:,None]),
                'torque':coeff[2]+coeff[1]*x0v[:,None]+.5*coeff[0]*x1v[:,None]}
        rows=[]
        for name,value in values.items():
            lo,hi=bounds[name]; center=(hi+lo)/2; lo=center+.95*(lo-center); hi=center+.95*(hi-center)
            width=(hi-lo)/2
            rows.extend([np.min((value-lo)/width,axis=0),np.min((hi-value)/width,axis=0)])
        return np.concatenate(rows)
    stamp=time.perf_counter()
    result=minimize(objective,x0,method='SLSQP',bounds=[(1e-5,100.)]*32,
                    constraints=[{'type':'ineq','fun':constraints}],
                    options={'maxiter':800,'ftol':1e-8,'disp':False})
    solution=result.x*scale
    summary={'solver':'SciPy SLSQP, 32 cubic squared-speed controls; Pinocchio torque coefficients',
             'geometry_max_error_rad':error,'sampled_constraints':len(grid),'maxiter':800,
             'solver_success':bool(result.success),'message':str(result.message),'iterations':int(result.nit),
             'function_evaluations':int(result.nfev),'runtime_s':time.perf_counter()-stamp,
             'native_seed_duration_integral_s':objective(x0),'candidate_duration_integral_s':objective(result.x),
             'initial_min_scaled_margin':float(constraints(x0).min()),
             'final_min_scaled_margin':float(constraints(result.x).min()),
             'qualification':'Unqualified until native-independent dense physical-time validation; minima are sampled.'}
    np.savez_compressed(out/'fixed_path_attempt.npz',path_grid=grid,path=np.array(path),
                        torque_coefficients=coeff,initial_timing=original,solution_timing=solution)
    (out/'fixed_path_attempt.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
