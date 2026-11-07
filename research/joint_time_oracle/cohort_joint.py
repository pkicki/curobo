# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run bounded multistart quintic joint optimization on frozen pilot trajectories."""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import BSpline
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from .curobo_adapter import CuroboOracleModel
from .oracle import Boundary, JointTimeOracle, OracleConfig, JointLimits
from .torque_active import read_profile

ROOT=Path(__file__).resolve().parents[2]
PILOT=Path(os.environ.get('ORACLE_COHORT_SOURCE', ROOT/'research/joint_time_oracle/results/cohort_pilot_isolated_20261106'))
RESULTS=Path(os.environ.get('ORACLE_COHORT_RESULTS', ROOT/'research/joint_time_oracle/results/cohort_joint_pilot_20261106'))


def main() -> None:
    """Restore fitted seeds, run two spline resolutions and retain each trial."""
    parser=argparse.ArgumentParser();parser.add_argument('case');parser.add_argument('--maxiter',type=int,default=40)
    parser.add_argument('--resolutions',default='10,14')
    parser.add_argument('--attempts',type=int,default=2)
    parser.add_argument('--trust-radius',type=float,default=0.5)
    parser.add_argument('--seed-noise',type=float,default=0.04)
    parser.add_argument('--solver',choices=('slsqp','ipopt'),default='slsqp')
    args=parser.parse_args();task=next(t for t in json.loads((PILOT/'tasks.json').read_text()) if t['case_id']==args.case)
    archive=np.load(PILOT/args.case/'a10_interpolated.npz');base={k:archive[k] for k in archive.files}
    duration=float(base['dt'])*(len(base['q'])-1)
    planner=MotionPlanner(MotionPlannerCfg.create(robot='franka.yml',scene_model=task['scene'],
                          graph_planner_algorithm='rrt',enable_timeopt=True,use_cuda_graph=True))
    model=CuroboOracleModel(planner); boundary=Boundary.rest(task['start'],task['goal'])
    limits=model.limits()
    profile_path=os.environ.get('ORACLE_TORQUE_PROFILE')
    if profile_path:
        profile=read_profile(profile_path)
        if profile['source_case_id']!=args.case:
            raise ValueError('torque profile/source task mismatch')
        expected_nominal=np.stack((profile['nominal_effort_lower'],profile['nominal_effort_upper']))
        if not np.allclose(limits.torque,expected_nominal,rtol=1e-5,atol=1e-5):
            raise ValueError('frozen nominal effort limits differ from current CUDA robot model')
        limits=JointLimits(position=limits.position,velocity=limits.velocity,
                           acceleration=limits.acceleration,jerk=limits.jerk,
                           torque=np.stack((profile['effort_lower'],profile['effort_upper'])))
    dest=RESULTS/args.case;dest.mkdir(parents=True,exist_ok=True);attempts=[];best=None
    resolutions=tuple(int(x) for x in args.resolutions.split(','))
    if not resolutions or any(x<8 for x in resolutions):
        raise ValueError('invalid quintic resolutions')
    if args.attempts<1 or args.maxiter<1:
        raise ValueError('invalid optimizer budget')
    for resolution in resolutions:
        cfg=OracleConfig(controls=resolution,samples_per_span=8,audit_samples_per_span=64,
                         maxiter=args.maxiter,attempts=args.attempts,rng_seed=task['seed'],
                          trust_radius=args.trust_radius,seed_noise=args.seed_noise,
                          constraint_tol=1e-4,solver=args.solver)
        oracle=JointTimeOracle(limits,boundary,model.torque,model.collision,config=cfg,device=str(model.device))
        restoration=[];seed_duration=duration
        for factor in (1.,1.2,1.5,2.,3.):
            seed_duration=duration*factor; seed=oracle._seed(base['q'],seed_duration);audit=oracle.audit(seed)
            restoration.append({'duration':seed_duration,'sampled_feasible':audit['feasible'],
                                'max_violation':audit['max_violation'],'collision_cost':audit['collision_cost']})
            if audit['feasible']:
                audit['candidate_origin']='feasibility_restored_initial_fit'
                if best is None or audit['duration']<best['duration']:
                    best=audit
                break
        torch.cuda.synchronize();stamp=time.perf_counter()
        result=oracle.solve(base['q'],seed_duration);torch.cuda.synchronize()
        trial_restoration=[]
        for trial in result['trials']:
            controls=np.asarray(trial['coefficients']);interior=controls[3:-3].reshape(-1)
            for factor in (1.,1.001,1.005,1.02,1.05,1.1,1.2):
                candidate_duration=trial['duration']*factor
                z=np.r_[interior,0.,np.log(candidate_duration)]
                audit=oracle.audit(z)
                trial_restoration.append({'attempt':trial['attempt'],'time_factor':factor,
                    'duration':candidate_duration,'sampled_feasible':audit['feasible'],
                    'max_violation':audit['max_violation'],'collision_cost':audit['collision_cost']})
                if audit['feasible']:
                    audit['candidate_origin']='restored_joint_solver_iterate'
                    if best is None or audit['duration']<best['duration']:best=audit
                    break
        attempts.append({'resolution':resolution,'restoration':restoration,'trial_restoration':trial_restoration,
                         'result':result,'wall_s':time.perf_counter()-stamp})
        if result['best'] is not None and (best is None or result['best']['duration']<best['duration']):
            best=result['best'];best['candidate_origin']='joint_solver_iterate'
        (dest/'c_attempts.json').write_text(json.dumps({'attempts':attempts,'best_found':best is not None,
             'note':'Native sampled checks only; independent validation and adaptive refinement required.'},indent=2)+'\n')
        print(args.case,'resolution',resolution,'status',result['status'],'wall',attempts[-1]['wall_s'],flush=True)
    if best is not None:
        if profile_path:
            best['torque_profile_sha256']=profile['profile_sha256']
        (dest/'c_best.json').write_text(json.dumps(best,indent=2)+'\n')
        controls=np.asarray(best['coefficients']);n=len(controls);T=best['duration']
        spline=BSpline(np.r_[np.zeros(5),np.linspace(0,1,n-4),np.ones(5)],controls,5)
        u=np.linspace(0,1,8193)
        knots=np.unique(spline.t);interior=knots[(knots>1e-9)&(knots<1-1e-9)]
        u=np.unique(np.r_[u,interior-1e-9,interior,interior+1e-9]);extra=[]
        for derivative in range(4):
            values=spline(u,nu=derivative)
            for joint in range(values.shape[1]):
                for index in (int(np.argmin(values[:,joint])),int(np.argmax(values[:,joint]))):
                    extra.extend(np.clip(u[index]+np.array([-1.,-.5,.5,1.])/8192,0,1))
        u=np.unique(np.r_[u,extra])
        np.savez_compressed(dest/'c_dense.npz',time=u*T,q=spline(u),qd=spline(u,nu=1)/T,
                            qdd=spline(u,nu=2)/T**2,jerk=spline(u,nu=3)/T**3)

    planner.destroy()


if __name__=='__main__':main()
