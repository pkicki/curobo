# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent sampled Franka dynamics and sphere collision diagnostics for pilot tasks."""
from __future__ import annotations

import argparse
import json
from .torque_active import read_profile
from pathlib import Path
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation
import yaml

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research/joint_time_oracle/results/cohort_pilot_isolated_20261106'


def main() -> None:
    """Evaluate saved samples; fail closed because adaptive interval checks are absent."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=OUT)
    parser.add_argument('--extra',type=Path)
    parser.add_argument('--methods',default='A10,A40')
    parser.add_argument('--report',default='independent_sampled.json')
    parser.add_argument('--torque-profile-dir',type=Path,
                        help='Per-task directory containing frozen torque_profile.json files')
    options=parser.parse_args();output=options.input;methods=options.methods.split(',')
    cfg=yaml.safe_load((ROOT/'curobo/content/configs/robot/franka.yml').read_text())['robot_cfg']['kinematics']
    model=pin.buildModelFromUrdf(str(ROOT/'curobo/content/assets/robot/franka_description/franka_panda.urdf'))
    inertia=pin.Inertia(.01,np.zeros(3),np.eye(3)*.0001)
    frame=model.frames[model.getFrameId('right_gripper',pin.BODY)];model.inertias[frame.parentJoint]-=frame.placement.act(inertia)
    for name in ('panda_link8','panda_leftfinger','panda_rightfinger','panda_hand'):
        frame=model.frames[model.getFrameId(name,pin.BODY)];model.inertias[frame.parentJoint]+=frame.placement.act(inertia)
    data=model.createData();q=pin.neutral(model);v=np.zeros(model.nv);a=np.zeros(model.nv)
    qi=[model.joints[model.getJointId(f'panda_joint{i}')].idx_q for i in range(1,8)]
    vi=[model.joints[model.getJointId(f'panda_joint{i}')].idx_v for i in range(1,8)]
    for name in ('panda_finger_joint1','panda_finger_joint2'):q[model.joints[model.getJointId(name)].idx_q]=.04
    links=[];local=[];radii=[]
    for name,spheres in cfg['collision_spheres'].items():
        for sphere in spheres:
            links.append(name);local.append(sphere['center']);radii.append(sphere['radius'])
    local=np.asarray(local);radii=np.asarray(radii);ignore=cfg['self_collision_ignore'];padding=cfg['self_collision_buffer']
    grouped={name:(np.asarray([i for i,n in enumerate(links) if n==name]),model.getFrameId(name,pin.BODY)) for name in set(links)}
    pairs=[(i,j) for i in range(len(links)) for j in range(i+1,len(links)) if links[i]!=links[j]
           and links[j] not in ignore.get(links[i],[]) and links[i] not in ignore.get(links[j],[])]
    ia,ib=np.asarray(pairs).T
    pair_dist=radii[ia]+radii[ib]+np.array([padding.get(links[i],0)+padding.get(links[j],0) for i,j in pairs])
    historical=np.load(ROOT/'research/joint_time_oracle/results/native_reconstruction_20261106/baseline_validation_states.npz')
    bounds={name:historical[name+'_bounds'] for name in ('position','velocity','acceleration','jerk','torque')}
    tasks=json.loads((output/'tasks.json').read_text()); rows=[json.loads(line) for line in (output/'pilot.jsonl').read_text().splitlines()]
    byid={task['case_id']:task for task in tasks}; results={}
    result_path=(options.extra/options.report) if options.extra else (output/options.report)
    for row in rows:
        task=byid[row['case_id']]; scene={}
        case_bounds={key:np.array(value,copy=True) for key,value in bounds.items()}
        if options.torque_profile_dir:
            profile=read_profile(options.torque_profile_dir/task['case_id']/'torque_profile.json')
            if profile['source_case_id']!=task['case_id']:
                raise ValueError('case/profile mismatch: '+task['case_id'])
            case_bounds['torque']=np.stack((profile['effort_lower'],profile['effort_upper']))
        if task['scene']:
            scene=yaml.safe_load((ROOT/'curobo/content/configs/scene'/task['scene']).read_text())
        for method in methods:
            source_dir = options.extra if method in ('B','C','D') or method.endswith('_scaled') or method.endswith('_adaptive') else output
            if source_dir is None: continue
            file=source_dir/task['case_id']/f'{method.lower()}_dense.npz'
            if not file.exists():continue
            source=np.load(file);s={k:source[k] for k in source.files}; n=len(s['time']);tau=np.empty_like(s['q'],dtype=np.float64)
            world_min=(float('inf'),None,None);self_min=(float('inf'),None,None)
            ee=np.empty((n,3));hand=model.getFrameId('panda_hand',pin.BODY)
            for i in range(n):
                q[qi]=s['q'][i];v[vi]=s['qd'][i];a[vi]=s['qdd'][i]
                tau[i]=pin.rnea(model,data,q,v,a)[vi];pin.framesForwardKinematics(model,data,q)
                ee[i]=data.oMf[hand].translation
                xyz=np.empty_like(local)
                for ids,frame_id in grouped.values():
                    placement=data.oMf[frame_id];xyz[ids]=local[ids]@placement.rotation.T+placement.translation
                for name,box in scene.get('cuboid',{}).items():
                    pose=box['pose']; rotation=Rotation.from_quat([pose[4],pose[5],pose[6],pose[3]])
                    delta=np.abs(rotation.inv().apply(xyz-np.array(pose[:3])))-np.array(box['dims'])/2
                    distance=np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(np.max(delta,axis=1),0)-radii
                    j=int(np.argmin(distance))
                    if distance[j]<world_min[0]:world_min=(float(distance[j]),i,name)
                for name, sphere in scene.get('sphere',{}).items():
                    center=np.asarray(sphere.get('position',sphere.get('pose',[0,0,0])[:3]))
                    distance=np.linalg.norm(xyz-center,axis=1)-radii-float(sphere['radius']);j=int(np.argmin(distance))
                    if distance[j]<world_min[0]:world_min=(float(distance[j]),i,name)
                for name,capsule in scene.get('capsule',{}).items():
                    pose=capsule['pose'];rotation=Rotation.from_quat([pose[4],pose[5],pose[6],pose[3]])
                    start=rotation.apply(capsule['base'])+pose[:3];end=rotation.apply(capsule['tip'])+pose[:3]
                    axis=end-start;u=np.clip((xyz-start)@axis/(axis@axis),0,1)
                    distance=np.linalg.norm(xyz-(start+u[:,None]*axis),axis=1)-radii-float(capsule['radius']);j=int(np.argmin(distance))
                    if distance[j]<world_min[0]:world_min=(float(distance[j]),i,name)
                separation=np.linalg.norm(xyz[ia]-xyz[ib],axis=1)-pair_dist;j=int(np.argmin(separation))
                if separation[j]<self_min[0]:self_min=(float(separation[j]),i,[links[ia[j]],links[ib[j]]])
            dest_dir=(options.extra if options.extra else source_dir)/task['case_id']
            dest_dir.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest_dir/f'{method.lower()}_pin_torque.npz', time=s['time'], torque=tau, end_effector=ee)
            result={'samples':n,'world_min_clearance_m':None if world_min[1] is None else world_min[0],
                    'world_min_time_s':None if world_min[1] is None else float(s['time'][world_min[1]]),
                    'world_object':world_min[2],'self_min_clearance_m':self_min[0],
                    'self_min_time_s':float(s['time'][self_min[1]]),'self_pair':self_min[2], 'families':{}}
            for family,values in [('position',s['q']),('velocity',s['qd']),('acceleration',s['qdd']),('jerk',s['jerk']),('torque',tau)]:
                low,high=case_bounds[family];center=(low+high)/2;result['families'][family]={}
                for label,scale in [('physical',1.),('scaled95',1. if family=='position' else .95)]:
                    if family=='torque':
                        lower=low*scale;upper=high*scale
                    else:
                        lower=center+scale*(low-center);upper=center+scale*(high-center)
                    residual=np.maximum(lower-values,values-upper);i,j=np.unravel_index(np.argmax(residual),residual.shape)
                    result['families'][family][label]={'violation':float(max(0,residual[i,j])),
                        'signed_residual':float(residual[i,j]),'joint':int(j+1),'time_s':float(s['time'][i]),
                        'value':float(values[i,j]),'lower':float(lower[j]),'upper':float(upper[j])}
            result['endpoints']={'position_start':float(np.max(abs(s['q'][0]-task['start']))),
                'position_goal':float(np.max(abs(s['q'][-1]-task['goal']))),
                'velocity':float(max(np.max(abs(s['qd'][0])),np.max(abs(s['qd'][-1])))),
                'acceleration':float(max(np.max(abs(s['qdd'][0])),np.max(abs(s['qdd'][-1]))))}
            result['note']='Independent Pinocchio aligned model and analytic sphere primitives, sampled only; adaptive checks not run.'
            if options.torque_profile_dir:
                result['torque_profile_sha256']=profile['profile_sha256']
            results[f"{task['case_id']}:{method}"]=result
            result_path.write_text(json.dumps(results,indent=2)+'\n')
            print(task['case_id'],method,'world',result['world_min_clearance_m'],'self',self_min[0],flush=True)


if __name__=='__main__':main()
