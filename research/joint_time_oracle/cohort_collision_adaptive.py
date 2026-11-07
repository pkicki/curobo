# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independently inspect exact geometry near sampled minimum-clearance intervals."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pinocchio as pin
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation
import yaml
from .cohort_geometry import NativeGeometry, QuinticGeometry
from .cohort_retime import ROOT,PILOT,RESULTS


def main() -> None:
    """Evaluate configured spheres near each sampled world and self clearance minimum."""
    tasks=json.loads((PILOT/'tasks.json').read_text());rows=[json.loads(x) for x in (PILOT/'pilot.jsonl').read_text().splitlines()]
    rows={r['case_id']:r for r in rows}
    diagnostics=json.loads((RESULTS/'independent_original.json').read_text())
    diagnostics.update(json.loads((RESULTS/'independent_native.json').read_text()))
    cfg=yaml.safe_load((ROOT/'curobo/content/configs/robot/franka.yml').read_text())['robot_cfg']['kinematics']
    model=pin.buildModelFromUrdf(str(ROOT/'curobo/content/assets/robot/franka_description/franka_panda.urdf'));data=model.createData();q=pin.neutral(model)
    qi=[model.joints[model.getJointId(f'panda_joint{i}')].idx_q for i in range(1,8)]
    for name in ('panda_finger_joint1','panda_finger_joint2'):q[model.joints[model.getJointId(name)].idx_q]=.04
    links=[];centers=[];radii=[]
    for name,spheres in cfg['collision_spheres'].items():
        for sphere in spheres:links.append(name);centers.append(sphere['center']);radii.append(sphere['radius'])
    centers=np.asarray(centers);radii=np.asarray(radii);ignore=cfg['self_collision_ignore'];padding=cfg['self_collision_buffer']
    grouped={name:(np.asarray([i for i,n in enumerate(links) if n==name]),model.getFrameId(name,pin.BODY)) for name in set(links)}
    pairs=[(i,j) for i in range(len(links)) for j in range(i+1,len(links)) if links[i]!=links[j] and
           links[j] not in ignore.get(links[i],[]) and links[i] not in ignore.get(links[j],[])]
    ia,ib=np.asarray(pairs).T;pair_r=radii[ia]+radii[ib]+np.array([padding.get(links[i],0)+padding.get(links[j],0) for i,j in pairs])
    result={}
    for task in tasks:
        case=task['case_id'];scene={} if task['scene'] is None else yaml.safe_load((ROOT/'curobo/content/configs/scene'/task['scene']).read_text())
        buffers=np.load(PILOT/case/'a10_buffers.npz');native=NativeGeometry(buffers,rows[case]['methods']['A10']['lane'])
        cfile=RESULTS/case/'c_best.json';quintic=QuinticGeometry(np.asarray(json.loads(cfile.read_text())['coefficients'])) if cfile.exists() else None
        for method in ('A10','A40','B','C','D'):
            key=case+':'+method;diag=diagnostics.get(key)
            source=(PILOT if method in ('A10','A40') else RESULTS)/case/f'{method.lower()}_dense.npz'
            if not diag or not source.exists() or (method in ('C','D') and quintic is None):continue
            with np.load(source) as z:state={k:z[k] for k in z.files}
            if method in ('A10','A40'):fn=native.evaluate
            elif method=='B':fn=native.evaluate_trimmed
            else:fn=quintic.evaluate
            grid=np.linspace(0,1,2049);path=fn(grid)[0];locations=[];errors=[]
            for t in (diag['world_min_time_s'],diag['self_min_time_s']):
                if t is None:continue
                idx=int(np.argmin(np.abs(state['time']-t)));target=state['q'][idx]
                if 'progress' in state:location=float(state['progress'][idx])
                elif method=='C':location=float(state['time'][idx]/state['time'][-1])
                else:
                    nearest=int(np.argmin(np.linalg.norm(path-target,axis=1)))
                    lo=grid[max(0,nearest-1)];hi=grid[min(len(grid)-1,nearest+1)]
                    location=float(minimize_scalar(lambda u:float(np.sum((fn(np.array([u]))[0][0]-target)**2)),
                                                   bounds=(lo,hi),method='bounded',options={'xatol':1e-12}).x)
                error=float(np.max(np.abs(fn(np.array([location]))[0][0]-target)));errors.append(error)
                locations.extend(np.clip(location+np.array([-2.,-1.,-.5,0.,.5,1.,2.])/8192,0,1))
            points=fn(np.unique(locations))[0];world=float('inf');self_distance=float('inf')
            for point in points:
                q[qi]=point;pin.framesForwardKinematics(model,data,q);xyz=np.empty_like(centers)
                for ids,frame_id in grouped.values():
                    placement=data.oMf[frame_id];xyz[ids]=centers[ids]@placement.rotation.T+placement.translation
                for box in scene.get('cuboid',{}).values():
                    pose=box['pose'];rotation=Rotation.from_quat([pose[4],pose[5],pose[6],pose[3]])
                    delta=np.abs(rotation.inv().apply(xyz-np.array(pose[:3])))-np.array(box['dims'])/2
                    world=min(world,float(np.min(np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(np.max(delta,axis=1),0)-radii)))
                for sphere in scene.get('sphere',{}).values():
                    center=np.asarray(sphere.get('position',sphere.get('pose',[0,0,0])[:3]))
                    world=min(world,float(np.min(np.linalg.norm(xyz-center,axis=1)-radii-float(sphere['radius']))))
                for capsule in scene.get('capsule',{}).values():
                    pose=capsule['pose'];rotation=Rotation.from_quat([pose[4],pose[5],pose[6],pose[3]])
                    start=rotation.apply(capsule['base'])+pose[:3];end=rotation.apply(capsule['tip'])+pose[:3];axis=end-start
                    u=np.clip((xyz-start)@axis/(axis@axis),0,1)
                    world=min(world,float(np.min(np.linalg.norm(xyz-(start+u[:,None]*axis),axis=1)-radii-float(capsule['radius']))))
                self_distance=min(self_distance,float(np.min(np.linalg.norm(xyz[ia]-xyz[ib],axis=1)-pair_r)))
            result[key]={'samples':len(points),'world_clearance_m':None if not np.isfinite(world) else world,
                         'self_clearance_m':self_distance,'path_match_error_rad':max(errors,default=0.),
                         'qualified':bool(max(errors,default=0.)<=1e-5 and self_distance>=-1e-6 and world>=-1e-6),
                         'note':'Sampled neighborhoods of detected minima only; not a swept or continuous certificate.'}
            (RESULTS/'independent_collision_adaptive.json').write_text(json.dumps(result,indent=2)+'\n')
        print(case,'adaptive collision neighborhoods',flush=True)


if __name__=='__main__':main()
