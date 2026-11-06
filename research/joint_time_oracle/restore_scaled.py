# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate mild uniform slowdowns of A and C against common 95% dynamic limits."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pinocchio as pin


def main() -> None:
    """Scale sampled derivatives, recompute aligned RNEA and record all margins."""
    root=Path(__file__).resolve().parents[2];out=root/'research/joint_time_oracle/results/native_reconstruction_20261106'
    model=pin.buildModelFromUrdf(str(root/'curobo/content/assets/robot/franka_description/franka_panda.urdf'))
    inertia=pin.Inertia(.01,np.zeros(3),np.eye(3)*.0001)
    frame=model.frames[model.getFrameId('right_gripper',pin.BODY)];model.inertias[frame.parentJoint]-=frame.placement.act(inertia)
    for name in ('panda_link8','panda_leftfinger','panda_rightfinger','panda_hand'):
        frame=model.frames[model.getFrameId(name,pin.BODY)];model.inertias[frame.parentJoint]+=frame.placement.act(inertia)
    data=model.createData();q=pin.neutral(model);v=np.zeros(model.nv);a=np.zeros(model.nv)
    qi=[model.joints[model.getJointId(f'panda_joint{i}')].idx_q for i in range(1,8)]
    vi=[model.joints[model.getJointId(f'panda_joint{i}')].idx_v for i in range(1,8)]
    for name in ('panda_finger_joint1','panda_finger_joint2'): q[model.joints[model.getJointId(name)].idx_q]=.04
    result={}
    for name, factor in [('baseline',1.0001),('candidate',1.0002)]:
        archive=np.load(out/f'{name}_validation_states.npz');state={k:archive[k] for k in archive.files}
        time=state['time']*factor;pos=state['q'];qd=state['qd']/factor;qdd=state['qdd']/factor**2;jerk=state['jerk']/factor**3
        torque=np.empty_like(pos)
        for i in range(len(pos)):
            q[qi]=pos[i];v[vi]=qd[i];a[vi]=qdd[i];torque[i]=pin.rnea(model,data,q,v,a)[vi]
        values={'position':pos,'velocity':qd,'acceleration':qdd,'jerk':jerk,'torque':torque}
        row={'duration_s':float(time[-1]),'time_scale':factor,'samples':len(time),'families':{}}
        for family,value in values.items():
            lo,hi=state[family+'_bounds'];center=(lo+hi)/2;scale=1 if family=='position' else .95
            lo=center+scale*(lo-center);hi=center+scale*(hi-center)
            residual=np.maximum(lo-value,value-hi);i,j=np.unravel_index(np.argmax(residual),residual.shape)
            row['families'][family]={'signed_residual':float(residual[i,j]),'max_violation':float(max(0,residual[i,j])),
                'time_s':float(time[i]),'joint':int(j+1),'value':float(value[i,j]),'lower':float(lo[j]),'upper':float(hi[j])}
        result[name]=row
        np.savez_compressed(out/f'{name}_scaled_restored.npz',time=time,q=pos,qd=qd,qdd=qdd,jerk=jerk,tau=torque)
    result['note']='8193 samples, unchanged geometry, uniform time scaling; collision profiles unchanged. Dense intermediate extrema and continuous behavior require further verification.'
    (out/'scaled_restorations.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__': main()
