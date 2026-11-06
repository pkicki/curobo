# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-check fixed-path torque coefficients against direct aligned Pinocchio RNEA."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pinocchio as pin


def main() -> None:
    """Check every saved state using direct RNEA and save the residual."""
    root=Path(__file__).resolve().parents[2];out=root/'research/joint_time_oracle/results/native_reconstruction_20261106'
    archive=np.load(out/'pointwise_dense_restored.npz')
    s={key: archive[key] for key in archive.files}
    model=pin.buildModelFromUrdf(str(root/'curobo/content/assets/robot/franka_description/franka_panda.urdf'))
    inertia=pin.Inertia(.01,np.zeros(3),np.eye(3)*.0001)
    frame=model.frames[model.getFrameId('right_gripper',pin.BODY)];model.inertias[frame.parentJoint]-=frame.placement.act(inertia)
    for name in ('panda_link8','panda_leftfinger','panda_rightfinger','panda_hand'):
        frame=model.frames[model.getFrameId(name,pin.BODY)];model.inertias[frame.parentJoint]+=frame.placement.act(inertia)
    data=model.createData();q=pin.neutral(model);v=np.zeros(model.nv);a=np.zeros(model.nv)
    qi=[model.joints[model.getJointId(f'panda_joint{i}')].idx_q for i in range(1,8)]
    vi=[model.joints[model.getJointId(f'panda_joint{i}')].idx_v for i in range(1,8)]
    for name in ('panda_finger_joint1','panda_finger_joint2'): q[model.joints[model.getJointId(name)].idx_q]=.04
    torque=np.empty_like(s['q'])
    for i in range(len(torque)):
        q[qi]=s['q'][i];v[vi]=s['qd'][i];a[vi]=s['qdd'][i]
        torque[i]=pin.rnea(model,data,q,v,a)[vi]
    error=np.abs(torque-s['tau']);i,j=np.unravel_index(np.argmax(error),error.shape)
    result={'samples':len(torque),'max_difference_nm':float(error[i,j]),'joint':int(j+1),'time_s':float(s['time'][i]),
            'note':'Direct aligned Pinocchio RNEA compared to the affine path-dynamics coefficients, not a second independent dynamics implementation.'}
    (out/'pointwise_direct_pinocchio.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__': main()
