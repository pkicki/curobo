# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate dense saved baseline and candidate torques with aligned Pinocchio."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pinocchio as pin


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root/'research/joint_time_oracle/results/native_reconstruction_20261106'
    model = pin.buildModelFromUrdf(str(root/'curobo/content/assets/robot/franka_description/franka_panda.urdf'))
    synthetic = pin.Inertia(0.01, np.zeros(3), np.eye(3)*0.0001)
    frame = model.frames[model.getFrameId('right_gripper', pin.BODY)]
    model.inertias[frame.parentJoint] -= frame.placement.act(synthetic)
    for name in ('panda_link8', 'panda_leftfinger', 'panda_rightfinger', 'panda_hand'):
        frame = model.frames[model.getFrameId(name, pin.BODY)]
        model.inertias[frame.parentJoint] += frame.placement.act(synthetic)
    data = model.createData()
    names = [f'panda_joint{i}' for i in range(1,8)]
    qi = [model.joints[model.getJointId(name)].idx_q for name in names]
    vi = [model.joints[model.getJointId(name)].idx_v for name in names]
    q = pin.neutral(model)
    for name in ('panda_finger_joint1','panda_finger_joint2'):
        q[model.joints[model.getJointId(name)].idx_q] = 0.04
    v, a = np.zeros(model.nv), np.zeros(model.nv)
    result = {}
    for name in ('baseline','candidate','fixed_path'):
        states=np.load(out/f'{name}_validation_states.npz')
        tau=np.empty_like(states['q'],dtype=np.float64)
        for i,(pos,vel,acc) in enumerate(zip(states['q'],states['qd'],states['qdd'])):
            q[qi],v[vi],a[vi]=pos,vel,acc
            tau[i]=pin.rnea(model,data,q,v,a)[vi]
        delta=np.abs(tau-states['tau_curobo']); i,j=np.unravel_index(np.argmax(delta),delta.shape)
        low,high=states['torque_bounds']; result[name]={
            'samples':len(tau),'max_difference_nm':float(delta[i,j]),'joint':names[j],
            'time_s':float(states['time'][i]),'pinocchio_nm':float(tau[i,j]),
            'curobo_nm':float(states['tau_curobo'][i,j])}
        for label,scale in [('physical',1.),('scaled_95',.95)]:
            center=(low+high)/2; lower=center+scale*(low-center); upper=center+scale*(high-center)
            residual=np.maximum(lower-tau,tau-upper); i,j=np.unravel_index(np.argmax(residual),residual.shape)
            result[name][label]={'max_violation_nm':float(max(0.,residual[i,j])),
                'signed_residual_nm':float(residual[i,j]),'joint':names[j],
                'time_s':float(states['time'][i]),'observed_nm':float(tau[i,j]),
                'lower_nm':float(lower[j]),'upper_nm':float(upper[j])}
        np.savez_compressed(out/f'{name}_pinocchio_dense.npz', tau=tau)
    (out/'pinocchio_dense.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
