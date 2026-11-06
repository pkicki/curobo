# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare saved cuRobo inverse dynamics to Pinocchio on the original URDF."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pinocchio as pin


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root/'research/joint_time_oracle/results/diagnosis_20261106'
    kind = sys.argv[1] if len(sys.argv) > 1 else 'recovery'
    if kind not in ('recovery', 'baseline'):
        raise ValueError('expected recovery or baseline')
    source = np.load(out / ('recovery_dense.npz' if kind == 'recovery' else 'baseline_samples.npz'))
    urdf = root/'curobo/content/assets/robot/franka_description/franka_panda.urdf'
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    names = [f'panda_joint{i}' for i in range(1, 8)]
    arm_q = [model.joints[model.getJointId(n)].idx_q for n in names]
    arm_v = [model.joints[model.getJointId(n)].idx_v for n in names]
    q = pin.neutral(model)
    for name in ('panda_finger_joint1', 'panda_finger_joint2'):
        q[model.joints[model.getJointId(name)].idx_q] = 0.04
    v = np.zeros(model.nv)
    a = np.zeros(model.nv)
    torques = []
    for qi, vi, ai in zip(source['q'], source['qd'], source['qdd']):
        q[arm_q], v[arm_v], a[arm_v] = qi, vi, ai
        torques.append(pin.rnea(model, data, q, v, a)[arm_v].copy())
    torques = np.asarray(torques)
    difference = np.abs(torques-source['tau_curobo'])
    ix, joint = np.unravel_index(np.argmax(difference), difference.shape)
    result = {'pinocchio_version': pin.__version__, 'urdf': str(urdf.relative_to(root)),
              'sample_count': len(torques), 'max_abs_difference_nm': float(difference[ix, joint]),
              'worst_joint': names[joint], 'worst_time_s': float(source['time'][ix]),
              'pinocchio_nm': float(torques[ix, joint]),
              'curobo_nm': float(source['tau_curobo'][ix, joint]),
              'max_abs_pinocchio_torque_nm_by_joint': np.max(np.abs(torques), axis=0).tolist(),
              'note': 'URDF inertials and 0.04 m locked fingers; must establish model parity before qualification.'}
    # cuRobo supplies default inertials for massless retained links and excludes
    # the off-chain right_gripper URDF link. Align those model differences.
    synthetic = pin.Inertia(0.01, np.zeros(3), np.eye(3) * 0.0001)
    right = model.frames[model.getFrameId('right_gripper', pin.BODY)]
    model.inertias[right.parentJoint] -= right.placement.act(synthetic)
    for name in ('panda_link8', 'panda_leftfinger', 'panda_rightfinger', 'panda_hand'):
        frame = model.frames[model.getFrameId(name, pin.BODY)]
        model.inertias[frame.parentJoint] += frame.placement.act(synthetic)
    data_aligned = model.createData()
    aligned = []
    for qi, vi, ai in zip(source['q'], source['qd'], source['qdd']):
        q[arm_q], v[arm_v], a[arm_v] = qi, vi, ai
        aligned.append(pin.rnea(model, data_aligned, q, v, a)[arm_v].copy())
    aligned = np.asarray(aligned)
    aligned_difference = np.abs(aligned-source['tau_curobo'])
    ai, aj = np.unravel_index(np.argmax(aligned_difference), aligned_difference.shape)
    result['aligned_max_abs_difference_nm'] = float(aligned_difference[ai, aj])
    result['aligned_worst_joint'] = names[aj]
    result['aligned_worst_time_s'] = float(source['time'][ai])
    limits = source['torque_bounds']
    ratio = np.maximum(aligned / limits[1], aligned / limits[0])
    result['aligned_max_torque_utilization'] = float(np.max(ratio))
    result['aligned_torque_within_physical_limits'] = bool(np.all(aligned >= limits[0]) and
                                                            np.all(aligned <= limits[1]))
    result['model_alignment'] = ('Added cuRobo default 0.01 kg inertials at link8, both fingers '
                                 'and attached_object at hand; removed URDF right_gripper inertia.')
    np.savez_compressed(out/f'{kind}_pinocchio_torques.npz', tau=torques, aligned_tau=aligned)
    (out/f'{kind}_pinocchio_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
