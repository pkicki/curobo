# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check configured robot spheres against the table and each other using Pinocchio FK."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pinocchio as pin
import yaml


def main() -> None:
    """Record minimum sampled signed clearances without cuRobo collision kernels."""
    root = Path(__file__).resolve().parents[2]
    out = root / 'research/joint_time_oracle/results/native_reconstruction_20261106'
    cfg = yaml.safe_load((root / 'curobo/content/configs/robot/franka.yml').read_text())['robot_cfg']['kinematics']
    model = pin.buildModelFromUrdf(str(root / 'curobo/content/assets/robot/franka_description/franka_panda.urdf'))
    data = model.createData()
    q = pin.neutral(model)
    qi = [model.joints[model.getJointId(f'panda_joint{i}')].idx_q for i in range(1, 8)]
    for name in ('panda_finger_joint1', 'panda_finger_joint2'):
        q[model.joints[model.getJointId(name)].idx_q] = .04
    links, centers, radii = [], [], []
    for name, spheres in cfg['collision_spheres'].items():
        for sphere in spheres:
            links.append(name); centers.append(sphere['center']); radii.append(sphere['radius'])
    centers, radii = np.asarray(centers), np.asarray(radii)
    frames = {name: model.getFrameId(name, pin.BODY) for name in set(links)}
    ignore, padding = cfg['self_collision_ignore'], cfg['self_collision_buffer']
    pairs = [(i, j) for i in range(len(links)) for j in range(i+1, len(links))
             if links[i] != links[j] and links[j] not in ignore.get(links[i], [])
             and links[i] not in ignore.get(links[j], [])]
    a, b = np.asarray(pairs).T
    pair_radii = radii[a] + radii[b] + np.array([padding.get(links[i], 0) + padding.get(links[j], 0) for i, j in pairs])
    # Axis-aligned cuboid specified by collision_table.yml, with signed distance.
    table = yaml.safe_load((root / 'curobo/content/configs/scene/collision_table.yml').read_text())['cuboid']['table']
    box_center, half = np.array(table['pose'][:3]), np.array(table['dims']) / 2
    result = {}
    cases = {'baseline': 'baseline_validation_states.npz', 'fixed_path_historical': 'fixed_path_validation_states.npz',
             'candidate': 'candidate_validation_states.npz', 'fixed_path_pointwise_restored': 'pointwise_dense_restored.npz'}
    for name, filename in cases.items():
        file = out / filename
        if not file.exists():
            continue
        states = np.load(file); positions, times = states['q'], states['time']
        world_min, self_min = (float('inf'), None, None), (float('inf'), None, None)
        world_profile, self_profile = [], []
        for index, position in enumerate(positions):
            q[qi] = position
            pin.framesForwardKinematics(model, data, q)
            xyz = np.empty_like(centers)
            for link, frame in frames.items():
                select = np.array([n == link for n in links])
                xyz[select] = centers[select] @ data.oMf[frame].rotation.T + data.oMf[frame].translation
            delta = np.abs(xyz-box_center)-half
            distances = np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(np.max(delta, axis=1), 0) - radii
            j = int(np.argmin(distances)); world_profile.append(distances[j])
            if distances[j] < world_min[0]: world_min = (float(distances[j]), index, j)
            sep = np.linalg.norm(xyz[a]-xyz[b], axis=1) - pair_radii
            j = int(np.argmin(sep)); self_profile.append(sep[j])
            if sep[j] < self_min[0]: self_min = (float(sep[j]), index, j)
        w, wi, ws = world_min; d, di, dp = self_min
        result[name] = {'samples': len(positions), 'world_clearance_m': w, 'world_time_s': float(times[wi]),
                        'world_link': links[ws], 'self_clearance_with_link_padding_m': d,
                        'self_time_s': float(times[di]), 'self_links': [links[a[dp]], links[b[dp]]],
                        'sampled_world_feasible': w >= 0, 'sampled_self_feasible': d >= 0}
        np.savez_compressed(out / f'independent_collision_{name}.npz', time=times, world=world_profile, self_collision=self_profile)
    result['description'] = {'robot': '61 configured Franka collision spheres, Pinocchio URDF frames, locked fingers 0.04 m',
        'world': 'analytic sphere versus axis-aligned table cuboid', 'self': 'sphere pairs with configured ignore pairs and per-link padding',
        'limits': 'discrete samples only; not a swept or continuous collision certificate; excludes mesh accuracy and attached object placeholder'}
    (out / 'independent_collision.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
