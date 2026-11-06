# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Save dense states and native torques for an independent dynamics check."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from .curobo_adapter import CuroboOracleModel
from .oracle import QuinticBasis


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root/'research/joint_time_oracle/results/diagnosis_20261106'
    result = json.loads((out/'recovery.json').read_text())
    if result['best'] is None:
        raise RuntimeError('No sampled-feasible recovery to export')
    planner = MotionPlanner(MotionPlannerCfg.create(robot='franka.yml', scene_model='collision_table.yml',
                            graph_planner_algorithm='rrt', enable_timeopt=True, use_cuda_graph=True))
    model = CuroboOracleModel(planner)
    basis = QuinticBasis(10, 128, device=model.device)
    controls = torch.tensor(result['best']['coefficients'], device=model.device, dtype=torch.float32)
    duration = result['best']['duration']
    params = model.kinematics.config.kinematics_config
    (out/'curobo_inertials.json').write_text(json.dumps({
        'link_name_to_index': params.link_name_to_idx_map,
        'mass_com': params.link_masses_com.cpu().tolist(),
        'inertia': params.link_inertias.cpu().tolist()}, indent=2)+'\n')
    with torch.no_grad():
        q, qd, qdd, jerk = [basis.B[i] @ controls / duration**i for i in range(4)]
        tau = model.torque(q, qd, qdd)
        kin = model.kinematics.compute_kinematics(model._state(q))
        buffer = CollisionBuffer.from_shape(kin.robot_spheres.shape, model.device_cfg)
        swept = model.scene.get_swept_sphere_distance(
            kin, buffer, model.world_weight, model.activation,
            torch.tensor([duration/(len(q)-1)], device=model.device, dtype=torch.float32),
            sum_distance=False)
        self_cost = model.collision_components(q)[1]
        arrays = {name: val.cpu().numpy() for name, val in
                  [('q', q), ('qd', qd), ('qdd', qdd), ('jerk', jerk), ('tau_curobo', tau)]}
        limits = model.limits()
        np.savez_compressed(out/'recovery_dense.npz', **arrays, time=basis.points*duration,
                            **{name+'_bounds': value for name, value in vars(limits).items()})
        base = np.load(root/'research/joint_time_oracle/results/debug_long_20261106/task_0000_fast.npz')
        bq, bqd, bqdd, bj = [torch.as_tensor(base[name], device=model.device, dtype=torch.float32)
                              for name in ('q', 'qd', 'qdd', 'jerk')]
        btau = model.torque(bq, bqd, bqdd)
        bkin = model.kinematics.compute_kinematics(model._state(bq))
        bbuffer = CollisionBuffer.from_shape(bkin.robot_spheres.shape, model.device_cfg)
        bswept = model.scene.get_swept_sphere_distance(
            bkin, bbuffer, model.world_weight, model.activation,
            torch.tensor([float(base['dt'])], device=model.device, dtype=torch.float32),
            sum_distance=False)
        bself = model.collision_components(bq)[1]
        np.savez_compressed(out/'baseline_samples.npz', q=bq.cpu().numpy(), qd=bqd.cpu().numpy(),
                            qdd=bqdd.cpu().numpy(), jerk=bj.cpu().numpy(), tau_curobo=btau.cpu().numpy(),
                            time=np.arange(len(bq))*float(base['dt']),
                            **{name+'_bounds': value for name, value in vars(limits).items()})
        validation = {'world_swept_cost_max': float(swept.max().item()),
                      'self_discrete_cost_max': float(self_cost.max().item()),
                      'samples': len(q), 'duration_s': duration,
                      'baseline_world_swept_cost_max': float(bswept.max().item()),
                      'baseline_self_discrete_cost_max': float(bself.max().item()),
                      'baseline_samples': len(bq),
                      'note': 'Swept spheres approximate FK arcs; self collision remains sampled.'}
    (out/'native_validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(validation)
    planner.destroy()


if __name__ == '__main__':
    main()
