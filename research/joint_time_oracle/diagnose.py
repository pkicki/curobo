# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce and inspect the saved single-task debugging trajectories."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import brentq

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from .curobo_adapter import CuroboOracleModel
from .oracle import Boundary, JointTimeOracle, OracleConfig


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / 'research/joint_time_oracle/results/debug_long_20261106'
    out = root / 'research/joint_time_oracle/results/diagnosis_20261106'
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads((source / 'per_task.json').read_text())[0]
    base = np.load(source / 'task_0000_fast.npz')
    planner = MotionPlanner(MotionPlannerCfg.create(robot='franka.yml', scene_model='collision_table.yml',
                            graph_planner_algorithm='rrt', enable_timeopt=True, use_cuda_graph=True))
    model = CuroboOracleModel(planner)
    limits = model.limits()
    bc = Boundary(base['q'][0], base['q'][-1], base['qd'][0], base['qd'][-1],
                  base['qdd'][0], base['qdd'][-1])
    cfg = OracleConfig(controls=10, samples_per_span=3, audit_samples_per_span=64,
                       attempts=1, maxiter=80)
    solver = JointTimeOracle(limits, bc, model.torque, model.collision, config=cfg, device=str(model.device))
    seed = solver._seed(base['q'], old['fast_duration_s'])
    final = seed.copy()
    final[:-2] = np.asarray(old['trials'][0]['coefficients'])[3:-3].reshape(-1)
    final[-1] = np.log(old['trials'][0]['duration'])

    def measure(z: np.ndarray, dense: bool) -> dict:
        basis = solver.audit_basis if dense else solver.basis
        with torch.no_grad():
            T, states = solver._states(torch.tensor(z, device=model.device, dtype=torch.float32), basis)
            q, v, a, j, _ = states
            values = [q, v, a, j, model.torque(q, v, a)]
            collisions = model.collision_components(q)
            result = {}
            for name, tensor in zip(('position', 'velocity', 'acceleration', 'jerk', 'torque'), values):
                ar = tensor.cpu().numpy()
                low, high = getattr(limits, name)
                center = (low + high) / 2
                scale = 1.0 if name == 'position' else cfg.limit_scale
                lower, upper = center + scale * (low-center), center + scale * (high-center)
                width = (upper-lower)/2
                residual = np.maximum((lower-ar)/width, (ar-upper)/width)
                ix, joint = np.unravel_index(np.argmax(residual), residual.shape)
                is_upper = ar[ix, joint] > center[joint]
                u = float(basis.points[ix])
                result[name] = {'normalized_residual': float(residual[ix, joint]),
                                'violation': float(max(0., residual[ix, joint])),
                                'joint': model.joint_names[joint], 'time_s': float(T)*u,
                                'span': min(int(u*basis.nspans), basis.nspans-1),
                                'observed': float(ar[ix, joint]),
                                'scaled_limit': float((upper if is_upper else lower)[joint]),
                                'physical_limit': float((high if is_upper else low)[joint])}
            for name, tensor in zip(('world_collision', 'self_collision'), collisions):
                c = tensor.reshape(len(q), -1).cpu().numpy()
                ix, sphere = np.unravel_index(np.argmax(c), c.shape)
                u = float(basis.points[ix])
                result[name] = {'cost': float(c[ix, sphere]), 'time_s': float(T)*u,
                                'span': min(int(u*basis.nspans), basis.nspans-1),
                                'element': int(sphere), 'limit': 0.0}
            result['endpoints_max_abs'] = {name: float(np.max(np.abs(obs - target)))
                for name, obs, target in [('q0', q[0].cpu().numpy(), bc.q0),
                ('q1', q[-1].cpu().numpy(), bc.q1), ('v0', v[0].cpu().numpy(), bc.v0),
                ('v1', v[-1].cpu().numpy(), bc.v1), ('a0', a[0].cpu().numpy(), bc.a0),
                ('a1', a[-1].cpu().numpy(), bc.a1)]}
            result['duration_s'] = float(T)
            result['samples'] = len(q)
            return result

    result = {'seed_collocation': measure(seed, False), 'seed_dense': measure(seed, True),
              'final_collocation': measure(final, False), 'final_dense': measure(final, True)}
    # Stretching normalized physical time keeps geometry fixed for rest endpoints only.
    def worst(log_t: float) -> float:
        z = seed.copy()
        z[-1] = log_t
        metrics = measure(z, True)
        return max(metrics[name]['normalized_residual'] for name in
                   ('position', 'velocity', 'acceleration', 'jerk', 'torque'))
    stretched = brentq(lambda t: worst(np.log(t)), old['fast_duration_s'], 5.0, xtol=1e-5)
    stretch_z = seed.copy(); stretch_z[-1] = np.log(stretched * 1.001)
    result['stretched_seed_dense'] = measure(stretch_z, True)
    fitted = solver.basis.spline(np.linspace(0, 1, len(base['q']))) @ np.asarray(
        solver.audit(seed)['coefficients'])
    result['fit_max_abs_error_rad'] = float(np.max(np.abs(fitted-base['q'])))
    result['fit_max_joint_norm_error_rad'] = float(np.max(np.linalg.norm(fitted-base['q'], axis=1)))
    (out/'diagnosis.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(out/'states.npz', seed=seed, final=final, stretched=stretch_z)
    print(json.dumps(result, indent=2))
    planner.destroy()


if __name__ == '__main__':
    main()
