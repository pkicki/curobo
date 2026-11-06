# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare selected native CUDA VJP entries with float32 finite differences."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from .curobo_adapter import CuroboOracleModel
from .oracle import Boundary, JointTimeOracle, OracleConfig


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / 'research/joint_time_oracle/results/diagnosis_20261106'
    output.mkdir(parents=True, exist_ok=True)
    base = np.load(root / 'research/joint_time_oracle/results/debug_long_20261106/task_0000_fast.npz')
    planner = MotionPlanner(MotionPlannerCfg.create(robot='franka.yml', scene_model='collision_table.yml',
                            graph_planner_algorithm='rrt', enable_timeopt=True, use_cuda_graph=True))
    model = CuroboOracleModel(planner)
    rng = np.random.default_rng(2718)
    bounds = model.limits().position
    random_q = rng.uniform(bounds[0], bounds[1], size=(300, 7)).astype(np.float32)
    with torch.no_grad():
        cs = model.collision_components(torch.as_tensor(random_q, device=model.device))
        candidates = [x.reshape(len(random_q), -1).max(dim=1).values.cpu().numpy() for x in cs]
    cases = {}
    for index, name in enumerate(('world_collision', 'self_collision')):
        idx = int(np.argmax(candidates[index]))
        cases[name] = (random_q[idx:idx+1], float(candidates[index][idx]))
    entries = []
    steps = (0.01, 0.003, 0.001)

    def check(name: str, inputs: list[np.ndarray], function, indices: list[tuple[int, int, int]]) -> None:
        tensors = [torch.tensor(x, device=model.device, dtype=torch.float32, requires_grad=True) for x in inputs]
        value = function(*tensors)
        gradients = torch.autograd.grad(value, tensors, allow_unused=True)
        analytic = [None if g is None else g.detach().cpu().numpy().copy() for g in gradients]
        for arg, row, col in indices:
            if analytic[arg] is None:
                continue
            actual = float(analytic[arg][row, col])
            for h in steps:
                shifted = [x.copy() for x in inputs]
                shifted[arg][row, col] += h
                plus = float(function(*[torch.as_tensor(x, device=model.device) for x in shifted]).item())
                shifted[arg][row, col] -= 2*h
                minus = float(function(*[torch.as_tensor(x, device=model.device) for x in shifted]).item())
                fd = (plus-minus)/(2*h)
                entries.append(dict(family=name, arg=arg, sample=row, joint=col, step=h,
                                    analytic=actual, finite_difference=fd,
                                    abs_error=abs(actual-fd),
                                    relative_error=abs(actual-fd)/max(abs(fd), abs(actual), 1e-5)))

    boundary = Boundary(base['q'][0], base['q'][-1], base['qd'][0], base['qd'][-1],
                        base['qdd'][0], base['qdd'][-1])
    cfg = OracleConfig(controls=10, samples_per_span=3, attempts=1)
    oracle = JointTimeOracle(model.limits(), boundary, model.torque, model.collision,
                             config=cfg, device=str(model.device))
    duration = 0.4128432236611843
    seed = oracle._seed(base['q'], duration).astype(np.float32)[None, :]
    spline_weights = torch.as_tensor(rng.normal(size=(len(oracle.basis.points), 7)).astype(np.float32),
                                     device=model.device)
    for state_index, family in enumerate(('position', 'velocity', 'acceleration', 'jerk')):
        def spline_fn(x, state_index=state_index):
            return (oracle._states(x.reshape(-1), oracle.basis)[1][state_index] * spline_weights).sum()
        indices = [(0, 0, i) for i in (0, 3, 10, 20, seed.shape[1]-1)]
        check(family, [seed], spline_fn, indices)
    check('duration_objective', [seed], lambda x: x[0, -1].exp(), [(0, 0, seed.shape[1]-1)])
    q = base['q'][[4, 8, 12]].astype(np.float32)
    qd = base['qd'][[4, 8, 12]].astype(np.float32)
    qdd = base['qdd'][[4, 8, 12]].astype(np.float32)
    weights = torch.as_tensor(rng.normal(size=(3, 7)).astype(np.float32), device=model.device)
    check('rnea', [q, qd, qdd], lambda x, y, z: (model.torque(x, y, z)*weights).sum(),
          [(arg, row, col) for arg in range(3) for row in range(3) for col in range(7)])
    for family, component in (('world_collision', 0), ('self_collision', 1)):
        values, cost = cases[family]
        if cost <= 0:
            continue
        def fn(x, component=component):
            return model.collision_components(x)[component].sum() * 0.37
        check(family, [values], fn, [(0, 0, i) for i in range(7)])
    result = {'steps': steps, 'entries': entries,
              'collision_case_costs': {k: v[1] for k, v in cases.items()},
              'note': 'Individual VJP entries at selected states; not a full Jacobian or general certificate.'}
    (output/'derivatives.json').write_text(json.dumps(result, indent=2)+'\n')
    for family in ('position', 'velocity', 'acceleration', 'jerk', 'duration_objective', 'rnea', 'world_collision', 'self_collision'):
        a = [x for x in entries if x['family'] == family]
        print(family, 'entries', len(a), 'max abs error', max((x['abs_error'] for x in a), default=None))
    planner.destroy()


if __name__ == '__main__':
    main()
