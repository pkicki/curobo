# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one bounded dense-collocation recovery attempt on the saved task."""
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
    source = root / 'research/joint_time_oracle/results/debug_long_20261106'
    output = root / 'research/joint_time_oracle/results/diagnosis_20261106'
    base = np.load(source/'task_0000_fast.npz')
    diag = json.loads((output/'diagnosis.json').read_text())
    planner = MotionPlanner(MotionPlannerCfg.create(robot='franka.yml', scene_model='collision_table.yml',
                            graph_planner_algorithm='rrt', enable_timeopt=True, use_cuda_graph=True))
    model = CuroboOracleModel(planner)
    boundary = Boundary(base['q'][0], base['q'][-1], base['qd'][0], base['qd'][-1],
                        base['qdd'][0], base['qdd'][-1])
    cfg = OracleConfig(controls=10, samples_per_span=16, audit_samples_per_span=128,
                       maxiter=100, attempts=1, ftol=2e-5)
    solver = JointTimeOracle(model.limits(), boundary, model.torque, model.collision,
                             config=cfg, device=str(model.device))
    torch.cuda.synchronize()
    import time
    stamp = time.perf_counter()
    result = solver.solve(base['q'], diag['stretched_seed_dense']['duration_s'])
    torch.cuda.synchronize()
    result['wall_s'] = time.perf_counter() - stamp
    (output/'recovery.json').write_text(json.dumps(result, indent=2)+'\n')
    print('status', result['status'], 'wall', result['wall_s'])
    for trial in result['trials']:
        print(trial['message'], trial['duration'], trial['max_violation'])
    planner.destroy()


if __name__ == '__main__':
    main()
