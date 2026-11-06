# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rerun the frozen task and capture the selected native TimeOpt buffers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState
from .run_study import read_plan


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / 'research/joint_time_oracle/results/debug_long_20261106'
    out = root / 'research/joint_time_oracle/results/native_reconstruction_20261106'
    out.mkdir(parents=True, exist_ok=True)
    task = json.loads((source / 'per_task.json').read_text())[0]
    old = np.load(source/'task_0000_fast.npz')
    planner = MotionPlanner(MotionPlannerCfg.create(robot='franka.yml', scene_model='collision_table.yml',
                            graph_planner_algorithm='rrt', enable_timeopt=True, use_cuda_graph=True))
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    names = planner.joint_names
    device = planner.kinematics.device_cfg.device
    start = JointState.from_position(torch.tensor(task['start'], device=device, dtype=torch.float32)[None], joint_names=names)
    goal = JointState.from_position(torch.tensor(task['goal'], device=device, dtype=torch.float32)[None], joint_names=names)
    result = planner.plan_cspace(goal, start, max_attempts=5)
    if result is None or not bool(result.success.any()):
        raise RuntimeError('frozen task rerun did not succeed')
    plan = read_plan(result, names)
    native = planner.trajopt_solver._result_processor.get_interpolation_input()
    native_inputs = {key: getattr(native, key).detach().cpu().numpy().copy()
                     for key in ('knot','knot_dt','position') if getattr(native, key) is not None}
    np.savez_compressed(out/'native_input.npz', **native_inputs)
    retimer = planner.trajopt_solver._timeopt.optimizer
    count = retimer.batch_size
    # Match all retained retimer lanes to the actual 18-point selected rerun.
    times = torch.linspace(0, float(plan['dt'])*(len(plan['q'])-1), len(plan['q']),
                           device=device, dtype=torch.float32).repeat(count, 1).contiguous()
    retimer.reset_cuda_graph()
    sampled = retimer.sample(times)
    sampled_arrays = {key: getattr(sampled, attr).detach().cpu().numpy().copy()
                      for key, attr in [('q','position'),('qd','velocity'),('qdd','acceleration'),('jerk','jerk')]}
    scores = np.max(np.abs(sampled_arrays['q'] - plan['q'][None]), axis=(1,2))
    lane = int(np.argmin(scores))
    agreement = {key: float(np.max(np.abs(sampled_arrays[key][lane]-plan[key]))) for key in sampled_arrays}
    historical = {key: float(np.max(np.abs(plan[key]-old[key]))) if plan[key].shape == old[key].shape else None
                  for key in sampled_arrays}
    duration = float(retimer._times[lane,-1].item())
    buffers = {'controls': retimer._controls.detach().cpu().numpy().copy(),
               'start_boundary': retimer._start_boundary.detach().cpu().numpy().copy(),
               'goal': retimer._goal.detach().cpu().numpy().copy(),
               'timing': retimer._timing.detach().cpu().numpy().copy(),
               'integrated_times': retimer._times.detach().cpu().numpy().copy(),
               'success': retimer._success.detach().cpu().numpy().copy(),
               'stationary': retimer._stationary.detach().cpu().numpy().copy()}
    np.savez_compressed(out/'native_buffers.npz', **buffers)
    np.savez_compressed(out/'rerun_interpolated.npz', **plan)
    summary = {'historical_duration_s': task['fast_duration_s'], 'rerun_duration_s': duration,
               'selected_lane_by_position_match': lane, 'lane_count': count,
               'native_to_rerun_max_abs': agreement, 'rerun_to_historical_max_abs': historical,
               'historical_reconstruction_valid': bool(all(value is not None and value <= tol for value, tol in
                    zip(historical.values(), (1e-5, 1e-4, 1e-3, 1e-2))) and
                    abs(duration-task['fast_duration_s']) <= 1e-5),
               'note': 'Buffers cover every retimer lane; native sampler outputs borrow internal buffers.'}
    for num in (641,4097,8193):
        retimer.reset_cuda_graph()
        times = torch.linspace(0, duration, num, device=device, dtype=torch.float32).repeat(count,1).contiguous()
        dense = retimer.sample(times)
        np.savez_compressed(out/f'native_{num}.npz', time=times[lane].cpu().numpy(),
                            **{key:getattr(dense,attr)[lane].detach().cpu().numpy().copy()
                               for key,attr in [('q','position'),('qd','velocity'),('qdd','acceleration'),('jerk','jerk')]},
                            progress=retimer._sample_progress[lane].detach().cpu().numpy().copy())
    summary['sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/name for name in ('native_input.npz','native_buffers.npz',
                              'rerun_interpolated.npz','native_641.npz','native_4097.npz','native_8193.npz')]}
    (out/'reconstruction.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    planner.destroy()


if __name__ == '__main__':
    main()
