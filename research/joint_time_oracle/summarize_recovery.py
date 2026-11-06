# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize saved samples; do not treat this as continuous-time certification."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root/'research/joint_time_oracle/results/diagnosis_20261106'
    original = np.load(root/'research/joint_time_oracle/results/debug_long_20261106/task_0000_fast.npz')
    summaries = {}
    for name, file in [('baseline', 'baseline_samples.npz'), ('recovery', 'recovery_dense.npz')]:
        data = np.load(out/file)
        row = {'samples': len(data['q']), 'duration_s': float(data['time'][-1]),
               'joint_space_path_length_rad': float(np.linalg.norm(np.diff(data['q'], axis=0), axis=1).sum())}
        for label, key, bound in [('position', 'q', 'position'), ('velocity', 'qd', 'velocity'),
                                  ('acceleration', 'qdd', 'acceleration'), ('jerk', 'jerk', 'jerk'),
                                  ('torque', 'tau_curobo', 'torque')]:
            values = data[key]
            low, high = data[bound+'_bounds']
            residual = np.maximum(low-values, values-high)
            index, joint = np.unravel_index(np.argmax(residual), residual.shape)
            row[label] = {'max_physical_violation': float(max(0.0, residual[index, joint])),
                          'max_residual': float(residual[index, joint]),
                          'joint': int(joint+1), 'time_s': float(data['time'][index]),
                          'value': float(values[index, joint]),
                          'lower_limit': float(low[joint]), 'upper_limit': float(high[joint])}
        row['endpoint_max_abs_residuals'] = {
            'q0': float(np.max(np.abs(data['q'][0]-original['q'][0]))),
            'q1': float(np.max(np.abs(data['q'][-1]-original['q'][-1]))),
            'v0': float(np.max(np.abs(data['qd'][0]-original['qd'][0]))),
            'v1': float(np.max(np.abs(data['qd'][-1]-original['qd'][-1]))),
            'a0': float(np.max(np.abs(data['qdd'][0]-original['qdd'][0]))),
            'a1': float(np.max(np.abs(data['qdd'][-1]-original['qdd'][-1])))}
        summaries[name] = row
    summaries['status'] = ('Unqualified comparison: baseline represented by 18 stored samples; '
                           'fixed-path TimeOpt control and complete derivative qualification missing.')
    (out/'physical_summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
