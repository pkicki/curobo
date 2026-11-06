# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot the saved final spline acceleration at audit and optimizer samples."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root/'research/joint_time_oracle/results/debug_long_20261106/per_task.json'
    row = json.loads(source.read_text())[0]
    controls = np.asarray(row['trials'][0]['coefficients'])
    duration = row['trials'][0]['duration']
    spline = BSpline(np.r_[np.zeros(5), np.linspace(0, 1, 6), np.ones(5)], controls, 5)
    fine = np.linspace(0, 1, 321)
    collocation = np.linspace(0, 1, 16)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(fine*duration, spline(fine, nu=2)[:, 6]/duration**2, label='dense audit samples')
    ax.scatter(collocation*duration, spline(collocation, nu=2)[:, 6]/duration**2,
               label='optimizer samples', color='black', s=22, zorder=3)
    ax.axhline(14.25, color='tab:orange', ls='--', label='95% scaled limit')
    ax.axhline(15, color='tab:red', ls=':', label='physical limit')
    ax.set(xlabel='Physical time (s)', ylabel='Joint 7 acceleration (rad/s²)',
           title='Original SLSQP result: acceleration peak between samples')
    ax.legend(loc='lower right'); fig.tight_layout()
    fig.savefig(root/'reports/joint_time_oracle/assets/acceleration.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
