# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression checks for the fixed-path pointwise timing derivatives."""
from pathlib import Path
import numpy as np
from research.joint_time_oracle.fixed_path_pointwise import PointwiseTiming


def test_pointwise_jacobian_and_objective() -> None:
    """Pointwise derivatives agree with central differences at saved controls."""
    directory = Path(__file__).resolve().parents[1] / 'research/joint_time_oracle/results/native_reconstruction_20261106'
    problem = PointwiseTiming(directory, stride=8)
    x = (problem.initial / problem.scale).astype(np.float32)
    values, jacobian = problem.values_jac(x)
    assert len(values) == 2 * 4 * len(problem.grid) * 7
    for column in (0, 12, 31):
        direction = np.zeros(32, dtype=np.float32)
        direction[column] = .001
        finite = (problem.values_jac(x + direction)[0] - problem.values_jac(x - direction)[0]) / .002
        assert np.max(np.abs(finite - jacobian[:, column])) < .005
        objective_fd = (problem.objective(x + direction) - problem.objective(x - direction)) / .002
        assert abs(objective_fd - problem.gradient(x)[column]) < .0001
