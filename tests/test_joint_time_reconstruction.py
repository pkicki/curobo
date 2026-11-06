# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression checks for the frozen TimeOpt reconstruction evidence."""
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip('pinocchio')
from research.joint_time_oracle.fixed_path_control import path_at


DATA = Path(__file__).resolve().parents[1] / 'research/joint_time_oracle/results/native_reconstruction_20261106'


def test_native_path_matches_retained_sampler_points() -> None:
    buffers = np.load(DATA/'native_buffers.npz')
    sample = np.load(DATA/'native_8193.npz')
    knots = buffers['controls'].shape[1]
    valid = (sample['progress'] > 0) & (sample['progress'] < 1)
    s = (sample['progress'][valid]*(knots+2)+1)/(knots+3)
    position = path_at(s, buffers, 0)[0]
    np.testing.assert_allclose(position, sample['q'][valid], rtol=0, atol=2e-5)


def test_native_dense_replay_matches_shared_samples() -> None:
    first = np.load(DATA/'native_8193.npz')
    replay = np.load(DATA/'native_16385_replay.npz')
    for name in ('q','qd','qdd','jerk'):
        np.testing.assert_array_equal(first[name], replay[name][::2])
