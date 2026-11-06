# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify exact geometric adapters preserve spline paths and derivatives."""
import numpy as np
import pytest

pytest.importorskip('pinocchio')
from research.joint_time_oracle.cohort_geometry import NativeGeometry, QuinticGeometry


def test_quintic_and_native_stationary_trim() -> None:
    """Constant endpoint spans can be removed without changing the moving geometry."""
    controls=np.zeros((1,6,2),dtype=np.float32)
    controls[0,:,0]=np.linspace(0,1,6,dtype=np.float32)
    buffers={'controls':controls,'start_boundary':np.zeros((1,4,2),dtype=np.float32),
             'goal':np.ones((1,2),dtype=np.float32)}
    geometry=NativeGeometry(buffers,0); start,end=geometry.nonstationary_interval()
    assert start>0 and end>start
    u=np.linspace(0,1,11,dtype=np.float32)
    trim=geometry.evaluate_trimmed(u); original=geometry.evaluate(start+(end-start)*u)
    assert np.allclose(trim[0],original[0])
    assert np.allclose(trim[1],original[1]*(end-start))
    quintic=QuinticGeometry(np.zeros((10,2),dtype=np.float32))
    assert all(np.max(np.abs(value)) == 0 for value in quintic.evaluate(u))
