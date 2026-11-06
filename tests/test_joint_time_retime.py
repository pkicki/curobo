# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checks for exact-geometry pointwise timing."""
import numpy as np
import pytest

pytest.importorskip('pinocchio')
pytest.importorskip('cyipopt')
from research.joint_time_oracle.cohort_geometry import QuinticGeometry
from research.joint_time_oracle.cohort_retime import problem, states


def test_timing_jacobian_and_path_preservation() -> None:
    """Timing reconstructs the original curve and has the expected pointwise Jacobian."""
    controls=np.zeros((8,7),dtype=np.float32)
    controls[3:5,0]=[.2,.7]
    controls[5:,0]=1.
    geometry=QuinticGeometry(controls)
    speed=np.ones(32,dtype=np.float32)*4
    p=problem(geometry.evaluate,np.unique(geometry.spline.t),33,speed)
    x=speed/p.scale;analytical=p.values_jac(x)[1]
    step=np.float32(.001);direction=np.zeros(32,dtype=np.float32);direction[12]=step
    numeric=(p.values_jac(x+direction)[0]-p.values_jac(x-direction)[0])/(2*step)
    assert np.max(np.abs(numeric-analytical[:,12]))<.002
    output=states(geometry.evaluate,speed,count=257)
    assert np.max(np.abs(output['q']-geometry.evaluate(output['progress'])[0]))<1e-7
    assert abs(output['time'][-1]-.5)<1e-5
