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
    extra=np.array([.12345,.76543],dtype=np.float32)
    adaptive=states(geometry.evaluate,speed,count=257,extra=extra)
    assert np.all(np.isin(extra,adaptive['progress']))
    assert np.max(np.abs(adaptive['q']-geometry.evaluate(adaptive['progress'])[0]))<1e-7


def test_independent_sampled_acceptance_and_arc_alignment() -> None:
    """Sampled pair gates fail closed and compare paths independently of timing."""
    from research.joint_time_oracle.cohort_analyze import arc_samples, passed
    families={name:{'physical':{'violation':0.},'scaled95':{'violation':0.}}
              for name in ('position','velocity','acceleration','jerk','torque')}
    diag={'families':families,'world_min_clearance_m':None,'self_min_clearance_m':.1,'samples':8193,
          'endpoints':{'position_start':0.,'position_goal':0.,'velocity':0.,'acceleration':0.}}
    assert passed(diag,'physical')
    diag['families']['jerk']['scaled95']['violation']=.02
    assert passed(diag,'physical') and not passed(diag,'scaled95')
    q=np.stack([np.linspace(0,1,10,dtype=np.float32),np.zeros(10,dtype=np.float32)],axis=1)
    repeat=np.repeat(q,[1,2,3,1,1,2,3,2,1,1],axis=0)
    assert np.allclose(arc_samples(q),arc_samples(repeat),atol=1e-6)
