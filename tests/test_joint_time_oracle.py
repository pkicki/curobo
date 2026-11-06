# SPDX-License-Identifier: Apache-2.0
"""CPU-only mathematical smoke tests. Real cuRobo CUDA qualification is separate."""
import numpy as np
import pytest
import torch

from research.joint_time_oracle.oracle import (
    Boundary, JointLimits, JointTimeOracle, OracleConfig, QuinticBasis,
)


def limits(n):
    def pair(v):
        return np.stack([-np.ones(n)*v, np.ones(n)*v])
    return JointLimits(pair(2),pair(3),pair(15),pair(200),pair(25))


def dynamics(q, qd, qdd):
    return 1.5*qdd + .2*qd + .05*q


def test_endpoint_constraints_hold_with_moving_boundaries():
    boundary=Boundary(
        np.array([.1,.2]),np.array([1.,-.4]),np.array([.2,-.1]),
        np.array([.05,.03]),np.array([.3,.1]),np.array([-.2,.04]))
    basis=QuinticBasis(15,4,device=torch.device("cpu"))
    interior=torch.randn(basis.variable_count,2,requires_grad=True)
    T=torch.tensor(3.,requires_grad=True)
    q,v,a,jerk,controls=basis.evaluate(interior,T,boundary)
    for observed,expected in [(q[0],boundary.q0),(q[-1],boundary.q1),
                              (v[0],boundary.v0),(v[-1],boundary.v1),
                              (a[0],boundary.a0),(a[-1],boundary.a1)]:
        np.testing.assert_allclose(observed.detach().numpy(),expected,atol=2e-4)
    q.square().sum().backward()
    assert torch.isfinite(interior.grad).all()
    assert torch.isfinite(T.grad)


def test_slow_seed_can_be_improved_with_joint_free_time_optimization():
    cfg=OracleConfig(controls=10,samples_per_span=2,audit_samples_per_span=8,
                     maxiter=90,attempts=1,ftol=1e-6,min_duration=.1,
                     max_duration=20,trust_radius=1.,constraint_tol=.005)
    solver=JointTimeOracle(limits(1),Boundary.rest(np.array([0.]),np.array([1.])),
                           dynamics,config=cfg)
    seed=np.linspace(0,1,100)[:,None]
    assert solver.audit(solver._seed(seed,5.))["feasible"]
    result=solver.solve(seed,5.)
    assert result["status"]=="LOCAL_FEASIBLE",result["trials"]
    assert result["best"]["duration"]<5.
    assert result["best"]["max_violation"]<=.0051
    assert result["global_lower_bound"] is None


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        OracleConfig(controls=7)
    with pytest.raises(ValueError):
        JointTimeOracle(limits(1),Boundary.rest(np.array([0.]),np.array([1.,2.])),dynamics)
    solver=JointTimeOracle(limits(1),Boundary.rest(np.array([0.]),np.array([1.])),dynamics)
    with pytest.raises(ValueError):
        solver._seed(np.zeros((15,1)),0.)
