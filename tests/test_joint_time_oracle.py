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
        np.array([.1,.2],dtype=np.float32),np.array([1.,-.4],dtype=np.float32),
        np.array([.2,-.1],dtype=np.float32),np.array([.05,.03],dtype=np.float32),
        np.array([.3,.1],dtype=np.float32),np.array([-.2,.04],dtype=np.float32))
    basis=QuinticBasis(15,4,device=torch.device("cpu"))
    interior=torch.randn(basis.variable_count,2,requires_grad=True)
    T=torch.tensor(3.,requires_grad=True)
    q,v,a,jerk,controls=basis.evaluate(interior,T,boundary)
    for observed,expected in [(q[0],boundary.q0),(q[-1],boundary.q1),
                              (v[0],boundary.v0),(v[-1],boundary.v1),
                              (a[0],boundary.a0),(a[-1],boundary.a1)]:
        np.testing.assert_allclose(observed.detach().numpy(),expected,atol=2e-4)
    assert max(float(torch.max(torch.abs(observed.detach() - torch.as_tensor(expected))))
               for observed, expected in [(q[0], boundary.q0), (q[-1], boundary.q1),
                                          (v[0], boundary.v0), (v[-1], boundary.v1),
                                          (a[0], boundary.a0), (a[-1], boundary.a1)]) < 2e-5
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


def test_collision_slack_is_not_accepted_as_a_feasible_solution():
    cfg=OracleConfig(controls=10,samples_per_span=2,audit_samples_per_span=6,
                     maxiter=12,attempts=1,trust_radius=.6)
    # Positive cost represents collision at every trajectory sample.
    def impossible_collision(q):
        return q[:,0]*0 + 0.05
    solver=JointTimeOracle(
        limits(1),Boundary.rest(np.array([0.]),np.array([1.])),
        dynamics,collision=impossible_collision,config=cfg)
    z=solver._seed(np.linspace(0,1,40)[:,None],5.)
    assert not solver.audit(z)["feasible"]
    out=solver.solve(np.linspace(0,1,40)[:,None],5.)
    assert out["best"] is None
    assert out["global_lower_bound"] is None


def test_zero_collision_cost_allows_feasible_solution_and_gradients():
    cfg=OracleConfig(controls=10,samples_per_span=2,audit_samples_per_span=6,
                     maxiter=80,attempts=1,trust_radius=1.)
    def free_space(q):
        assert q.dtype == torch.float32
        return q[:,0]*0
    solver=JointTimeOracle(
        limits(1),Boundary.rest(np.array([0.]),np.array([1.])),
        dynamics,collision=free_space,config=cfg)
    z=solver._seed(np.linspace(0,1,50)[:,None],5.)
    jac=solver._evaluate(z,True)
    assert jac.shape==(6,len(z))
    assert jac[-1,-2]==pytest.approx(1.)
    assert solver.audit(z)["feasible"]


def test_constraint_jacobian_matches_central_differences():
    cfg = OracleConfig(controls=10, samples_per_span=3, audit_samples_per_span=6,
                       attempts=1, solver="ipopt", smooth_temperature=0.01)
    boundary = Boundary.rest(np.array([0.1, -0.2], dtype=np.float32),
                             np.array([0.8, 0.5], dtype=np.float32))
    def collision(q):
        return 0.03 * q[:, 0] + 0.01 * q[:, 1]
    solver = JointTimeOracle(limits(2), boundary, dynamics, collision, config=cfg)
    path = np.linspace(boundary.q0, boundary.q1, 45)
    z = solver._seed(path, 3.0)
    z[:-2] += np.random.default_rng(12).normal(0, 0.02, len(z) - 2)
    analytic = solver._evaluate(z, True)
    step = 2e-3
    numeric = np.empty_like(analytic)
    for i in range(len(z)):
        offset = np.zeros_like(z)
        offset[i] = step
        numeric[:, i] = (solver._evaluate(z + offset, False) -
                         solver._evaluate(z - offset, False)) / (2 * step)
    np.testing.assert_allclose(analytic, numeric, atol=4e-3, rtol=3e-2)
    objective = lambda x: np.exp(x[-1]) + 1000 * x[-2]
    objective_grad = np.r_[np.zeros(len(z) - 2), 1000., np.exp(z[-1])]
    numeric_obj = np.empty_like(z)
    for i in range(len(z)):
        offset = np.zeros_like(z)
        offset[i] = step
        numeric_obj[i] = (objective(z + offset) - objective(z - offset)) / (2 * step)
    np.testing.assert_allclose(objective_grad, numeric_obj, atol=1e-4, rtol=1e-5)
