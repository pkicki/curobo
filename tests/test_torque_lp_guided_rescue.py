# SPDX-License-Identifier: Apache-2.0
"""Math and stage-capture regression tests; no GPU required for LP tests."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from research.joint_time_oracle.torque_dual_guidance import (
    normalized_bottlenecks, normalized_descent_direction, solve_point_lp,
)


def test_point_lp_dual_bound_and_signed_marginal():
    # Gravity exceeds a physical actuator bound and a=m=0. Retiming cannot help.
    bounds=np.array([[-2.,-2.],[2.,2.]])
    a=m=np.zeros(2)
    g=np.array([4.,0.])
    r=solve_point_lp(a,m,g,bounds)
    assert r.status=="SOLVED"
    assert r.expansion==pytest.approx(1.,abs=1e-9)
    assert r.dual_bound==pytest.approx(1.,abs=1e-9)
    assert r.duality_gap==pytest.approx(0.,abs=1e-8)
    assert r.primal_residual<1e-8 and r.dual_residual<1e-8
    assert r.upper_dual[0]>0 and r.lower_dual[0]==pytest.approx(0.)
    # Dual *signed* multiplier gives envelope derivative d epsilon/d gravity.
    h=1e-4
    rp=solve_point_lp(a,m,g+np.array([h,0]),bounds)
    rm=solve_point_lp(a,m,g-np.array([h,0]),bounds)
    derivative=(rp.expansion-rm.expansion)/(2*h)
    assert derivative==pytest.approx((r.upper_dual-r.lower_dual)[0],abs=1e-6)


def test_lp_dual_detects_path_aware_cancellation():
    bounds=np.array([[-2.,-2.],[2.,2.]])
    # Independent acceleration can remove a one-joint gravitational excess.
    free=solve_point_lp(np.array([1.,0.]),np.zeros(2),
                        np.array([4.,0.]),bounds)
    assert free.status=="SOLVED" and free.expansion<1e-8
    # Opposing acceleration coupling makes the joint pair impossible.
    blocked=solve_point_lp(np.array([1.,1.]),np.zeros(2),
                           np.array([4.,-4.]),bounds)
    assert blocked.status=="SOLVED" and blocked.expansion>.9
    assert blocked.dual_bound>.9


def test_lp_bottleneck_ordering_and_invalid_shape():
    bounds=np.array([[-2.,-2.],[2.,2.]])
    coeff=np.zeros((3,3,2))
    coeff[2,:,0]=[0.,5.,3.]
    rows=normalized_bottlenecks(coeff,bounds,top_k=3)
    assert [x["index"] for x in rows]==[1,2,0]
    assert rows[0]["expansion"]>rows[1]["expansion"]>rows[2]["expansion"]
    with pytest.raises(ValueError):
        normalized_bottlenecks(np.zeros((2,3,2)),bounds)


def test_gradient_descent_normalization_fails_closed():
    g=np.zeros((14,7))
    g[4,2]=1.0
    d=normalized_descent_direction(g)
    assert d[4,2]==pytest.approx(-1.)
    assert np.count_nonzero(d[:3])==0
    assert np.count_nonzero(d[-3:])==0
    assert normalized_descent_direction(np.zeros((14,7))) is None
    bad=g.copy();bad[4,2]=np.nan
    assert normalized_descent_direction(bad) is None


def test_native_stage_recorder_snapshots_failed_pretimeopt(monkeypatch,tmp_path):
    torch=pytest.importorskip("torch")
    from research.joint_time_oracle.torque_native_stage_trace import FinalizeRecorder

    class Processor:
        def get_interpolation_input(self):
            return SimpleNamespace(
                position=torch.zeros((4,7,7)),
                knot=torch.zeros((4,7,7)),
                knot_dt=torch.ones((4,)),
            )

    class TimeOpt:
        def __init__(self):
            self.optimizer=SimpleNamespace(
                _controls=torch.zeros((4,12,7)),
                _start_boundary=torch.zeros((4,3,7)),
                _goal=torch.zeros((4,7)),
            )
            self.validation_masks={"path_valid":torch.zeros((4,),dtype=torch.bool)}

    class Solver:
        def __init__(self):
            self._result_processor=Processor()
            self._timeopt=TimeOpt()
        def _finalize_result(self):
            return SimpleNamespace(
                success=torch.zeros((4,),dtype=torch.bool),
                debug_info={"validation":{"collision_valid":
                                          torch.zeros((4,),dtype=torch.bool)}})

    solver=Solver()
    original=solver._finalize_result
    planner=SimpleNamespace(trajopt_solver=solver)
    with FinalizeRecorder(planner,tmp_path) as trace:
        out=solver._finalize_result()
        assert not bool(out.success.any())
    assert solver._finalize_result==original
    assert len(trace.events)==1
    assert trace.events[0]["pre_status"]=="RECORDED"
    assert "post_path_buffers" in trace.events[0]
    with np.load(tmp_path/"pre_timeopt_000.npz") as f:
        assert f["position"].shape==(4,7,7)
    with np.load(tmp_path/"post_timeopt_000.npz") as f:
        assert f["controls"].shape==(4,12,7)


def test_shared_quintic_fit_and_support_are_endpoint_preserving():
    pytest.importorskip("pinocchio")
    from research.joint_time_oracle.torque_guided_rescue import (
        fit_nominal_quintic, local_support, normalized_step,
        random_direction, score_metrics,
    )
    class FakeNative:
        def evaluate_trimmed(self,s):
            s=np.asarray(s)
            q0=np.array([0.,-.5,0.,-2.,0.,1.5,0.])
            q1=q0+np.array([.4,.2,-.3,.2,.1,-.2,.1])
            q=q0[None,:]+s[:,None]*(q1-q0)[None,:]
            return (q,)
    fake=FakeNative()
    q0=fake.evaluate_trimmed(np.array([0.]))[0][0]
    q1=fake.evaluate_trimmed(np.array([1.]))[0][0]
    controls,summary=fit_nominal_quintic(fake,q0,q1,count=22)
    np.testing.assert_allclose(controls[:3],q0)
    np.testing.assert_allclose(controls[-3:],q1)
    assert summary["maximum_same_progress_fit_deviation_rad"]<.2
    support=local_support(22,.5)
    assert support.shape==(22,) and support.sum()>0
    assert not support[:3].any() and not support[-3:].any()
    rng=np.random.default_rng(13)
    delta=random_direction(rng,controls.shape,support)
    np.testing.assert_allclose(delta[:3],0)
    np.testing.assert_allclose(delta[-3:],0)
    assert np.max(np.linalg.norm(delta,axis=1))==pytest.approx(1.)
    assert score_metrics(.2,.70,{"position_violation_rad":0,
            "world_clearance_m":.1,"self_clearance_m":.1},.3)==pytest.approx(.2)
    assert score_metrics(.3,.7,{"position_violation_rad":0,
            "world_clearance_m":.1,"self_clearance_m":.1},.3)>.2
