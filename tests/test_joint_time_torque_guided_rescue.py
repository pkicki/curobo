# SPDX-License-Identifier: Apache-2.0
"""Offline tests: torque LP dual direction, matched search and native probe."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.joint_time_oracle.torque_dual_guidance import (
    normalized_bottlenecks, normalized_descent_direction, solve_point_lp)
from research.joint_time_oracle.torque_guided_rescue import (
    fit_nominal_quintic,local_support,normalized_step,random_direction,stable_seed)
from research.joint_time_oracle.native_pretimeopt_probe import NativePreTimeOptProbe
from research.joint_time_oracle.torque_guided_report import qualified


def test_lp_dual_constraint_sign_and_strong_gap():
    lo=-np.ones(7)*2
    hi=np.ones(7)*2
    bounds=np.stack((lo,hi))
    a=np.zeros(7); m=np.zeros(7)
    g=np.array([5.,0.,0.,0.,0.,0.,0.])
    r=solve_point_lp(a,m,g,bounds)
    assert r.status=="SOLVED"
    assert r.expansion==pytest.approx(1.5,abs=1e-9)
    assert r.dual_bound==pytest.approx(r.expansion,abs=1e-9)
    assert np.max(r.upper_dual)>0
    assert np.max(r.lower_dual)<1e-9
    assert r.primal_residual<1e-8
    ok=solve_point_lp(a,m,np.zeros(7),bounds)
    assert ok.status=="SOLVED"
    assert ok.expansion<1e-8


def test_competing_joint_constraints_can_make_lp_infeasible():
    lo=-np.ones(2)
    hi=np.ones(2)
    bounds=np.stack((lo,hi))
    g=np.array([3.,-3.])
    a=np.array([1.,1.])
    m=np.zeros(2)
    r=solve_point_lp(a,m,g,bounds)
    assert r.status=="SOLVED"
    assert r.expansion>=2.-1e-8
    assert r.dual_bound>1.


def test_bottleneck_ordering_and_fail_closed_shapes():
    bounds=np.stack((-np.ones(2),np.ones(2)))
    coeff=np.zeros((3,3,2))
    coeff[2,0]=[3.,0.]
    coeff[2,1]=[0.,0.]
    coeff[2,2]=[2.,0.]
    b=normalized_bottlenecks(coeff,bounds,top_k=3)
    assert [x["index"] for x in b]==[0,2,1]
    assert b[0]["expansion"]>b[1]["expansion"]>b[2]["expansion"]-1e-10
    with pytest.raises(ValueError):
        normalized_bottlenecks(coeff[:2],bounds)


def test_only_interior_quintic_controls_may_change():
    mask=local_support(22,.5)
    assert mask.shape==(22,)
    assert np.any(mask[3:-3])
    assert not np.any(mask[:3]) and not np.any(mask[-3:])
    rng=np.random.default_rng(4)
    d=random_direction(rng,(22,7),mask)
    np.testing.assert_array_equal(d[:3],0)
    np.testing.assert_array_equal(d[-3:],0)
    assert np.max(np.linalg.norm(d,axis=1))==pytest.approx(1.)
    noisy=np.ones((22,7))
    x=normalized_step(noisy)
    np.testing.assert_array_equal(x[:3],0)
    np.testing.assert_array_equal(x[-3:],0)
    assert stable_seed("task","dual",3)!=stable_seed("task","random_local",3)


class DummyOriginal:
    def evaluate_trimmed(self,s):
        s=np.asarray(s)
        curve=np.array([s, s**2, np.sin(s), s**3, s*.1, s*.2, s*.3]).T
        return (curve,)


def test_both_arms_receive_same_deterministic_initial_fit():
    initial,diag=fit_nominal_quintic(DummyOriginal(),
            np.zeros(7),
            np.array([1,1,np.sin(1.),1,.1,.2,.3]),
            count=22)
    assert initial.shape==(22,7)
    np.testing.assert_allclose(initial[:3],0.)
    assert diag["maximum_same_progress_fit_deviation_rad"]>=0
    np.testing.assert_allclose(initial[-3:],np.array(
        [1,1,np.sin(1.),1,.1,.2,.3])[None],atol=1e-12)


class FakeProcessor:
    def get_interpolation_input(self):
        x=torch.zeros((4,5,7))
        return SimpleNamespace(position=x,knot=torch.ones((4,3,7)),
                               knot_dt=torch.ones(4),velocity=x,
                               acceleration=x,jerk=x,dt=torch.ones(4))


class FakeOptimizer:
    def __init__(self):
        self._controls=torch.ones((4,8,7))
        self._start_boundary=torch.zeros((4,3,7))
        self._goal=torch.ones((4,7))
        self._success=torch.zeros(4,dtype=torch.bool)
        self._stationary=torch.zeros(4,dtype=torch.bool)


class FakeSolver:
    def __init__(self):
        self._result_processor=FakeProcessor()
        self._timeopt=SimpleNamespace(
            optimizer=FakeOptimizer(),
            validation_masks={"path_valid":torch.zeros(4,dtype=torch.bool),
                              "limits_and_dynamics_valid":torch.zeros(4,dtype=torch.bool)})
        self.calls=0

    def _finalize_result(self,current_state,num_seeds,return_seeds,seed_valid_mask=None):
        self.calls+=1
        return SimpleNamespace(success=torch.zeros(4,dtype=torch.bool))


def test_pre_timeopt_probe_records_failed_native_geometry_without_changing_result(tmp_path):
    solver=FakeSolver()
    planner=SimpleNamespace(trajopt_solver=solver)
    original=solver._finalize_result.__func__
    with NativePreTimeOptProbe(planner,tmp_path,"F000") as capture:
        out=solver._finalize_result(None,4,1)
        assert out.success.sum()==0
        assert solver.calls==1
        assert len(capture.attempts)==1
    assert solver._finalize_result.__func__ is original
    info=json.loads((tmp_path/"F000_pretimeopt_inventory.json").read_text())
    assert info[0]["state"]=="PRETIMEOPT_SNAPSHOT_SAVED"
    assert info[0]["finalizer_returned"]
    assert info[0]["native_geometry_lane_count"]==4
    assert info[0]["timeopt_validation_masks"]["path_valid"]==[False]*4
    with np.load(tmp_path/"F000_pretimeopt_000_native_geometry.npz") as z:
        assert z["controls"].shape==(4,8,7)


def test_probe_does_not_swallow_original_finalizer_failure(tmp_path):
    solver=FakeSolver()
    planner=SimpleNamespace(trajopt_solver=solver)
    def fail(*args,**kwargs):
        raise ValueError("native original failure")
    solver._finalize_result=fail
    with pytest.raises(ValueError,match="native original failure"):
        with NativePreTimeOptProbe(planner,tmp_path,"F004"):
            solver._finalize_result(None,4,1)
    assert solver._finalize_result is fail
    inventory=json.loads((tmp_path/"F004_pretimeopt_inventory.json").read_text())
    assert inventory[0]["finalizer_returned"] is False


def test_independent_receipt_required_before_reporting_rescue(tmp_path):
    arm=tmp_path
    case="test"
    path=arm/case
    path.mkdir()
    row={"candidate_witness_prescreen":True,
         "torque_profile_sha256":"p",
         "w_dense_sha256":"w"}
    okay,why=qualified(arm,case,row)
    assert not okay
    assert any("independent" in m for m in why)



def test_kinematic_spline_hull_certifies_easy_continuous_kinematics():
    from research.joint_time_oracle.torque_guided_kinematic_hull import certificate
    ctrl=np.zeros((14,7))
    ctrl[-3:]=.1
    ctrl[3:-3]=np.linspace(0,.1,8)[:,None]
    wide=lambda b: np.vstack((-np.ones(7)*b,np.ones(7)*b))
    limits={"position":wide(2),"velocity":wide(100),
            "acceleration":wide(1000),"jerk":wide(10000)}
    cert=certificate(ctrl,10.,limits)
    assert cert["all_kinematic_families_conservatively_verified"]
    assert "no torque" in cert["scope"]
    limits["velocity"]=np.stack((np.ones(7)*.01,np.ones(7)*100))
    restricted=certificate(ctrl,10.,limits)
    assert not restricted["per_family_pass"]["velocity"]


def test_kinematic_hull_fails_closed_for_wrong_joint_bounds():
    from research.joint_time_oracle.torque_guided_kinematic_hull import certificate
    ctrl=np.zeros((14,7))
    limits={key:np.vstack((-np.ones(7),np.ones(7))) for key in (
        "position","velocity","acceleration","jerk")}
    limits["jerk"]=np.zeros((2,7))
    with pytest.raises(ValueError,match="invalid jerk"):
        certificate(ctrl,2.,limits)
