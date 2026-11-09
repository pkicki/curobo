# SPDX-License-Identifier: Apache-2.0
"""CPU numerical contracts for prospective global-path LP screen comparison."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.joint_time_oracle.torque_global_bank import (
    bank_seed, proposal_controls, DEVELOPMENT_CASE, HOLDOUT_CASES,
)
from research.joint_time_oracle.torque_global_screen import (
    LP_REJECTION, positive_lp_at_points,
)
from research.joint_time_oracle.torque_global_report import collect
from research.joint_time_oracle.native_timeopt_boundary import (
    worst_violation, compare_native_mask,
)
from research.joint_time_oracle.native_pretimeopt_probe import NativePreTimeOptProbe
from research.joint_time_oracle.torque_global_freeze import verify


def test_lp_blind_global_bank_is_deterministic_and_preserves_endpoints():
    controls=np.zeros((22,7))
    controls[-3:]=np.array([.2,.3,.1,0,0,.1,-.2])
    x,log=proposal_controls(controls,count=128,seed=bank_seed("task",3))
    y,again=proposal_controls(controls,count=128,seed=bank_seed("task",3))
    assert x.shape==(128,22,7)
    np.testing.assert_array_equal(x,y)
    assert log==again
    np.testing.assert_array_equal(x[:,:3],np.broadcast_to(controls[:3],(128,3,7)))
    np.testing.assert_array_equal(x[:,-3:],np.broadcast_to(controls[-3:],(128,3,7)))
    assert np.max(np.abs(x[:,3:-3]))>.8
    assert set(r["mode"] for r in log)=={"independent","walk","two_lobe"}


def test_bank_seed_changes_for_different_cases():
    assert bank_seed("case1",27)!=bank_seed("case2",27)
    assert bank_seed("case1",27)==bank_seed("case1",27)


class FakeDynamics:
    def __init__(self,gravity):
        self.gravity=np.asarray(gravity)
        self.rnea_calls=0

    def coefficients(self,_geometry,_s):
        self.rnea_calls+=3
        return np.zeros_like(self.gravity),np.zeros_like(self.gravity),self.gravity


def test_necessary_lp_gate_rejects_inevitably_overloaded_path():
    r=FakeDynamics([4.,0.])
    lim=np.stack((-np.ones(2),np.ones(2)))
    row=positive_lp_at_points(r,object(),lim)
    assert row["decision"]=="LP_EXCLUDED"
    assert row["expansion"]>LP_REJECTION
    assert r.rnea_calls==3*row["evaluations"]


def test_necessary_lp_gate_zero_is_only_inconclusive():
    r=FakeDynamics([0.,0.])
    lim=np.stack((-np.ones(2),np.ones(2)))
    row=positive_lp_at_points(r,object(),lim,meshes=(9,17))
    assert row["decision"]=="LP_NOT_EXCLUDED"
    assert row["evaluations"]==26
    assert "necessary" in row["note"]


def test_post_state_mask_taxonomy_and_asymmetric_bounds():
    assert compare_native_mask(False,True).startswith("NATIVE_FAIL")
    assert compare_native_mask(True,False)=="NATIVE_PASS_INDEPENDENT_FAIL"
    assert compare_native_mask(None,True)=="UNKNOWN"
    limits=np.array([[-2.,-1.],[4.,2.]])
    assert worst_violation(np.array([[3.,0.]]),limits)==0
    assert worst_violation(np.array([[4.5,0.]]),limits)==pytest.approx(.5)
    assert worst_violation(np.array([[4.5,0.]]),limits,scale=.95,
                           centered=False)==pytest.approx(.7)


def test_frozen_holdout_unavailable_without_development(tmp_path):
    with pytest.raises(ValueError,match="before development freeze"):
        verify(tmp_path)


def test_report_leaves_missing_runs_unexecuted(tmp_path):
    data=collect(tmp_path)
    assert data["experiment_status"]=="INCOMPLETE"
    assert data["qualified_counts"]=={"no_lp":0,"lp_gate":0}
    assert len(data["rows"])==2*(1+len(HOLDOUT_CASES))
    assert data["decision"]=="UNEXECUTED_OR_INCOMPLETE"


class DummyProcessor:
    def get_interpolation_input(self):
        zeros=torch.zeros((2,9,7))
        return SimpleNamespace(position=zeros,velocity=zeros,acceleration=zeros,
                               jerk=zeros,knot=torch.zeros((2,9,7)),
                               knot_dt=torch.ones((2,9)),dt=torch.ones(2))


class DummyOptimizer:
    def __init__(self):
        self._controls=torch.zeros((2,8,7))
        self._start_boundary=torch.zeros((2,3,7))
        self._goal=torch.zeros((2,7))
        self._stationary=torch.zeros(2,dtype=torch.bool)
        self._success=torch.zeros(2,dtype=torch.bool)
        self._timing=torch.ones((2,9))
        self._times=torch.tensor([[0.,1.],[0.,1.]])
        self.sample_count=0

    def sample(self,grid):
        self.sample_count+=1
        base=torch.zeros((2,grid.shape[1],7))
        return SimpleNamespace(position=base,velocity=base,
                               acceleration=base,jerk=base)


class DummySolver:
    def __init__(self):
        self._result_processor=DummyProcessor()
        self._timeopt=SimpleNamespace(
            optimizer=DummyOptimizer(),
            validation_masks={"path_valid":torch.ones(2,dtype=torch.bool),
                              "collision_valid":torch.zeros(2,dtype=torch.bool),
                              "limits_and_dynamics_valid":torch.zeros(2,dtype=torch.bool)})
        self.calls=0

    def _finalize_result(self,*args):
        self.calls+=1
        return SimpleNamespace(success=torch.zeros(2,dtype=torch.bool))


def test_post_timeopt_probe_opt_in_preserves_native_return_and_saves_samples(tmp_path):
    solver=DummySolver()
    planner=SimpleNamespace(trajopt_solver=solver)
    original=solver._finalize_result.__func__
    with NativePreTimeOptProbe(planner,tmp_path,"F005",post_samples=65):
        out=solver._finalize_result(None,2,1)
        assert solver.calls==1
        assert not out.success.any()
    assert solver._finalize_result.__func__ is original
    inv=json.loads((tmp_path/"F005_pretimeopt_inventory.json").read_text())
    assert inv[0]["post_timeopt_samples"]==65
    with np.load(tmp_path/"F005_pretimeopt_000_post_timeopt_states.npz") as f:
        assert f["q"].shape==(2,65,7)
        np.testing.assert_array_equal(f["valid_duration_mask"],[True,True])
    assert solver._timeopt.optimizer.sample_count==1


def test_post_capture_invalid_count_rejected():
    with pytest.raises(ValueError):
        NativePreTimeOptProbe(SimpleNamespace(trajopt_solver=DummySolver()),
                              "/tmp/no_written", "F",post_samples=8)
