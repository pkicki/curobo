# SPDX-License-Identifier: Apache-2.0
"""CPU numerical/fail-closed regression contracts for margin recovery and NaN."""
from __future__ import annotations

from types import SimpleNamespace
import json

import numpy as np
import pytest

from research.joint_time_oracle.torque_margin_core import (
    DILATION_FACTORS, TIGHTENING_FACTORS, family_bounds, immutable_json,
    pilot_limits, sampled_max_violation, scale_torque, slowdown_states,
    tighten_pointwise_problem,
)
from research.joint_time_oracle.native_timeopt_nan_trace import (
    numeric_summary, per_lane_diagnosis, observe_optimizer, save_trace,
)
from research.joint_time_oracle.torque_margin_report import collect


def sample_states():
    t=np.linspace(0,1,129)
    q=np.stack((t,t**2),axis=1)
    return {"time":t,"progress":t,"q":q,
            "qd":np.ones_like(q)*.4,"qdd":np.ones_like(q)*.3,
            "jerk":np.ones_like(q)*.2}


def test_dilation_exact_chain_rule_and_geometry():
    original=sample_states()
    changed=slowdown_states(original,2.)
    np.testing.assert_array_equal(changed["q"],original["q"])
    np.testing.assert_array_equal(changed["progress"],original["progress"])
    np.testing.assert_allclose(changed["time"],2*original["time"])
    np.testing.assert_allclose(changed["qd"],original["qd"]/2)
    np.testing.assert_allclose(changed["qdd"],original["qdd"]/4)
    np.testing.assert_allclose(changed["jerk"],original["jerk"]/8)
    assert original["time"][-1]==pytest.approx(1.)
    assert DILATION_FACTORS==tuple(sorted(DILATION_FACTORS))


def test_dilation_rejects_invalid_inputs():
    s=sample_states()
    with pytest.raises(ValueError,match="rho"):
        slowdown_states(s,.8)
    s["time"][12]=np.nan
    with pytest.raises(ValueError,match="nonfinite"):
        slowdown_states(s,1.05)


def test_torque_gravity_asymptote_and_nonmonotone_feasibility():
    # P torque is feasible, but slowing towards high gravity becomes infeasible.
    tau=np.array([[-6.,3.]])
    gravity=np.array([[10.,3.]])
    np.testing.assert_allclose(scale_torque(tau,gravity,1),tau)
    np.testing.assert_allclose(scale_torque(tau,gravity,2),[[6.,3.]])
    np.testing.assert_allclose(scale_torque(tau,gravity,1e6),gravity,rtol=1e-11)
    lim=np.array([[-7.,-4.],[5.,4.]])
    assert sampled_max_violation(tau,lim)["violation"]==0
    assert sampled_max_violation(scale_torque(tau,gravity,2),lim)["violation"]>0


def test_torque_bounds_scale_toward_zero_and_asymmetric_kinematics():
    asym=np.array([[-10.,-5.],[20.,15.]])
    tau=family_bounds(asym,"torque",scaled95=True)
    np.testing.assert_allclose(tau,asym*.95)
    vel=family_bounds(asym,"velocity",scaled95=True)
    center=asym.mean(axis=0)
    np.testing.assert_allclose(vel[0],center+.95*(asym[0]-center))
    np.testing.assert_allclose(vel[1],center+.95*(asym[1]-center))
    np.testing.assert_array_equal(family_bounds(asym,"position",scaled95=True),asym)


def test_pilot_gate_respects_family_tolerances_not_source_ipopt_status():
    s=sample_states()
    s["gravity"]=np.zeros((len(s["time"]),2))
    bounds={name:np.array([[-1.,-1.],[1.,1.]]) for name in
            ("position","velocity","acceleration","jerk","torque")}
    s["q"]*=.5
    z=pilot_limits(s,np.zeros((len(s["time"]),2)),bounds,1.1)
    assert z["conservative_screen_pass"] is True
    assert z["duration_s"]==pytest.approx(1.1)


def test_finite_residual_checks_fail_on_nonfinite_or_wrong_shape():
    limits=np.array([[-1.,-2.],[1.,2.]])
    r=sampled_max_violation(np.array([[1.2,0.]]),limits)
    assert r["side"]=="upper" and r["joint"]==1
    assert r["violation"]==pytest.approx(.2)
    with pytest.raises(ValueError):
        sampled_max_violation(np.array([[np.nan,0.]]),limits)


def test_tightening_scales_torque_center_and_width_around_zero():
    p=SimpleNamespace(center=np.zeros((4,7)),width=np.ones((4,7)))
    p.center[3]=3.
    tighten_pointwise_problem(p,.94)
    np.testing.assert_allclose(p.center[:3],0)
    np.testing.assert_allclose(p.center[3],2.82)
    np.testing.assert_allclose(p.width,.94)
    assert TIGHTENING_FACTORS==tuple(sorted(TIGHTENING_FACTORS,reverse=True))
    with pytest.raises(ValueError):
        tighten_pointwise_problem(p,1.001)


def test_immutable_json_refuses_changed_protocol(tmp_path):
    path=tmp_path/"frozen.json"
    immutable_json(path,{"seed":5})
    immutable_json(path,{"seed":5})
    with pytest.raises(ValueError,match="frozen"):
        immutable_json(path,{"seed":6})
    assert json.loads(path.read_text())=={"seed":5}


def test_native_numeric_summary_detects_nan_and_inf():
    a=np.array([[np.nan,1],[np.inf,-np.inf]])
    result=numeric_summary(a)
    assert result["status"]=="NONFINITE"
    assert result["finite"]==1 and result["nan"]==1
    assert result["posinf"]==1 and result["neginf"]==1
    assert numeric_summary(np.array([True,False]))["false"]==1


def test_first_observed_invalid_native_stage_classification():
    snap={
        "input_valid":np.array([True,True,False]),
        "solver_seed_feasible":np.array([True,False,False]),
        "solver_numerical_ok":np.array([False,False,True]),
        "solver_output_timing":np.array([[np.nan,1],[1.,1.],[2.,1.]]),
        "finish_optimization_ok":np.array([False,False,False]),
        "integrated_times":np.array([[np.nan],[np.nan],[np.nan]])}
    result=per_lane_diagnosis(snap)
    assert result[0]["first_observed_problem"]=="solver_numerical_ok_FALSE"
    assert result[1]["first_observed_problem"]=="solver_seed_feasible_FALSE"
    assert result[2]["first_observed_problem"]=="input_valid_FALSE"


def test_stage_probe_is_observational_and_writes_sha_safe_file(tmp_path):
    numerical=SimpleNamespace(feasible=np.array([True,False]),
          numerical_ok=np.array([False,True]),
          output=np.array([[np.nan,.3],[.2,.2]]),
          uniform_duration=np.array([1.,1.]),
          seed_bounds=np.zeros((2,2,3)),
          output_bounds=np.zeros((2,2,3)))
    opt=SimpleNamespace(_solver=numerical,_input_valid=np.array([True,True]),
          _optimization_ok=np.array([False,False]),
          _timing=np.array([[np.nan,0],[.2,.3]]),
          _uniform_duration=np.array([1.,1.]),
          _times=np.array([[0.,np.nan],[0.,np.nan]]),
          _success=np.array([False,False]),
          _status=np.array([1,2]))
    wrapper=SimpleNamespace(optimizer=opt)
    result=observe_optimizer(wrapper,read_tensor=lambda a:a.copy())
    assert result["summary"]["integrated_times"]["status"]=="NONFINITE"
    assert len(result["per_lane"])==2
    saved=save_trace(tmp_path,"F000",result)
    assert len(saved["first_observed_issues"])==2
    assert (tmp_path/"F000_timeopt_phase_trace.json").exists()
    with pytest.raises(ValueError):
        save_trace(tmp_path,"F000",{"modified":True})


def test_margin_report_is_explicitly_unexecuted_when_no_source(tmp_path):
    out=collect(tmp_path)
    assert out["status"]=="NOT_EXECUTED"
    assert out["cases"]==[]
    assert out["native_trace"]["status"]=="NOT_EXECUTED"
