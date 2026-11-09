# SPDX-License-Identifier: Apache-2.0
"""CPU numerical/fail-closed regression contracts for margin recovery and NaN."""
from __future__ import annotations

from types import SimpleNamespace
import json

import numpy as np
import pytest

from research.joint_time_oracle.torque_margin_core import (
    DILATION_FACTORS, TIGHTENING_FACTORS, family_bounds, immutable_json,
    pilot_limits, sampled_dilation_interval, sampled_max_violation, scale_torque, slowdown_states,
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



def test_sampled_dilation_interval_can_be_bounded_on_both_sides_by_gravity():
    # Torque requires a *minimum* amount of dynamic compensation.
    # Velocity requires a minimum slowdown, so valid rho is a narrow interval.
    states={"q":np.zeros((3,7)),"qd":np.ones((3,7))*1.5,
            "qdd":np.zeros((3,7)),"jerk":np.zeros((3,7)),
            "gravity":np.ones((3,7))*1.4}
    tau=np.zeros((3,7))
    bounds={key:np.stack((-np.ones(7)*limit,np.ones(7)*limit))
            for key,limit in (("position",2),("velocity",1),
                              ("acceleration",10),("jerk",10),("torque",1))}
    interval=sampled_dilation_interval(states,tau,bounds)
    assert interval["status"]=="SAMPLED_ADMISSIBLE_INTERVAL"
    # Torque: 1.4(1-z)<=.95 -> z>=.45/1.4.
    assert interval["z_interval"][0]==pytest.approx(.45/1.4)
    # Velocity: 1.5 sqrt(z)<=.95 -> z<=(.95/1.5)**2.
    assert interval["z_interval"][1]==pytest.approx((.95/1.5)**2)
    assert 1.5<interval["rho_min"]<interval["rho_max"]<2.
    states["qd"]*=2
    rejected=sampled_dilation_interval(states,tau,bounds)
    assert rejected["status"]=="EMPTY_INTERSECTION"


def test_native_phase_trace_preserves_numeric_buffers_with_checksum(tmp_path):
    trace=observe_optimizer(SimpleNamespace(optimizer=SimpleNamespace(
        _solver=SimpleNamespace(
            feasible=np.array([True]),numerical_ok=np.array([False]),
            output=np.array([[np.nan,.2]])),
        _input_valid=np.array([True]),_times=np.array([[0.,np.nan]]),
        _success=np.array([False]))),read_tensor=lambda a:a.copy())
    info=save_trace(tmp_path,"F000",trace)
    assert info["timeopt_phase_buffers_sha256"] is not None
    assert info["timeopt_phase_trace_sha256"] is not None
    with np.load(tmp_path/"F000_timeopt_phase_buffers.npz") as data:
        assert np.isnan(data["solver_output_timing"][0,0])
        assert np.isnan(data["integrated_times"][0,-1])
    # Same buffers are idempotent, altered observations must not overwrite.
    assert save_trace(tmp_path,"F000",trace)==info
    altered={**trace,"raw_buffers":{**trace["raw_buffers"],
                                   "solver_output_timing":[[1.,.2]]}}
    with pytest.raises(ValueError,match="snapshot changed"):
        save_trace(tmp_path,"F000",altered)



def test_independent_validator_source_sha_detects_post_audit_change(tmp_path):
    from research.joint_time_oracle.cohort_validate import source_sha256
    file=tmp_path/"r_dense.npz"
    np.savez_compressed(file,time=np.array([0.,1.]),q=np.zeros((2,7)))
    frozen=source_sha256(file)
    assert source_sha256(file)==frozen
    np.savez_compressed(file,time=np.array([0.,1.]),q=np.ones((2,7)))
    assert source_sha256(file)!=frozen


def test_sealed_r_speed_controls_refuses_changed_npz(tmp_path):
    from research.joint_time_oracle.torque_margin_retime import seal_r_timing
    from research.joint_time_oracle.torque_margin_core import sha256_file
    case="test_case"
    folder=tmp_path/case
    folder.mkdir()
    source={"source_files_sha256":{"p_timing_controls.npz":"P_SHA",
                                   "w_geometry.json":"W_SHA"}}
    np.savez_compressed(folder/"r_dense.npz",time=np.array([0.,1.]),
                        q=np.zeros((2,7)))
    np.savez_compressed(folder/"r_speed_controls.npz",
                        squared_speed_controls=np.ones(32))
    (folder/"r_registration.json").write_text(json.dumps({
        "r_sha256":sha256_file(folder/"r_dense.npz")
    }))
    receipt=seal_r_timing(tmp_path,case,source,method="dilation")
    assert receipt["r_speed_controls_sha256"]==sha256_file(
        folder/"r_speed_controls.npz")
    assert seal_r_timing(tmp_path,case,source,method="dilation")==receipt
    np.savez_compressed(folder/"r_speed_controls.npz",
                        squared_speed_controls=np.ones(32)*.8)
    with pytest.raises(ValueError,match="frozen"):
        seal_r_timing(tmp_path,case,source,method="dilation")
