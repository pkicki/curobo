# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for torque decomposition and native full-planner selection."""
from __future__ import annotations

import numpy as np
import pytest

from research.joint_time_oracle.torque_mechanisms import (
    common_physical_times, decompose_rnea, metrics,
)
from research.joint_time_oracle.torque_full_select import best_native, summary


def fake_rnea(q,v,a):
    return (2.+q)*a + .3*v**2 + np.sin(q)


def profile():
    return {"profile_sha256":"p"*64,
            "effort_lower":[-10.,-10.],
            "effort_upper":[10.,10.]}


def valid_diag():
    return {"samples":8193,"torque_profile_sha256":"p"*64,
            "world_min_clearance_m":.01,"self_min_clearance_m":.02,
            "endpoints":{"position_start":0.,"position_goal":0.,
                         "velocity":0.,"acceleration":0.},
            "families":{name:{"physical":{"violation":0.},
                              "scaled95":{"violation":0.}}
                        for name in ("position","velocity","acceleration","jerk","torque")}}


def test_rnea_decomposition_exact_toy_model():
    q=np.array([[.1,.2],[.3,-.4],[.6,.0]])
    v=np.array([[1.,-2.],[-1.,2.],[3.,.5]])
    a=np.array([[2.,3.],[-1.,.5],[.4,-.8]])
    terms=decompose_rnea(q,v,a,fake_rnea)
    np.testing.assert_allclose(terms["inertia"],(2.+q)*a,rtol=0,atol=1e-14)
    np.testing.assert_allclose(terms["velocity_bias"],.3*v**2,rtol=0,atol=1e-14)
    np.testing.assert_allclose(terms["gravity"],np.sin(q),rtol=0,atol=1e-14)
    assert np.max(np.abs(terms["identity_residual"]))<1e-13


def test_decomposition_rejects_invalid_shapes():
    with pytest.raises(ValueError,match="matching"):
        decompose_rnea(np.zeros((3,2)),np.zeros((2,2)),np.zeros((3,2)),fake_rnea)


def test_common_time_reconstruction_and_strict_timestamps():
    t=np.array([0.,.25,1.])
    q=np.column_stack((t,t*t))
    traj={"time":t,"q":q,"qd":q*2,"qdd":q*0}
    dense=common_physical_times(traj,count=5)
    assert dense["q"].shape==(5,2)
    assert dense["time"][-1]==pytest.approx(1.)
    assert dense["normalized_time"][2]==pytest.approx(.5)
    with pytest.raises(ValueError,match="strictly increasing"):
        common_physical_times({**traj,"time":np.array([0.,0.,1.])})


def test_metrics_uses_asymmetric_signed_side():
    q=np.array([[0.,0.],[.1,.2],[.2,.3]])
    terms={"total":np.array([[-3.,2.],[-8.,3.],[-1.,2.]]),
           "gravity":np.zeros((3,2)),
           "inertia":np.array([[-3.,2.],[-8.,3.],[-1.,2.]]),
           "velocity_bias":np.zeros((3,2)),
           "identity_residual":np.zeros((3,2))}
    bounds=np.array([[-10.,-20.],[20.,10.]])
    m=metrics(terms,bounds,q,.5)
    assert m["max_torque_utilization"]==pytest.approx(.8)
    assert m["max_torque_joint"]==1
    assert m["max_torque_Nm"]==pytest.approx(-8.)
    assert m["max_torque_normalized_time"]==pytest.approx(.5)


def test_native_selector_filters_invalid_and_picks_best_verified():
    p=profile()
    c={
        "F000":{"status":"SUCCESS","duration_s":.41,"torque_profile_sha256":p["profile_sha256"]},
        "F001":{"status":"SUCCESS","duration_s":.30,"torque_profile_sha256":p["profile_sha256"]},
        "F002":{"status":"SUCCESS","duration_s":.40,"torque_profile_sha256":"incorrect"},
        "F003":{"status":"FAILED"}}
    a=valid_diag()
    b=valid_diag()
    b["families"]["torque"]["physical"]["violation"]=.1
    diags={"task:F000":a,"task:F001":b,"task:F002":a}
    receipts={"task:F000":{"physical":True,"scaled95":True,
                             "endpoint":True,"collision":True,
                             "torque_profile_sha256":p["profile_sha256"]}}
    assert best_native(c,diags,{},p,"task",1,scaled95=False,strict=False)["method"]=="F000"
    assert best_native(c,diags,{},p,"task",4,scaled95=False,strict=False)["method"]=="F000"
    assert best_native(c,diags,{},p,"task",4,scaled95=False,strict=True) is None
    assert best_native(c,diags,receipts,p,"task",4,scaled95=False,strict=True)["method"]=="F000"


def test_signed_summary_tracks_negative_improvements():
    x=summary([-4.,2.,5.],attempted=5)
    assert x["paired_n"]==3 and x["attempted_n"]==5
    assert x["median_pct"]==pytest.approx(2.)
    assert x["slower_count"]==1
    assert x["improved_gt_1pct"]==2
    assert x["paired_bootstrap95_median_pct"] is not None



def test_internal_native_seed_lanes_count_within_same_restart_budget():
    p=profile()
    candidates={
        "F000":{"status":"SUCCESS","duration_s":.45,"restart":0,
                "torque_profile_sha256":p["profile_sha256"]},
        "F000L02":{"status":"SUCCESS","duration_s":.40,"restart":0,
                   "torque_profile_sha256":p["profile_sha256"]},
        "F001":{"status":"SUCCESS","duration_s":.30,"restart":1,
                "torque_profile_sha256":p["profile_sha256"]}}
    diagnostics={"task:"+name:valid_diag() for name in candidates}
    first=best_native(candidates,diagnostics,{},p,"task",1,scaled95=False,strict=False)
    assert first["method"]=="F000L02"
    first_two=best_native(candidates,diagnostics,{},p,"task",2,scaled95=False,strict=False)
    assert first_two["method"]=="F001"
