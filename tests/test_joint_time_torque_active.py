# SPDX-License-Identifier: Apache-2.0
"""CPU-only reproducibility and fail-closed tests for torque-stress profiles."""
from __future__ import annotations

import json

import numpy as np
import pytest

from research.joint_time_oracle.torque_active import (
    canonical_hash, make_profile, read_profile, torque_utilization, write_profile,
)
from research.joint_time_oracle.cohort_torque_report import sampled_acceptance


def fixture_profile(target=0.9, gravity=None):
    lim = np.array([[-10., -20.], [20., 40.]])
    reference = np.array([[8., -12.], [-1., 4.]])
    if gravity is None:
        gravity = np.array([[4., -5.], [2., -4.]])
    profile = make_profile(
        case_id="moderate::source_001", source_case_id="source_001",
        regime="moderate", nominal_bounds=lim, baseline_torques=reference,
        endpoint_gravity_torques=gravity, target=target,
        source_task_sha256="s"*64, source_torque_sha256="t"*64)
    return profile


def test_signed_asymmetric_utilization():
    bounds=np.array([[-10.,-20.],[20.,40.]])
    x=np.array([[10.,-10.],[-5.,20.]])
    assert np.allclose(torque_utilization(x,bounds),
                       np.array([[.5,.5],[.5,.5]]))


def test_moderate_calibration_uses_only_a40_and_endpoint_gravity():
    p=fixture_profile()
    assert p.admitted
    assert p.derate == pytest.approx(2./3.)
    assert p.expected_reference_utilization == pytest.approx(.9)
    assert p.endpoint_gravity_utilization == pytest.approx(.25)
    assert p.effort_lower == pytest.approx([-10*p.derate,-20*p.derate])


def test_severe_baseline_is_intentionally_torque_infeasible():
    p=fixture_profile(target=1.1)
    assert p.admitted
    assert p.expected_reference_utilization == pytest.approx(1.1)
    assert p.derate < fixture_profile().derate


def test_endpoint_static_reserve_rejects_unsound_profile():
    gravity=np.array([[19.,0.], [19.,0.]])
    p=fixture_profile(target=1.1,gravity=gravity)
    assert not p.admitted
    assert p.rejection_reason is not None
    assert "static" in p.rejection_reason


def test_profiles_are_checksummed_and_tampering_fails(tmp_path):
    p=fixture_profile()
    path=tmp_path/"torque_profile.json"
    write_profile(path,p)
    loaded=read_profile(path)
    assert loaded["profile_sha256"] == p.to_json()["profile_sha256"]
    x=json.loads(path.read_text());x["derate"]=0.3
    path.write_text(json.dumps(x))
    with pytest.raises(ValueError,match="checksum"):
        read_profile(path)


def test_unadmitted_profile_cannot_be_loaded(tmp_path):
    bad=fixture_profile(target=1.1,gravity=np.array([[19.,0.],[19.,0.]]))
    path=tmp_path/"rejected.json"
    write_profile(path,bad)
    with pytest.raises(ValueError,match="not admitted"):
        read_profile(path)


def test_canonical_hash_reproducible():
    assert canonical_hash({"a":2,"b":3})==canonical_hash({"b":3,"a":2})


def test_sampled_acceptance_requires_matching_profile_and_dense_check():
    p=fixture_profile().to_json()
    d={"samples":8193,"torque_profile_sha256":p["profile_sha256"],
       "world_min_clearance_m":.01,"self_min_clearance_m":.03,
       "endpoints":{"position_start":1e-8,"position_goal":1e-8,
                    "velocity":0.,"acceleration":0.},
       "families":{name:{
           "physical":{"violation":0.},
           "scaled95":{"violation":0.}} for name in
           ("position","velocity","acceleration","jerk","torque")}}
    assert sampled_acceptance(d,p,scaled=False)[0]
    assert sampled_acceptance(d,p,scaled=True)[0]
    d["samples"]=200
    assert not sampled_acceptance(d,p,scaled=False)[0]
    d["samples"]=8193
    d["families"]["torque"]["scaled95"]["violation"]=0.1
    assert sampled_acceptance(d,p,scaled=False)[0]
    assert not sampled_acceptance(d,p,scaled=True)[0]
    d["torque_profile_sha256"]="incorrect"
    assert not sampled_acceptance(d,p,scaled=False)[0]


def test_reject_nonfinite_torques():
    with pytest.raises(ValueError,match="nonfinite"):
        torque_utilization(np.array([float("nan"),1.]), np.array([[-2.,-2.],[2.,2.]]))


def test_asymmetric_torque_scaled_about_zero_like_native_timeopt():
    import torch
    from research.joint_time_oracle.oracle import normalized_worst

    bounds=torch.tensor([[-10.],[20.]],dtype=torch.float64)
    upper=torch.tensor([[19.]],dtype=torch.float64)
    lower=torch.tensor([[-9.5]],dtype=torch.float64)
    assert abs(float(normalized_worst(upper,bounds,.95,scale_about_zero=True)[0]))<1e-12
    assert abs(float(normalized_worst(lower,bounds,.95,scale_about_zero=True)[0]))<1e-12
    # The old midpoint-scaled convention gave +0.25 Nm extra upper headroom.
    assert float(normalized_worst(upper,bounds,.95)[0])>0
