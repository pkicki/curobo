# SPDX-License-Identifier: Apache-2.0
"""CPU-only integrity tests for full-native adaptive receipts."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from research.joint_time_oracle.torque_active import make_profile, write_profile
from research.joint_time_oracle.torque_full_receipts import validate_adaptive_candidate


def test_adaptive_receipt_sha_and_replay_required(tmp_path):
    arm=tmp_path/"severe"
    case="pilot_free_00"
    source=arm/"native_full"/case
    profile_dir=arm/case
    source.mkdir(parents=True)
    profile_dir.mkdir(parents=True)
    lim=np.array([[-10.,-12.],[10.,12.]])
    prof=make_profile(case_id="severe::"+case,source_case_id=case,
                      regime="severe",nominal_bounds=lim,
                      baseline_torques=np.array([[5.,-4.],[1.,3.]]),
                      endpoint_gravity_torques=np.array([[1.,-1.],[2.,-2.]]),
                      target=1.10,source_task_sha256="a"*64,
                      source_torque_sha256="b"*64)
    assert prof.admitted
    write_profile(profile_dir/"torque_profile.json",prof)
    time=np.linspace(0.,1.,8193)
    np.savez_compressed(source/"f000_dense.npz",time=time)
    np.savez_compressed(source/"f000_adaptive_dense.npz",
                        time=np.sort(np.unique(np.r_[time,.123456789])))
    digest=hashlib.sha256((source/"f000_dense.npz").read_bytes()).hexdigest()
    (source/"native_candidates.json").write_text(json.dumps({
        "F000":{"status":"SUCCESS","dense_sha256":digest,
                "torque_profile_sha256":prof.to_json()["profile_sha256"]}}))
    (source/"native_adaptive_replay.json").write_text(json.dumps({
        "F000":{"status":"REPLAYED","max_shared_state_error":0.}}))
    family={k:{"physical":{"violation":0.},
               "scaled95":{"violation":0.}} for k in
            ("position","velocity","acceleration","jerk","torque")}
    diagnostic={case+":F000_adaptive":{
        "torque_profile_sha256":prof.to_json()["profile_sha256"],
        "samples":8194,"world_min_clearance_m":None,
        "self_min_clearance_m":.05,
        "endpoints":{"position_start":0.,"position_goal":0.,
                     "velocity":0.,"acceleration":0.},
        "families":family}}
    good=validate_adaptive_candidate(arm,case,"F000",diagnostic)
    assert good["physical"] is True and good["scaled95"] is True
    assert good["added_samples"] == 1
    (source/"native_adaptive_replay.json").write_text(json.dumps({
        "F000":{"status":"ERROR"}}))
    bad=validate_adaptive_candidate(arm,case,"F000",diagnostic)
    assert bad["physical"] is False and bad["collision"] is False
    (source/"native_adaptive_replay.json").write_text(json.dumps({
        "F000":{"status":"REPLAYED"}}))
    diagnostic[case+":F000_adaptive"]["torque_profile_sha256"]="bad"
    bad=validate_adaptive_candidate(arm,case,"F000",diagnostic)
    assert bad["physical"] is False


def test_missing_candidate_receipt_fails_closed(tmp_path):
    case="x"
    (tmp_path/case).mkdir()
    # A profile is required; an unknown task may not silently qualify.
    try:
        validate_adaptive_candidate(tmp_path,case,"F000",{})
    except (FileNotFoundError, ValueError):
        pass
    else:
        raise AssertionError("missing profile must not qualify")
