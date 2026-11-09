# SPDX-License-Identifier: Apache-2.0
"""Offline numerical/data-contract tests for retiming-guided payload study.

No native GPU, physics campaign or collision parity is simulated by
these tests. They must be followed by real pinned CUDA/Pinocchio tests.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from research.joint_time_oracle.timeopt_payload_model import (
    MASSES_KG,Payload,classify_endpoint_gravity,from_record,
)
from research.joint_time_oracle.timeopt_feedback_search import (
    ARMS,INIT_BATCH,Candidate,nominal_path,propose,rng_seed,
    search,
)
from research.joint_time_oracle.timeopt_oracle_contract import (
    CONSTRAINT_FAMILIES,K107_ROUTE,check_response,path_record,request,
)
from research.joint_time_oracle.timeopt_k107_independent_audit import (
    reconstruct,elapsed_time,max_bounds,event_grid,
)
from research.joint_time_oracle.cohort_geometry import QuinticGeometry


def task():
    payload=Payload(1.0).as_record()
    effort=np.array([87,87,87,87,12,12,12.])
    return {
        "case_id":"test_nominal_payload","task_sha256":"task_sha_test",
        "payload":payload,"actuator_scale_by_joint":[1.]*7,
        "original_effort_lower":(-effort).tolist(),
        "original_effort_upper":effort.tolist(),
    }


def limits():
    return {
        "velocity":np.stack((-np.ones(7)*2,np.ones(7)*2)).tolist(),
        "acceleration":np.stack((-np.ones(7)*5,np.ones(7)*5)).tolist(),
        "jerk":np.stack((-np.ones(7)*50,np.ones(7)*50)).tolist(),
        "torque":np.stack((-np.array([87,87,87,87,12,12,12.]),
                           np.array([87,87,87,87,12,12,12.]))).tolist()
    }


def test_payload_mass_and_inertia_digest_are_immutable():
    assert MASSES_KG[0]==0 and MASSES_KG[-1]==5
    row=Payload(3.).as_record()
    assert row["central_inertia_kg_m2"]==pytest.approx(.4*3*.06**2)
    assert from_record(row)==Payload(3.)
    for key,value in (("mass_kg",2.),("radius_m",.5),
                      ("central_inertia_kg_m2",.1),
                      ("payload_sha256","changed")):
        edited={**row,key:value}
        with pytest.raises((ValueError,TypeError,KeyError)):
            from_record(edited)


def test_negative_payload_mass_not_admissible():
    with pytest.raises(ValueError):
        Payload(-.1)
    with pytest.raises(ValueError):
        Payload(float("nan"))


def test_endpoint_static_gravity_is_not_labeled_planner_failure():
    limits=np.stack((-np.ones(7)*8,np.ones(7)*8))
    gravity=np.zeros((2,7))
    gravity[1,2]=9.
    result=classify_endpoint_gravity(gravity,limits)
    assert result["category"]=="STATIC_ENDPOINT_IMPOSSIBLE"
    assert not result["endpoint_static_feasible_nominal"]
    assert result["endpoint_utilization"]>1.


def test_quintic_nominal_geometry_is_end_flat_and_exact():
    q0=np.array([0,-.8,0,-2.3,0,1.5,.7])
    q1=q0+np.array([.3,-.2,.3,.1,-.3,-.2,.4])
    ctrl=nominal_path(q0,q1)
    assert ctrl.shape==(22,7)
    geom=QuinticGeometry(ctrl)
    ends=geom.evaluate(np.array([0.,1.]))
    np.testing.assert_allclose(ends[0],[q0,q1],atol=1e-9)
    np.testing.assert_allclose(ends[1],0,atol=1e-8)
    np.testing.assert_allclose(ends[2],0,atol=1e-7)
    path=path_record(ctrl)
    assert path["degree"]==5 and len(path["knots"])==28
    assert path["path_progress_range"]==[0.,1.]


def test_initial_proposals_identical_across_all_arms():
    base=nominal_path(np.zeros(7),np.ones(7)*.25)
    outputs=[]
    for arm in ARMS:
        rng=np.random.default_rng(rng_seed("case",20261109,arm))
        proposal=[propose(base,[],arm=arm,step=i,rng=rng)[0]
                  for i in range(INIT_BATCH)]
        outputs.append(np.asarray(proposal))
    for value in outputs[1:]:
        np.testing.assert_array_equal(value,outputs[0])


def test_feedback_policy_uses_measured_constraint_bottleneck():
    base=nominal_path(np.zeros(7),np.ones(7)*.25)
    elite=Candidate(
        controls=base,status="FEASIBLE_SCREENED",
        response={"status":"FEASIBLE_SCREENED","duration_s":1.9,
                  "diagnostics":{"diagnostic_valid":True,
                                 "bottleneck_s":.3,
                                 "bottleneck_joint":4,
                                 "bottleneck_family":"torque"}},
        proposal=1,novelty=.5,cost=1.9)
    rng=np.random.default_rng(2)
    new,info=propose(base,[elite],arm="timeopt_feedback",
                     step=INIT_BATCH,rng=rng)
    assert info["feedback"]["joint"]==4
    assert info["direction_type"]=="measured_timeopt_bottleneck"
    np.testing.assert_array_equal(new[:3],base[:3])
    np.testing.assert_array_equal(new[-3:],base[-3:])


def test_k107_worker_contract_is_strict_and_cannot_self_certify():
    q0=np.zeros(7);q1=np.ones(7)*.3
    req=request(task(),nominal_path(q0,q1),request_id="test",limit_contract=limits())
    expected={
        "request_sha256":req["request_sha256"],
        "path_sha256":req["path"]["path_sha256"],
        "task_sha256":req["task_sha256"],
        "payload_sha256":req["payload"]["payload_sha256"],
        "retimer_actual":K107_ROUTE,
        "endpoint_contract":req["endpoint_contract"],
        "active_families":list(CONSTRAINT_FAMILIES),
        "torque_rate_constraint":req["torque_rate_constraint"],
        "status":"FEASIBLE_SCREENED",
        "duration_s":2.,"worker_wall_ms":3.,"retiming_ms":2.,
        "model_setup_ms":.7,"audit_ms":.3,
        "independent_scaled95_qualified":False,
    }
    assert check_response(req,expected)["status"]=="FEASIBLE_SCREENED"
    with pytest.raises(ValueError,match="retimer"):
        check_response(req,{**expected,"retimer_actual":"IPOPT"})
    with pytest.raises(ValueError,match="self-certify"):
        check_response(req,{**expected,"independent_scaled95_qualified":True})


def test_k107_contract_does_not_allow_synthetic_derated_nominal_tasks():
    t=task()
    t["actuator_scale_by_joint"]=[.3]*4+[1.]*3
    with pytest.raises(ValueError,match="synthetic"):
        request(t,nominal_path(np.zeros(7),np.ones(7)*.1),
                request_id="t",limit_contract=limits())


def test_independent_spline_timing_matches_unit_speed():
    geometry=QuinticGeometry(nominal_path(np.zeros(7),np.ones(7)*.3))
    coeff=np.ones(107)
    s=event_grid(geometry,len(coeff),257)
    states=reconstruct(geometry,coeff,s)
    assert all(np.isfinite(v).all() for v in states.values())
    assert elapsed_time(coeff,s)==pytest.approx(1.,rel=1e-8)
    assert np.max(np.abs(states["qd"][0]))<1e-8
    assert np.max(np.abs(states["qd"][-1]))<1e-8
    bounds=np.stack((-np.ones(7)*3,np.ones(7)*3))
    assert max_bounds(states["qd"],bounds,scaled=True)["max_violation"]==0


def test_mock_oracle_feedback_search_does_not_claim_qualification():
    start=np.zeros(7);goal=np.ones(7)*.1
    base=nominal_path(start,goal)
    def fake_oracle(_controls,index):
        if index>15:
            return {"status":"FEASIBLE_SCREENED","duration_s":3.,
                    "worker_wall_ms":1.,
                    "diagnostics":{"diagnostic_valid":True,
                                   "bottleneck_s":.45,
                                   "bottleneck_joint":3,
                                   "bottleneck_family":"torque"}}
        return {"status":"INFEASIBLE_SCREENED","worker_wall_ms":1.,
                "diagnostics":{"max_normalized_violation":1.}}
    result=search(base,fake_oracle,arm="timeopt_feedback",
                  task_id="test",budget=32)
    assert result["first_preliminary_feasible_proposal"]==16
    assert result["independent_qualified_success"] is False
    assert result["oracle_requests"]==32
