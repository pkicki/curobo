# SPDX-License-Identifier: Apache-2.0
"""No-GPU numerical/provenance tests for payload and retiming feedback."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.joint_time_oracle.payload_contract import (
    Payload, compare_native_inverse_dynamics, frozen_profiles,
)
from research.joint_time_oracle.payload_native import (
    combined_link_inertia, inertia_matrix, inertia_six, parallel_axis,
)
from research.joint_time_oracle.native_uniform_seed_replay import uniform_seed_bounds
from research.joint_time_oracle.retimer_oracle import (
    Oracle, digest, make_request, validate_response,
)
from research.joint_time_oracle.timeopt_feedback_search import (
    ARMS, INITIAL, proposal, run, seed_for,
)
from research.joint_time_oracle.payload_hardcase_registry import (
    classify, population_split,
)


def test_payload_mass_and_inertia_preserves_nominal_actuator_limits():
    registered=frozen_profiles()
    assert len(registered)==12
    assert all(x["effort_scale"]==1 for x in registered)
    assert all(x["payload"]["mass_kg"]>=0 for x in registered)
    with pytest.raises(ValueError):
        Payload(-1).validate()


def test_combined_inertia_parallel_axis_and_symmetry():
    old=np.array([0.,0.,0.,2.])
    six=np.array([.1,.1,.1,0.,0.,0.])
    p=Payload(2.,(.1,0.,0.),inertia_diagonal_kg_m2=(.01,.01,.01))
    updated,after=combined_link_inertia(old,six,p)
    assert updated[3]==pytest.approx(4.)
    np.testing.assert_allclose(updated[:3],[.05,0,0])
    mat=inertia_matrix(after)
    np.testing.assert_allclose(mat,mat.T)
    assert min(np.linalg.eigvalsh(mat))>0
    np.testing.assert_allclose(inertia_six(mat),after)
    np.testing.assert_allclose(parallel_axis(2.,np.zeros(3)),0)


def test_rnea_model_parity_gate_blocks_changed_payload():
    exact=np.ones((16,7))*5.
    good=compare_native_inverse_dynamics(exact,exact)
    assert good["aligned"]
    bad=compare_native_inverse_dynamics(exact+2,exact)
    assert not bad["aligned"]
    with pytest.raises(ValueError):
        compare_native_inverse_dynamics(exact[:2],exact)


def _bounds(sampled=3):
    q=np.zeros((sampled,7))
    return q,{"velocity":np.ones(7)*3,
              "acceleration":np.ones(7)*4,
              "jerk":np.ones(7)*10,
              "torque":np.stack((-np.ones(7)*8,np.ones(7)*8))}


def test_native_uniform_seed_infeasible_even_if_nonuniform_possible():
    q,limits=_bounds()
    qs=np.zeros_like(q);qss=np.zeros_like(q);qsss=np.zeros_like(q)
    gravity=np.zeros_like(q)
    dynamic=np.zeros_like(q)
    # At one point torque demands x >= 2; elsewhere velocity x <= 1.
    dynamic[0,0]=2.
    gravity[0,0]=-12.
    qs[1,0]=3.
    # torque at point 0: -8 <= -12+2*x <=8 => x>=2
    limits["velocity"][0]=3.
    outcome=uniform_seed_bounds(q,qs,qss,qsss,gravity,dynamic,limits)
    assert outcome["status"]=="UNIFORM_SEED_EMPTY"
    assert outcome["lower_witness"]["joint"]==1
    assert outcome["upper_witness"]["family"]=="velocity"
    assert outcome["interval_gap"]<0
    # Empty uniform x is NOT a proof of infeasible variable x(s).
    assert "nonuniform" in outcome["assumption"]


def test_uniform_seed_with_original_signed_torque_supports_asymmetry():
    q,limits=_bounds()
    qs=np.zeros_like(q);qss=np.zeros_like(q);qsss=np.zeros_like(q)
    g=np.zeros_like(q);dyn=np.zeros_like(q)
    limits["torque"][0,1]=-4.
    limits["torque"][1,1]=9.
    dyn[0,1]=2.;g[0,1]=-6.
    found=uniform_seed_bounds(q,qs,qss,qsss,g,dyn,limits)
    assert found["uniform_x_lower"]==pytest.approx(1.)
    assert found["status"]=="UNIFORM_SEED_POSSIBLE"


def test_feedback_arms_have_identical_first_16_proposals():
    base=np.zeros((22,7))
    for k in range(INITIAL):
        traces=[proposal(base,[],task="trial",iteration=k,arm=arm)[0]
                for arm in ARMS]
        for x in traces[1:]:
            np.testing.assert_array_equal(x,traces[0])
    assert seed_for("trial",3)!=seed_for("trial",4)


def test_feedback_uses_oracle_bottleneck_only_when_available():
    base=np.zeros((22,7))
    def fake(_controls,_iteration):
        return {"status":"INFEASIBLE_HEURISTIC",
                "infeasibility_score":2.5,"bottleneck_s":.18}
    a=run("task",base,fake,arm="feedback",budget=INITIAL+7)
    b=run("task",base,fake,arm="rank_only",budget=INITIAL+7)
    assert len(a["ledger"])==INITIAL+7
    assert [x["path_sha256"] for x in a["ledger"][:INITIAL]] == [
        x["path_sha256"] for x in b["ledger"][:INITIAL]]
    assert any(x["mode"]=="RETIME_BOTTLENECK_GUIDED" for x in a["ledger"])
    assert not any(x["mode"]=="RETIME_BOTTLENECK_GUIDED" for x in b["ledger"])


def contract():
    return {"nominal_effort_scale":1.,
            "native_pinocchio_payload_parity_verified":True,
            "attached_collision_parity_verified":True,
            "payload":Payload(0.).serialize()}


def test_retimer_request_rejects_mismatched_model_and_response():
    oracle=Oracle("toppra_newton_k107",("python","my-adapter.py"),"abc123")
    limits={"families":["position","velocity","acceleration","jerk","torque"],
            "signed_torque_bounds_sha256":"nominal_a",
            "endpoint_contract":"strict-rest"}
    base=np.zeros((22,7))
    request=make_request("case",base,source_sha="source",
                         model_contract=contract(),limit_contract=limits,engine=oracle)
    assert request["schema"]=="payload-timeopt-bridge-v1"
    response={"schema":request["schema"],"status":"INFEASIBLE_HEURISTIC",
              "request_sha256":request["request_sha256"],
              "engine":oracle.name,"engine_commit":oracle.engine_commit,
              "path_controls_sha256":request["path_controls_sha256"],
              "model_contract_sha256":digest(contract()),
              "constraint_contract_sha256":digest(limits)}
    assert validate_response(request,response)["status"]=="INFEASIBLE_HEURISTIC"
    response["engine_commit"]="unrelated"
    with pytest.raises(ValueError,match="revision"):
        validate_response(request,response)


def test_true_hard_case_requires_qualified_alternative_and_same_contract():
    common={x:"same" for x in (
        "task_sha256","payload_sha256","native_model_sha256",
        "collision_model_sha256","effort_contract_sha256",
        "constraint_contract_sha256","endpoint_contract_sha256")}
    audited={"physical":True,"scaled95":True,"collision":True,"endpoint":True,
             "dense_samples":8193,"adaptive_samples":8193,
             "source_trajectory_sha256":"abc","model_alignment_pass":True}
    native={**common,"status":"NO_VERIFIED_SOLUTION",
            "profile":{"effort_scale":1.},
            "payload_dynamics_parity_pass":True,
            "attached_collision_parity_pass":True,
            "endpoint_static_torque_admissible":True}
    alt={**common,"duration_s":4.,"independent_audit":audited}
    assert classify(native,alt)["status"]=="HARD_NATIVE_FAILURE_WITH_FEASIBLE_ALTERNATIVE"
    assert classify(native,None)["status"]=="NATIVE_FAILURE_NO_INDEPENDENT_EXISTENCE_WITNESS"
    bad={**alt,"payload_sha256":"different"}
    assert classify(native,bad)["status"]=="INCOMPARABLE_CONSTRAINT_OR_PAYLOAD_CONTRACT"
    assert population_split("case1")==population_split("case1")
