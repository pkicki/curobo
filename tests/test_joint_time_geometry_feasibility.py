# SPDX-License-Identifier: Apache-2.0
"""CPU numerical regressions for geometry-dependent torque infeasibility."""

from __future__ import annotations

import json

import numpy as np
import pytest

from research.joint_time_oracle.cohort_geometry import QuinticGeometry
from research.joint_time_oracle.geometry_feasibility_core import (
    CANDIDATE_CONTROLS,
    gate_l1,
    path_distance,
    perturb_controls,
    profile_from_scenario,
    scenarios,
    smooth_time,
    smoothstep_controls,
    timed_states,
)
from research.joint_time_oracle.geometry_feasibility_refine import (
    dual_slack_certificate,
    refinement_candidates,
)
from research.joint_time_oracle.geometry_feasibility_report import classify
from research.joint_time_oracle.torque_active import read_profile


def sample_task():
    return {
        "case_id": "fresh_example",
        "task_sha256": "s" * 64,
        "start": [0.0, -0.5, 0.0, -2.2, 0.0, 1.2, 0.0],
        "goal": [0.3, -0.1, 0.2, -1.7, 0.1, 1.5, 0.3],
    }


def sample_profile():
    task = sample_task()
    lim = np.array(
        [
            [-10.0, -20.0, -30.0, -40.0, -50.0, -60.0, -70.0],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        ]
    )
    return profile_from_scenario(
        source_task=task,
        original_limits=lim,
        endpoint_gravity=np.zeros((2, 7)),
        scenario=scenarios()[0],
        model_sha256="m" * 64,
        source_geometry_sha256="g" * 64,
    )


def test_joint_selective_profile_is_stable_asymmetric_and_loadable(tmp_path):
    p = sample_profile()
    assert p["admitted"]
    assert len(p["actuator_scale_by_joint"]) == 7
    scale = np.asarray(p["actuator_scale_by_joint"])
    assert np.sum(scale < 1.0) == 3
    assert set(np.where(scale < 1.0)[0] + 1) == set(
        scenarios()[0]["selected_joints_1based"]
    )
    f = tmp_path / "profile.json"
    f.write_text(json.dumps(p))
    checked = read_profile(f)
    assert checked["profile_sha256"] == p["profile_sha256"]
    bounds = np.stack((p["effort_lower"], p["effort_upper"]))
    np.testing.assert_allclose(
        bounds, np.array([p["nominal_effort_lower"], p["nominal_effort_upper"]]) * scale
    )
    p["effort_upper"][1] *= 1.1
    f.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="checksum"):
        read_profile(f)


def test_joint_selective_endpoint_admission_fails_closed():
    lim = np.vstack((-np.ones(7) * 10, np.ones(7) * 10))
    gravity = np.zeros((2, 7))
    scenario = scenarios()[0]
    gravity[0, scenario["selected_joints_1based"][0] - 1] = 4.0
    p = profile_from_scenario(
        source_task=sample_task(),
        original_limits=lim,
        endpoint_gravity=gravity,
        scenario=scenario,
        model_sha256="m" * 64,
        source_geometry_sha256="g" * 64,
    )
    assert p["admitted"] is False
    assert p["rejection_reason"] is not None


def test_smoothstep_exact_rest_to_rest_timing_chain_rule():
    q0 = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    q1 = np.array([0.8, 0.1, 0.2, -1.0, 0.2, -0.3, 0.0])
    controls = smoothstep_controls(q0, q1, count=CANDIDATE_CONTROLS)
    geom = QuinticGeometry(controls)
    state = timed_states(geom, 2.0, n=4001)
    np.testing.assert_allclose(state["q"][0], q0, atol=1e-12)
    np.testing.assert_allclose(state["q"][-1], q1, atol=1e-12)
    assert np.max(np.abs(state["qd"][[0, -1]])) < 1e-12
    assert np.max(np.abs(state["qdd"][[0, -1]])) < 1e-12
    assert np.isfinite(state["jerk"]).all()
    # Interior velocity and acceleration agree with finite-difference derivatives.
    dt = float(np.diff(state["time"])[0])
    k = 2000
    np.testing.assert_allclose(
        (state["q"][k + 1] - state["q"][k - 1]) / (2 * dt), state["qd"][k], atol=2e-5
    )
    np.testing.assert_allclose(
        (state["qd"][k + 1] - state["qd"][k - 1]) / (2 * dt), state["qdd"][k], atol=2e-4
    )


def test_smooth_phase_derivatives_and_deterministic_path_perturbation():
    z, zu, zuu, zuuu = smooth_time(np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(z, [0.0, 0.5, 1.0], atol=1e-12)
    np.testing.assert_allclose(zu[[0, -1]], 0, atol=1e-12)
    np.testing.assert_allclose(zuu[[0, -1]], 0, atol=1e-12)
    controls = smoothstep_controls(np.zeros(7), np.ones(7))
    rng = np.random.default_rng(10)
    a = perturb_controls(controls, rng, amplitude=0.8, target_joint=3)
    b = perturb_controls(
        controls, np.random.default_rng(10), amplitude=0.8, target_joint=3
    )
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[:3], controls[:3])
    np.testing.assert_array_equal(a[-3:], controls[-3:])
    assert not np.array_equal(a[3:-3], controls[3:-3])


def test_alternative_path_has_nontrivial_arc_distance():
    zero = smoothstep_controls(np.zeros(7), np.ones(7))
    x = QuinticGeometry(zero)
    y = QuinticGeometry(
        perturb_controls(zero, np.random.default_rng(2), amplitude=1.2, target_joint=2)
    )
    s = np.linspace(0, 1, 501)
    assert path_distance(x.evaluate(s)[0], y.evaluate(s)[0]) > 0.10


def test_lp_dual_lower_bound_confirms_fixed_curve_infeasibility():
    g = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    z = np.zeros(7)
    lim = np.vstack((-np.ones(7) * 2, np.ones(7) * 2))
    cert = dual_slack_certificate(z, z, g, lim)
    assert cert["status"] == "SOLVED"
    assert cert["numerically_strong_infeasibility"]
    assert cert["dual_lower_bound"] > 0.01
    # Changing the geometry coefficients can recover the pointwise torque LP.
    a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    feasible = dual_slack_certificate(a, z, g, lim)
    assert feasible["minimum_required_expansion"] < 1e-8


def test_strong_geometry_requires_LP_and_witness_not_native_failure():
    good = {"physical": True, "scaled95": True, "collision": True, "endpoint": True}
    lp = {
        "status": "NUMERICALLY_STRONG_ORIGINAL_PATH_INFEASIBILITY",
        "numerical_dual_certificate": {"numerically_strong_infeasibility": True},
    }
    nothing = classify(good, lp, {}, {}, budget=4)
    assert nothing["category"] == "GEOMETRY_OBSTRUCTION_NATIVE_BUDGET_INCOMPLETE"
    failures = {f"F{i:03d}": {"status": "FAILED", "restart": i} for i in range(4)}
    blocked = classify(good, lp, failures, {}, budget=4)
    assert blocked["category"] == "GEOMETRY_OBSTRUCTION_NATIVE_BUDGET_FAILURE"
    successful = dict(failures)
    successful["F001"] = {
        "status": "SUCCESS",
        "restart": 1,
        "torque_profile_sha256": "q",
    }
    verified = {"F001": {"physical": True, "torque_profile_sha256": "q"}}
    native = classify(good, lp, successful, verified, budget=4)
    assert (
        native["category"] == "GEOMETRY_OBSTRUCTION_BUT_NATIVE_REPLANNED_SUCCESSFULLY"
    )
    neither = classify(good, None, failures, {}, budget=4)
    assert neither["category"] == "NO_STRONG_ORIGINAL_GEOMETRY_INFEASIBILITY_PROOF"


def test_lp_discovery_gate_does_not_count_presolve_only_as_confirmation():
    p = sample_profile()
    o = {"max_required_torque_bound_expansion": 0.025}
    a = {"max_required_torque_bound_expansion": 0.0}
    r = gate_l1(p, o, a)
    assert r["discovery_pass"]
    assert "screen" in r["scope"]


def test_refinement_includes_positive_lp_without_constructive_witness(tmp_path):
    source = tmp_path / "source"
    arm = tmp_path / "arm"
    source.mkdir()
    arm.mkdir()
    tasks = [sample_task(), {**sample_task(), "case_id": "other"}]
    (source / "tasks.json").write_text(json.dumps(tasks))
    (source / "pilot.jsonl").write_text(
        "".join(json.dumps({"case_id": task["case_id"]}) + "\n" for task in tasks)
    )
    positive = arm / tasks[0]["case_id"]
    negative = arm / tasks[1]["case_id"]
    positive.mkdir()
    negative.mkdir()
    (positive / "discovery.json").write_text(
        json.dumps({"status": "POSITIVE_ORIGINAL_LP_NEEDS_ALTERNATIVE"})
    )
    (negative / "discovery.json").write_text(
        json.dumps({"status": "ORIGINAL_LP_NOT_INFEASIBLE"})
    )
    # The scenario's task list is witness-only and may be empty at this stage.
    (arm / "tasks.json").write_text("[]")

    assert [task["case_id"] for task in refinement_candidates(source, arm)] == [
        tasks[0]["case_id"]
    ]


def test_synthetic_scenario_grid_is_unique_and_reproducible():
    a = scenarios()
    assert len(a) > 20 and a == scenarios()
    assert len({x["scenario_id"] for x in a}) == len(a)
