# SPDX-License-Identifier: Apache-2.0
"""CPU-only mathematical and fail-closed evidence tests for extreme torque cases."""

from __future__ import annotations

import numpy as np
import pytest

from research.joint_time_oracle.extreme_torque_core import (
    derived_profile,
    dilate_states,
    implied_torque,
    regime_name,
    screen_slowdowns,
)
from research.joint_time_oracle.extreme_torque_path_lp import pointwise_slack
from research.joint_time_oracle.extreme_torque_report import classify_case
from research.joint_time_oracle.torque_active import make_profile


def sample_old():
    """Create an asymmetric source severe profile for unit tests."""
    nominal = np.array([[-20.0, -30.0], [20.0, 30.0]])
    old = make_profile(
        case_id="severe::task01",
        source_case_id="task01",
        regime="severe",
        nominal_bounds=nominal,
        baseline_torques=np.array([[12.0, -6.0], [-8.0, 4.0]]),
        endpoint_gravity_torques=np.array([[3.0, -2.0], [4.0, -2.0]]),
        target=1.1,
        source_task_sha256="s" * 64,
        source_torque_sha256="t" * 64,
    )
    assert old.admitted
    return old.to_json()


def test_derived_asymmetric_profile_and_endpoint_gate():
    """Derived profiles scale signed limits and enforce endpoint gravity reserve."""
    old = sample_old()
    p = derived_profile(old, 0.8, source_sha="s" * 64)
    assert p.admitted
    assert p.derate == pytest.approx(old["derate"] * 0.8)
    assert p.effort_lower == pytest.approx(
        (np.asarray(old["nominal_effort_lower"]) * p.derate).tolist()
    )
    with pytest.raises(ValueError, match="source"):
        derived_profile(old, 0.8, source_sha="x" * 64)


def test_derating_names_are_predeclared_and_distinct():
    """Registered factor labels are stable and reject unregistered precision."""
    assert regime_name(1.0) == "extreme_f1000"
    assert regime_name(0.65) == "extreme_f0650"
    with pytest.raises(ValueError):
        regime_name(0.7005)


def test_time_dilation_obeys_rigid_body_scaling_and_endpoints():
    """Time dilation rescales derivatives while retaining geometry."""
    t = np.linspace(0, 1, 51)
    vel = np.sin(np.pi * t) ** 2
    acc = 2 * np.sin(np.pi * t) * np.cos(np.pi * t) * np.pi
    # qdd endpoints aren't zero in this illustrative curve? exact zero yes
    data = {
        "time": t,
        "q": np.column_stack((t**2 * (3 - 2 * t), np.zeros_like(t))),
        "qd": np.column_stack((vel, np.zeros_like(t))),
        "qdd": np.column_stack((acc, np.zeros_like(t))),
        "jerk": np.ones((len(t), 2)),
        "progress": t,
        "controls": np.full(32, 2.0),
    }
    altered = dilate_states(data, 2.0)
    np.testing.assert_array_equal(data["q"], altered["q"])
    np.testing.assert_allclose(altered["qd"], data["qd"] / 2)
    np.testing.assert_allclose(altered["qdd"], data["qdd"] / 4)
    np.testing.assert_allclose(altered["jerk"], data["jerk"] / 8)
    np.testing.assert_allclose(altered["controls"], data["controls"] / 4)
    assert altered["time"][-1] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="non-rest"):
        dilate_states({**data, "qd": np.ones((len(t), 2))}, 1.1)


def test_rigid_body_torque_time_scaling_identity():
    """Rigid-body inverse dynamics follows the screened time-scaling identity."""
    # Synthetic RNEA model: tau(q,v,a)=M(q)a + C(q)v^2 + gravity(q).
    q = np.linspace(-0.2, 0.4, 17)[:, None]
    v = np.linspace(0, 1, 17)[:, None]
    a = np.linspace(0.7, -0.3, 17)[:, None]
    gravity = 1.2 + q
    tau = gravity + (2 + q) * a + 0.25 * v * v
    rho = 1.8
    expected = gravity + (2 + q) * (a / rho**2) + 0.25 * (v / rho) ** 2
    np.testing.assert_allclose(implied_torque(tau, gravity, rho), expected, atol=1e-12)


def test_gravity_limited_path_cannot_be_rescued_by_slowing():
    """Slowing cannot fix torque infeasibility caused by gravity alone."""
    bounds = np.array([[-5.0], [5.0]])
    torque = np.array([[10.0], [10.0]])
    grav = np.array([[10.0], [10.0]])
    selected, records = screen_slowdowns(torque, grav, bounds, [1.0, 2.0, 4.0])
    assert selected is None
    assert len(records) == 3
    # Opposing inertial torque can be removed by time dilation.
    torque = np.array([[9.0], [9.0]])
    grav = np.array([[1.0], [1.0]])
    selected, records = screen_slowdowns(torque, grav, bounds, [1.0, 2.0, 4.0])
    assert selected == 2.0


def test_fixed_path_relaxed_lp_proves_unavoidable_gravity_limit():
    """The relaxed LP detects a gravity-only torque bound violation."""
    # When both path coefficients vanish, only g(q) remains.
    bounds = np.array([[-2.0, -2.0], [2.0, 2.0]])
    r = pointwise_slack(np.zeros(2), np.zeros(2), np.array([4.0, 0.0]), bounds)
    assert r["status"] == "SOLVED" and r["min_relaxed_expansion"] > 0.9
    assert r["strongly_infeasible"]
    # If x'=2y can compensate, a valid LP must not label infeasible.
    valid = pointwise_slack(
        np.array([1.0, 0.0]), np.zeros(2), np.array([4.0, 0.0]), bounds
    )
    assert valid["min_relaxed_expansion"] < 1e-8


def test_native_failure_requires_full_budget_control_and_witness():
    """Only complete failed runs with a qualified witness pass failure gates."""
    witness = {"physical": True, "scaled95": True}
    thirty_two = {f"F{k:03d}": {"status": "FAILED", "restart": k} for k in range(32)}
    result = classify_case(witness, True, thirty_two, {}, expected_restarts=32)
    assert result["decision"] == "NATIVE_BUDGET_FAILURE_WITH_FEASIBLE_ALTERNATIVE"
    assert result["native_completed_failed_restarts"] == 32
    # Missing restart != failure
    missing = classify_case(
        witness, True, dict(list(thirty_two.items())[:-1]), {}, expected_restarts=32
    )
    assert missing["decision"] == "INCOMPLETE_NATIVE_BUDGET"
    error = dict(thirty_two)
    error["F003"] = {"status": "ERROR", "restart": 3}
    blocked = classify_case(witness, True, error, {}, expected_restarts=32)
    assert blocked["decision"] == "BLOCKED_NATIVE_EXECUTION_ERRORS"
    unverified = dict(thirty_two)
    unverified["F003"] = {
        "status": "SUCCESS",
        "restart": 3,
        "torque_profile_sha256": "p",
    }
    uncertain = classify_case(witness, True, unverified, {}, expected_restarts=32)
    assert uncertain["decision"] == "NATIVE_RETURNED_UNQUALIFIED_CANDIDATES"
    pass_receipt = {"F003": {"physical": True, "torque_profile_sha256": "p"}}
    success = classify_case(
        witness, True, unverified, pass_receipt, expected_restarts=32
    )
    assert success["decision"] == "NATIVE_FOUND_FEASIBLE_TRAJECTORY"
    no_witness = classify_case(
        {"physical": True, "scaled95": False},
        True,
        thirty_two,
        {},
        expected_restarts=32,
    )
    assert no_witness["decision"] == "NO_INDEPENDENT_FEASIBLE_WITNESS"


def test_native_failure_with_no_control_is_not_strong_claim():
    """A failed challenge without its negative control is not conclusive."""
    rows = {f"F{i:03d}": {"status": "FAILED", "restart": i} for i in range(4)}
    x = classify_case(
        {"physical": True, "scaled95": True}, False, rows, {}, expected_restarts=4
    )
    assert x["decision"] == "NATIVE_BUDGET_FAILURE_BUT_CONTROL_NOT_VERIFIED"
