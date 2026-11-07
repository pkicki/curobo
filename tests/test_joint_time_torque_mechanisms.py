# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for torque decomposition and native full-planner selection."""

from __future__ import annotations

import numpy as np
import pytest

from research.joint_time_oracle.torque_full_select import best_native, summary
from research.joint_time_oracle.torque_mechanisms import (
    common_physical_times,
    decompose_rnea,
    metrics,
)


def fake_rnea(q, v, a):
    return (2.0 + q) * a + 0.3 * v**2 + np.sin(q)


def profile():
    return {
        "profile_sha256": "p" * 64,
        "effort_lower": [-10.0, -10.0],
        "effort_upper": [10.0, 10.0],
    }


def valid_diag():
    return {
        "samples": 8193,
        "torque_profile_sha256": "p" * 64,
        "world_min_clearance_m": 0.01,
        "self_min_clearance_m": 0.02,
        "endpoints": {
            "position_start": 0.0,
            "position_goal": 0.0,
            "velocity": 0.0,
            "acceleration": 0.0,
        },
        "families": {
            name: {"physical": {"violation": 0.0}, "scaled95": {"violation": 0.0}}
            for name in ("position", "velocity", "acceleration", "jerk", "torque")
        },
    }


def test_rnea_decomposition_exact_toy_model():
    q = np.array([[0.1, 0.2], [0.3, -0.4], [0.6, 0.0]])
    v = np.array([[1.0, -2.0], [-1.0, 2.0], [3.0, 0.5]])
    a = np.array([[2.0, 3.0], [-1.0, 0.5], [0.4, -0.8]])
    terms = decompose_rnea(q, v, a, fake_rnea)
    np.testing.assert_allclose(terms["inertia"], (2.0 + q) * a, rtol=0, atol=1e-14)
    np.testing.assert_allclose(terms["velocity_bias"], 0.3 * v**2, rtol=0, atol=1e-14)
    np.testing.assert_allclose(terms["gravity"], np.sin(q), rtol=0, atol=1e-14)
    assert np.max(np.abs(terms["identity_residual"])) < 1e-13


def test_decomposition_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="matching"):
        decompose_rnea(np.zeros((3, 2)), np.zeros((2, 2)), np.zeros((3, 2)), fake_rnea)


def test_common_time_reconstruction_and_duplicate_timestamp_handling():
    t = np.array([0.0, 0.25, 1.0])
    q = np.column_stack((t, t * t))
    traj = {"time": t, "q": q, "qd": q * 2, "qdd": q * 0}
    dense = common_physical_times(traj, count=5)
    assert dense["q"].shape == (5, 2)
    assert dense["time"][-1] == pytest.approx(1.0)
    assert dense["normalized_time"][2] == pytest.approx(0.5)
    repeated = {
        **traj,
        "time": np.array([0.0, 0.25, 0.25, 1.0]),
        "q": np.insert(q, 2, q[1], axis=0),
        "qd": np.insert(q * 2, 2, (q * 2)[1], axis=0),
        "qdd": np.insert(q * 0, 2, (q * 0)[1], axis=0),
    }
    collapsed = common_physical_times(repeated, count=5)
    np.testing.assert_allclose(collapsed["q"], dense["q"], atol=1e-12)
    bad = {**repeated, "q": repeated["q"].copy()}
    bad["q"][2, 0] += 1e-3
    with pytest.raises(ValueError, match="inconsistent states"):
        common_physical_times(bad, count=5)


def test_metrics_uses_asymmetric_signed_side():
    q = np.array([[0.0, 0.0], [0.1, 0.2], [0.2, 0.3]])
    terms = {
        "total": np.array([[-3.0, 2.0], [-8.0, 3.0], [-1.0, 2.0]]),
        "gravity": np.zeros((3, 2)),
        "inertia": np.array([[-3.0, 2.0], [-8.0, 3.0], [-1.0, 2.0]]),
        "velocity_bias": np.zeros((3, 2)),
        "identity_residual": np.zeros((3, 2)),
    }
    bounds = np.array([[-10.0, -20.0], [20.0, 10.0]])
    m = metrics(terms, bounds, q, 0.5)
    assert m["max_torque_utilization"] == pytest.approx(0.8)
    assert m["max_torque_joint"] == 1
    assert m["max_torque_Nm"] == pytest.approx(-8.0)
    assert m["max_torque_normalized_time"] == pytest.approx(0.5)


def test_native_selector_filters_invalid_and_picks_best_verified():
    p = profile()
    c = {
        "F000": {
            "status": "SUCCESS",
            "duration_s": 0.41,
            "torque_profile_sha256": p["profile_sha256"],
        },
        "F001": {
            "status": "SUCCESS",
            "duration_s": 0.30,
            "torque_profile_sha256": p["profile_sha256"],
        },
        "F002": {
            "status": "SUCCESS",
            "duration_s": 0.40,
            "torque_profile_sha256": "incorrect",
        },
        "F003": {"status": "FAILED"},
    }
    a = valid_diag()
    b = valid_diag()
    b["families"]["torque"]["physical"]["violation"] = 0.1
    diags = {"task:F000": a, "task:F001": b, "task:F002": a}
    receipts = {
        "task:F000": {
            "physical": True,
            "scaled95": True,
            "endpoint": True,
            "collision": True,
            "torque_profile_sha256": p["profile_sha256"],
        }
    }
    assert (
        best_native(c, diags, {}, p, "task", 1, scaled95=False, strict=False)["method"]
        == "F000"
    )
    assert (
        best_native(c, diags, {}, p, "task", 4, scaled95=False, strict=False)["method"]
        == "F000"
    )
    assert best_native(c, diags, {}, p, "task", 4, scaled95=False, strict=True) is None
    assert (
        best_native(c, diags, receipts, p, "task", 4, scaled95=False, strict=True)[
            "method"
        ]
        == "F000"
    )


def test_signed_summary_tracks_negative_improvements():
    x = summary([-4.0, 2.0, 5.0], attempted=5)
    assert x["paired_n"] == 3 and x["attempted_n"] == 5
    assert x["median_pct"] == pytest.approx(2.0)
    assert x["slower_count"] == 1
    assert x["improved_gt_1pct"] == 2
    assert x["paired_bootstrap95_median_pct"] is not None


def test_internal_native_seed_lanes_count_within_same_restart_budget():
    p = profile()
    candidates = {
        "F000": {
            "status": "SUCCESS",
            "duration_s": 0.45,
            "restart": 0,
            "torque_profile_sha256": p["profile_sha256"],
        },
        "F000L02": {
            "status": "SUCCESS",
            "duration_s": 0.40,
            "restart": 0,
            "torque_profile_sha256": p["profile_sha256"],
        },
        "F001": {
            "status": "SUCCESS",
            "duration_s": 0.30,
            "restart": 1,
            "torque_profile_sha256": p["profile_sha256"],
        },
    }
    diagnostics = {"task:" + name: valid_diag() for name in candidates}
    first = best_native(
        candidates, diagnostics, {}, p, "task", 1, scaled95=False, strict=False
    )
    assert first["method"] == "F000L02"
    first_two = best_native(
        candidates, diagnostics, {}, p, "task", 2, scaled95=False, strict=False
    )
    assert first_two["method"] == "F001"


def test_matched_full_r_uses_same_restarter_budget_and_strict_receipts():
    p = profile()
    records = {
        "RF000": {
            "status": "COMPLETE",
            "duration_s": 0.42,
            "restart": 0,
            "torque_profile_sha256": p["profile_sha256"],
        },
        "RF000L02": {
            "status": "COMPLETE",
            "duration_s": 0.39,
            "restart": 0,
            "torque_profile_sha256": p["profile_sha256"],
        },
        "RF001": {
            "status": "COMPLETE",
            "duration_s": 0.31,
            "restart": 1,
            "torque_profile_sha256": p["profile_sha256"],
        },
    }
    diag = {"task:" + key: valid_diag() for key in records}
    receipts = {
        "task:RF000L02": {
            "physical": True,
            "scaled95": True,
            "endpoint": True,
            "collision": True,
            "torque_profile_sha256": p["profile_sha256"],
        }
    }
    v = best_native(
        records, diag, receipts, p, "task", 1, scaled95=False, strict=True, prefix="RF"
    )
    assert v["method"] == "RF000L02"
    v2 = best_native(
        records, diag, receipts, p, "task", 2, scaled95=False, strict=True, prefix="RF"
    )
    assert v2["method"] == "RF000L02"
    u = best_native(
        records, diag, receipts, p, "task", 2, scaled95=False, strict=False, prefix="RF"
    )
    assert u["method"] == "RF001"
