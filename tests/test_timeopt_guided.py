# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the TimeOpt-guided planning research harness (research/timeopt_guided).

Pure tests run anywhere. CUDA tests need cuRobo + ``time_optimal_curobo/src`` on
PYTHONPATH; Pinocchio tests need the Pinocchio environment. Each group skips
cleanly when its environment is absent.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from research.timeopt_guided.common import Payload, payload_grid, rest_controls, task_split
from research.timeopt_guided.geometry import (
    directed_proposal, evaluate, fit_quintic, global_proposal, local_proposal,
    native_cubic_to_clamped, native_uniform_spline,
)
from research.timeopt_guided.report import classify

HAS_CUDA_STACK = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("curobo_timeopt") is not None
HAS_PIN = importlib.util.find_spec("pinocchio") is not None


def _native_lane(rng, n=30, moving_start=False):
    q0 = rng.normal(size=7).astype(np.float32)
    q1 = rng.normal(size=7).astype(np.float32)
    sb = np.tile(q0, (4, 1))
    if moving_start:
        sb[3] += 0.01
    c = (np.cumsum(rng.normal(size=(n, 7)) * 0.1, 0) + q0).astype(np.float32)
    c[-3:] = q1
    return sb, c, q1


@pytest.mark.parametrize("moving_start", [False, True])
def test_native_cubic_conversion_is_exact(moving_start):
    rng = np.random.default_rng(0)
    sb, c, g = _native_lane(rng, moving_start=moving_start)
    out = native_cubic_to_clamped(sb, c, g)
    spl = native_uniform_spline(sb, c, g)
    for nu in range(4):
        # q''' of a cubic is piecewise constant: compare away from knots.
        u = np.linspace(0, 1, 301) if nu < 3 else (np.arange(300) + 0.37) / 300
        s = out["s0"] + (out["s1"] - out["s0"]) * u
        ref = spl(s, nu=nu) * (out["s1"] - out["s0"]) ** nu
        got = evaluate(out["controls"], u, degree=3, knots=out["knots"], order=nu)
        assert np.max(np.abs(ref - got)) < 1e-6 * max(1.0, np.max(np.abs(ref)))
    assert np.all(out["knots"][:4] == 0) and np.all(out["knots"][-4:] == 1)


def test_proposals_preserve_rest_endpoints_and_limits():
    rng = np.random.default_rng(1)
    q0, q1 = np.zeros(7, np.float32), np.ones(7, np.float32)
    base = rest_controls(q0, q1)
    lo, hi = -np.full(7, 0.5), np.full(7, 1.2)
    for c in (
        global_proposal(base, rng, amplitude=1.0, lower=lo, upper=hi),
        local_proposal(base, rng, amplitude=0.1, lower=lo, upper=hi),
        directed_proposal(base, rng, s_center=0.5, direction=np.ones(7), amplitude=0.5, lower=lo, upper=hi),
    ):
        assert np.allclose(c[:3], q0) and np.allclose(c[-3:], q1)
        assert np.all(c[3:-3] >= lo - 1e-12) and np.all(c[3:-3] <= hi + 1e-12)
        d1 = evaluate(c, np.array([0.0, 1.0]), order=1)
        assert np.allclose(d1, 0.0, atol=1e-9)  # rest endpoints: zero path velocity


def test_fit_quintic_recovers_quintic():
    rng = np.random.default_rng(2)
    c = global_proposal(rest_controls(np.zeros(7), np.ones(7)), rng, amplitude=0.5,
                        lower=-np.full(7, 3.0), upper=np.full(7, 3.0))
    s = np.linspace(0, 1, 400)
    fit = fit_quintic(s, evaluate(c, s), smooth=1e-12)
    assert np.max(np.abs(evaluate(fit, s) - evaluate(c, s))) < 1e-5


def test_payload_contract_and_split():
    grid = payload_grid()
    assert len({p.label for p in grid}) == len(grid)
    for p in grid:
        assert Payload.from_json(p.to_json()) == p
    with pytest.raises(ValueError):
        Payload(20.0, (0, 0, 0.1)).validate()
    ids = [f"t{i}" for i in range(500)]
    splits = [task_split(i) for i in ids]
    assert splits == [task_split(i) for i in ids]
    assert 0.1 < splits.count("confirmation") / len(ids) < 0.3


def test_classification_rules():
    nv = {"scaled95_pass": True, "duration_s": 2.0}
    run = lambda ok, d: {"best_valid": ok, "best_duration_s": d}  # noqa: E731
    fail = {"native_success": False, "native_validation": None, "arms": {"D": [run(True, 2.0)]}}
    assert classify(fail) == "CONFIRMED_HARD_FAILURE"
    fail["arms"] = {"D": [run(False, None)]}
    assert classify(fail) == "UNRESOLVED_FAILURE"
    succ = {"native_success": True, "native_validation": nv, "arms": {"B": [run(True, 1.85)]}}
    assert classify(succ) == "NATIVE_SUBOPTIMAL"
    succ["arms"] = {"B": [run(True, 1.95)]}
    assert classify(succ) == "NATIVE_SUCCESS"


@pytest.mark.skipif(not HAS_CUDA_STACK, reason="needs CUDA cuRobo env with time_optimal_curobo on PYTHONPATH")
def test_k107_oracle_payload_and_feasibility():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from research.timeopt_guided.k107_oracle import K107Oracle

    q0 = np.array([0.0, -0.8, 0.0, -2.4, 0.0, 1.6, 0.8], np.float32)
    q1 = np.array([1.2, 0.3, 0.4, -1.2, 0.3, 2.2, 0.0], np.float32)
    light, heavy = K107Oracle(Payload(0.0, (0, 0, 0.12))), K107Oracle(Payload(5.0, (0.08, 0, 0.16)))
    path = rest_controls(q0, q1)
    g0, g1 = light.gravity(path), heavy.gravity(path)
    assert np.max(np.abs(g1[:, 5])) > np.max(np.abs(g0[:, 5])) + 1.0  # COM offset loads joint 6
    r = light.retime(path)
    assert r.status == "FEASIBLE" and 0.3 < r.duration_s < 3.0
    tr = light.sample_trajectory(path, r, n=2049)
    assert abs(tr["duration_s"] - r.duration_s) < 0.02 * r.duration_s
    tau = light.torque(tr["q"], tr["qd"], tr["qdd"])
    assert np.max(np.abs(tau) / light.torque_limit) <= 1.01
    bad = heavy.retime(path)
    assert bad.status == "TOPPRA_INFEASIBLE" and bad.torque_expansion > 1.0


@pytest.mark.skipif(not HAS_PIN, reason="needs the Pinocchio environment")
def test_independent_validator_static_and_payload_gravity():
    from research.timeopt_guided.independent_validate import build_model, torques

    q = np.array([[0.0, 0.6, 0.0, -0.5, 0.0, 1.2, 0.8]], np.float32)
    z = np.zeros_like(q)
    base = build_model({"mass_kg": 0.0, "com_m": [0, 0, 0], "radius_m": 0.06})
    load = build_model({"mass_kg": 2.0, "com_m": [0.08, 0, 0.16], "radius_m": 0.06})
    t0 = torques(*base, q, z, z)
    t1 = torques(*load, q, z, z)
    assert np.abs(t1 - t0).max() > 1.0


@pytest.mark.skipif(not HAS_CUDA_STACK, reason="needs CUDA cuRobo env with time_optimal_curobo on PYTHONPATH")
def test_phase1_lp_start_matches_standard_route_on_regular_path():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from research.timeopt_guided.k107_oracle import K107Oracle
    from research.timeopt_guided.toppra_audit import toppra_profile

    o = K107Oracle(Payload(2.0, (0, 0, 0.12)), warmup=False)
    q0 = np.array([0.0, -0.8, 0.0, -2.4, 0.0, 1.6, 0.8], np.float32)
    q1 = np.array([1.2, 0.3, 0.4, -1.2, 0.3, 2.2, 0.0], np.float32)
    path = rest_controls(q0, q1)
    ref = o.retime(path)
    assert ref.feasible
    terms = o.polish_terms(path)
    prof = toppra_profile(terms, 1)
    c0 = o._phase1_lp_start(terms, np.interp(terms.grid.cpu().numpy(), prof["s"], prof["x"]))
    assert c0 is not None and c0[0] == 0 and c0[-1] == 0
    res = o.k["polish_timing_law_pd_fast"](
        terms, torch.as_tensor(c0, device="cuda"), margin=0.995, fixed_ends=True, max_iterations=16,
        max_decrease=3, linear_solver="banded", sync_mode="fused", kernel_variant="split", restore_bisect=True)
    assert abs(float(res.duration) - ref.duration_s) < 0.03 * ref.duration_s
