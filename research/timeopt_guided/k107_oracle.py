# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-process K107 TOPP-RA + primal-dual Newton retiming oracle (CUDA env).

Requires ``time_optimal_curobo/src`` on ``PYTHONPATH`` (pinned worktree, see
PLAN.md). Uses the SAME route as the ``joint_time_oracle_k107_bridge.py`` research
bridge: TOPP-RA seed fitted at K29, lifted to K107, Newton polish, online screen
on an event-aware 8193-point grid. Constraint contract: C3 without torque rate
(velocity, acceleration, jerk, torque), all at 0.95 of the ORIGINAL franka.yml
limits, rest-to-rest endpoints. This matches native cuRobo TimeOpt.

The oracle returns structured feedback for geometry search:
  * feasible: duration, per-family utilizations, and a bottleneck profile
    (where along s the time law is torque/velocity/acceleration-limited);
  * infeasible: a continuous infeasibility score ``torque_expansion`` (the
    smallest torque-limit multiplier at which TOPP-RA finds a law, bisection)
    and the static gravity-overload profile.
Nothing returned here is independently qualified; see ``independent_validate``.
"""

# ruff: noqa: F401  (K107 imports are collected via locals() in _import_k107)
from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .common import LIMIT_SCALE, Payload

_DEVICE = torch.device("cuda")


def _import_k107():
    """Import lazily so that pure-NumPy users of this package do not need K107."""
    from curobo_timeopt.benchmarks.certified_c3_reference import event_aware_grid
    from curobo_timeopt.curobo_adapter.dynamics import CuroboDynamicsAdapter
    from curobo_timeopt.curobo_adapter.robot_model import CuroboRobotModel
    from curobo_timeopt.path.bspline_path import BSplinePath, clamped_uniform_knots
    from curobo_timeopt.retiming.bspline.pdf_c3_fast_path import (
        build_pdf_c3_path_from_geometry,
        prepare_pdf_c3_geometry,
        screen_pdf_c3_candidates_fd5,
    )
    from curobo_timeopt.retiming.bspline.pdf_polish_fast_newton import (
        polish_timing_law_pd_fast,
    )
    from curobo_timeopt.retiming.bspline.pdf_timing_polish import (
        _state,
        prepare_polish_terms,
    )
    from curobo_timeopt.retiming.bspline.pdf_toppra_seed import toppra_newton_start_v2
    from curobo_timeopt.retiming.problem import RetimingProblem
    from curobo_timeopt.types import ConstraintMode, RobotLimits

    return dict(**locals())


@dataclass
class OracleResult:
    """Structured retiming feedback for one geometric path."""

    status: str  # FEASIBLE | SCREEN_REJECTED | NEWTON_START_FAILED | TOPPRA_INFEASIBLE | ERROR
    duration_s: float | None = None
    utilization: dict = field(default_factory=dict)
    # Bottleneck profile on a coarse s grid (len = n_profile):
    profile_s: np.ndarray | None = None
    torque_util: np.ndarray | None = None  # [n_profile, 7] |tau| / (0.95 tau_max)
    gravity_util: np.ndarray | None = None  # [n_profile, 7] |g| / (0.95 tau_max)
    time_density: np.ndarray | None = None  # dt/ds = 1/sqrt(x) at profile_s
    torque_expansion: float | None = None  # >1 needs larger torque limits (TOPP-RA)
    coefficients: np.ndarray | None = None  # K107 squared-speed controls (degree 3)
    timings_ms: dict = field(default_factory=dict)
    reason: str = ""

    @property
    def feasible(self) -> bool:
        return self.status == "FEASIBLE"

    def score(self) -> float:
        """Scalar for ranking: duration if feasible; otherwise 1e3 * (1 + violation)."""
        if self.feasible and self.duration_s is not None:
            return float(self.duration_s)
        if self.status == "SCREEN_REJECTED" and self.duration_s is not None:
            return 100.0 + float(self.duration_s)
        if self.status == "NEWTON_START_FAILED":
            # TOPP-RA found a law: closer to feasible than any TOPP-RA failure.
            return 500.0 + 100.0 * float(np.max(self.gravity_util) if self.gravity_util is not None else 1.0)
        e = self.torque_expansion
        if e is None or not math.isfinite(e):
            g = float(np.max(self.gravity_util)) if self.gravity_util is not None else 10.0
            e = max(g, 1.0) * 2.0
        return 1000.0 * (1.0 + max(e - 1.0, 0.0))

    def bottleneck(self) -> dict:
        """Most limiting (s, joint) by torque (feasible) or gravity (infeasible)."""
        prof = self.torque_util if self.feasible and self.torque_util is not None else self.gravity_util
        if prof is None:
            return {}
        i, j = np.unravel_index(int(np.argmax(prof)), prof.shape)
        return {"s": float(self.profile_s[i]), "joint": int(j), "util": float(prof[i, j])}

    def to_json(self, *, with_arrays: bool = False) -> dict:
        d = {
            "status": self.status,
            "duration_s": self.duration_s,
            "utilization": self.utilization,
            "torque_expansion": self.torque_expansion,
            "bottleneck": self.bottleneck(),
            "timings_ms": self.timings_ms,
            "reason": self.reason,
            "max_gravity_util": None if self.gravity_util is None else float(np.max(self.gravity_util)),
        }
        if with_arrays:
            for k in ("profile_s", "torque_util", "gravity_util", "time_density", "coefficients"):
                v = getattr(self, k)
                d[k] = None if v is None else np.asarray(v).tolist()
        return d


def clear_k107_caches() -> int:
    """Drop every module-level cache of the K107 package (dicts/OrderedDicts named *cache*/_PERSISTENT).

    Several are keyed by path or grid identity and grow without bound across many distinct
    paths (observed: ~1.5 GB GPU per benchmark problem). Call between problems."""
    import sys
    from collections import OrderedDict

    n = 0
    for name, mod in list(sys.modules.items()):
        if not name.startswith("curobo_timeopt") or mod is None:
            continue
        for attr, val in list(vars(mod).items()):
            if isinstance(val, (dict, OrderedDict)) and ("CACHE" in attr.upper() or attr == "_PERSISTENT"):
                n += len(val)
                val.clear()
    torch.cuda.empty_cache()
    return n


def apply_payload_to_dynamics(dynamics, payload: Payload) -> None:
    """Install the payload on a cuRobo ``Dynamics`` (or planner) via update_link_inertial."""
    p = payload.validate()
    if p.mass_kg <= 0:
        return
    dynamics.update_link_inertial(
        "attached_object",
        mass=float(p.mass_kg),
        com=torch.as_tensor(p.com_m, dtype=torch.float32),
        inertia=torch.as_tensor(np.r_[p.inertia_diag, 0.0, 0.0, 0.0], dtype=torch.float32),
    )


class K107Oracle:
    """One oracle per payload. Reuse it across all candidate paths of a cell."""

    def __init__(
        self,
        payload: Payload,
        *,
        grid: int = 2049,
        newton_iterations: int = 16,
        coefficients: int = 107,
        expansion_bisections: int = 8,
        n_profile: int = 257,
        warmup: bool = True,
        overload_aware_start: bool = False,
        start_method: str = "lp_then_tighten",
        lp_options: dict | None = None,
        inner_margins: tuple = (0.99, 0.98, 0.97, 0.95, 0.92, 0.9),
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("K107 oracle requires CUDA")
        self.k = _import_k107()
        self.payload = payload.validate()
        self.grid = grid
        self.iterations = newton_iterations
        self.K = coefficients
        self.bisections = expansion_bisections
        self.n_profile = n_profile
        self.overload_aware_start = overload_aware_start
        self.start_method = start_method
        self.lp_options = dict(lp_options or {})
        self.inner_margins = tuple(inner_margins)
        self.model, self.limits = self._prepare_model()
        self.calls = 0
        self.warmup_s = self._warmup() if warmup else 0.0

    def _warmup(self) -> float:
        """Exclude first-call compilation from timed requests (feasible + infeasible routes)."""
        t = time.perf_counter()
        q0 = np.array([0.0, -0.8, 0.0, -2.4, 0.0, 1.6, 0.8])
        q1 = np.array([0.6, -0.3, 0.2, -1.8, 0.2, 1.9, 0.4])
        from .common import rest_controls

        for _ in range(2):
            self.retime(rest_controls(q0, q1))
        self.calls = 0
        torch.cuda.synchronize()
        return time.perf_counter() - t

    # ---------------------------------------------------------------- model
    def _prepare_model(self):
        k = self.k
        model = k["CuroboRobotModel"].from_robot_yaml(
            "franka.yml", device="cuda", dtype=torch.float32
        )
        # NOTE: cuRobo caches the fixed-body-collapsed inertial model on the
        # KinematicsParams (``_dynamics_model``). Mutating ``link_masses_com`` and
        # rebuilding Dynamics does NOT take effect; use the Dynamics update API,
        # which refreshes the reduced model (same API as MotionPlanner).
        apply_payload_to_dynamics(model.dynamics._dynamics, self.payload)
        src = model.retiming_limits()
        limits = k["RobotLimits"](
            velocity=src.velocity * LIMIT_SCALE,
            acceleration=src.acceleration * LIMIT_SCALE,
            torque=src.torque * LIMIT_SCALE,
            jerk=src.jerk * LIMIT_SCALE,
            torque_rate=src.torque_rate,  # not enforced in C3_NO_TR
        )
        return model, limits

    def gravity(self, q: np.ndarray) -> np.ndarray:
        """Gravity torque [N, 7] under this payload (cuRobo RNEA)."""
        qt = torch.as_tensor(np.asarray(q, float), device=_DEVICE, dtype=torch.float32)
        z = torch.zeros_like(qt)
        return self.model.dynamics.inverse_dynamics(qt, z, z).detach().cpu().numpy()

    def torque(self, q, qd, qdd) -> np.ndarray:
        t = [torch.as_tensor(np.asarray(a, float), device=_DEVICE, dtype=torch.float32) for a in (q, qd, qdd)]
        return self.model.dynamics.inverse_dynamics(*t).detach().cpu().numpy()

    @property
    def torque_limit(self) -> np.ndarray:
        return self.limits.torque.detach().cpu().numpy()

    # ----------------------------------------------------------------- path
    def make_path(self, controls: np.ndarray, degree: int = 5, knots: np.ndarray | None = None):
        c = torch.as_tensor(np.asarray(controls, float), device=_DEVICE, dtype=torch.float32)
        kt = None if knots is None else torch.as_tensor(np.asarray(knots, float), device=_DEVICE, dtype=torch.float32)
        return self.k["BSplinePath"](c, degree=degree, knots=kt)

    def polish_terms(self, controls, *, degree: int = 5, knots=None, samples: int | None = None):
        """K107 path terms (same problem and contract as ``retime``) for diagnostics."""
        k = self.k
        path = self.make_path(controls, degree, knots)
        problem = k["RetimingProblem"](
            path, self.limits, torch.linspace(0, 1, 1025, device=_DEVICE), self.model.dynamics,
            mode=k["ConstraintMode"].c3_no_torque_rate(), initial_speed=0.0, final_speed=0.0,
            feasibility_tolerance=5e-4,
        )
        return k["prepare_polish_terms"](problem, self.K, samples=samples or self.grid)

    # --------------------------------------------------------------- retime
    def retime(
        self, controls: np.ndarray, *, degree: int = 5, knots: np.ndarray | None = None,
        diagnose_infeasible: bool = True,
    ) -> OracleResult:
        """Retime one rest-to-rest B-spline path. Never raises for numerical failures."""
        self.calls += 1
        c = np.asarray(controls, float)
        if c.ndim != 2 or c.shape[1] != 7 or not np.isfinite(c).all():
            return OracleResult("ERROR", reason="invalid controls")
        t0 = time.perf_counter()
        try:
            return self._retime(c, degree, knots, diagnose_infeasible, t0)
        except Exception as error:  # numerical errors are NOT infeasibility proofs
            return OracleResult(
                "ERROR", reason=f"{type(error).__name__}: {str(error)[:300]}",
                timings_ms={"total": (time.perf_counter() - t0) * 1e3},
            )

    def _retime(self, c, degree, knots, diagnose_infeasible, t0) -> OracleResult:
        k = self.k
        path = self.make_path(c, degree, knots)
        problem = k["RetimingProblem"](
            path, self.limits, torch.linspace(0, 1, 1025, device=_DEVICE), self.model.dynamics,
            mode=k["ConstraintMode"].c3_no_torque_rate(), initial_speed=0.0, final_speed=0.0,
            feasibility_tolerance=5e-4,
        )
        terms = k["prepare_polish_terms"](problem, self.K, samples=self.grid)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ends = torch.zeros(2, device=_DEVICE, dtype=torch.float32)
        seed, seed_diag = k["toppra_newton_start_v2"](
            terms, ends, margin=0.995, fixed_ends=True, stride=3, jerk_cap=True, fit_k=29,
            batched_scale=True, scale_backend="torch",
        )
        t2 = time.perf_counter()
        prof_idx = np.linspace(0, terms.grid.numel() - 1, self.n_profile).round().astype(int)
        s_prof = terms.grid[prof_idx].detach().cpu().numpy()
        lim_tau = terms.limit[2].detach().cpu().numpy()
        g = terms.const[2][prof_idx].detach().cpu().numpy()
        gravity_util = np.abs(g) / lim_tau[None]
        timings = {"prepare": (t1 - t0) * 1e3, "toppra": (t2 - t1) * 1e3}
        if seed is None:
            res = OracleResult(
                "TOPPRA_INFEASIBLE", profile_s=s_prof, gravity_util=gravity_util,
                reason=str(seed_diag.get("status", "")) + " " + str(seed_diag.get("reason", "")),
            )
            if diagnose_infeasible:
                res.torque_expansion = self._torque_expansion(terms, ends)
            timings["total"] = (time.perf_counter() - t0) * 1e3
            res.timings_ms = timings
            return res
        start_route = "toppra_slowdown_restore"
        err_msg = ""
        try:
            result = k["polish_timing_law_pd_fast"](
                terms, seed, margin=0.995, fixed_ends=True, max_iterations=self.iterations,
                max_decrease=3, linear_solver="banded", sync_mode="fused", kernel_variant="split",
                restore_bisect=True,
            )
        except ValueError as error:
            err_msg = f"{type(error).__name__}: {error}"
            result = self._overload_aware_start(terms, ends) if self.overload_aware_start else None
            if result is not None:
                result, start_route = result
        if result is None:
            # TOPP-RA (stride-3 grid, 2nd order + jerk cap) found a law, but its
            # K29 fit cannot be scaled into Newton's strict interior. Typical when
            # the path crosses a static gravity overload that must be "swung"
            # through at speed. Unresolved: neither a certificate nor a rescue.
            return OracleResult(
                "NEWTON_START_FAILED", profile_s=s_prof, gravity_util=gravity_util,
                reason=err_msg[:300],
                timings_ms=timings | {"total": (time.perf_counter() - t0) * 1e3},
            )
        coeff = result.coefficients.to(torch.float32)
        t3 = time.perf_counter()
        audit_grid = k["event_aware_grid"](path, problem.s, 8193, 0.01)
        prepared = k["prepare_pdf_c3_geometry"](problem, audit_grid, prepare_fixed_rnea_cache=False)
        screen_path = k["build_pdf_c3_path_from_geometry"](prepared, self.K, torque_rate_backend="native_jvp")
        check = k["screen_pdf_c3_candidates_fd5"](screen_path, coeff[None], torque_rate_backend="native_jvp")
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        # Bottleneck profile from the polished law on the polish grid.
        support = terms.start[:, None] + torch.arange(4, device=terms.start.device)[None]
        state = k["_state"](result.coefficients.to(terms.coef.dtype)[None], terms.basis, support)[0]
        x = state[:, 0].clamp_min(1e-12)
        tau = (terms.coef[2] * state[:, None, :]).sum(-1) + terms.const[2]
        torque_util = (tau.abs() / terms.limit[2][None])[prof_idx].detach().cpu().numpy()
        density = (1.0 / x.sqrt())[prof_idx].detach().cpu().numpy()
        duration = float(result.duration)
        ok = bool(check.feasible[0]) and bool(check.positive_timing[0]) and math.isfinite(duration)
        util = {f: float(getattr(check, "rho_" + f)[0]) for f in ("velocity", "acceleration", "torque", "jerk")}
        timings.update(newton=(t3 - t2) * 1e3, screen=(t4 - t3) * 1e3, total=(time.perf_counter() - t0) * 1e3)
        timings["start_route"] = start_route
        return OracleResult(
            "FEASIBLE" if ok else "SCREEN_REJECTED",
            duration_s=duration if math.isfinite(duration) else None,
            utilization=util, profile_s=s_prof, torque_util=torque_util,
            gravity_util=gravity_util, time_density=density,
            coefficients=coeff.detach().cpu().numpy(), timings_ms=timings,
            torque_expansion=1.0 if ok else None,
        )

    def _phase1_lp_start(self, terms, x_ref_grid: np.ndarray, *, min_slack: float = 0.003,
                         rounds: int = 1, lp_stride: int = 4, max_cuts: int = 1,
                         single_lp: bool = True, lp_time_limit_s: float = 20.0):
        """Phase-I start in K107's own timing-spline space (sequential LP, HiGHS).

        Variables: the K timing coefficients c (ends fixed to 0) and a relative
        slack t. Velocity^2, acceleration and torque rows of K107's barrier are
        LINEAR in (x, x_u, x_uu) = basis @ c, hence in c. The jerk row is
        lin * sqrt(x); it is linearized as |lin| <= (1 - t) * bound / sqrt(x_ref) and
        re-linearized at the LP solution (``rounds`` times). LP 1 maximizes t; LP 2
        maximizes the sum of x (speed) subject to t >= min(t*/2, min_slack).
        Cutting planes: solve on every ``lp_stride``-th grid point, add the violated
        rows of the full grid, re-solve (<= ``max_cuts`` times). Returns c or None.
        Unlike scaling x down, this never leaves the convex feasible set to restore
        interiority, so it works through static gravity overload.
        """
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix, hstack, vstack

        K = terms.coefficients
        G_full = terms.grid.numel()
        sel = np.arange(G_full)
        if max_cuts == 1:  # coarse-only LP: build rows for the selected points only
            sel = np.unique(np.r_[np.arange(0, G_full, lp_stride), G_full - 1])
            lp_stride = 1
        tsel = torch.as_tensor(sel, device=terms.grid.device)
        basis = terms.basis[:, tsel].detach().cpu().numpy()  # (3, G, 4)
        G = basis.shape[1]
        idx = terms.start[tsel].detach().cpu().numpy()[:, None] + np.arange(4)[None]
        coef = terms.coef[:, tsel].detach().cpu().numpy()  # (F, G, D, 3)
        const = terms.const[:, tsel].detach().cpu().numpy()  # (F, G, D)
        x_ref_grid = np.asarray(x_ref_grid, float)[sel]
        lim = terms.limit.detach().cpu().numpy().copy()
        bound = 0.995 * lim
        bound[0] = 0.995**2 * lim[0]
        F, _, D, _ = coef.shape
        free = np.arange(1, K - 1)
        colmap = -np.ones(K, dtype=int)
        colmap[free] = np.arange(len(free))
        nfree = len(free)

        def rows(w, cols):  # w (N, R, 4) weights on columns cols (N, 4)
            N, R, _ = w.shape
            r = np.arange(N * R).reshape(N, R)
            RR = np.broadcast_to(r[:, :, None], w.shape)
            CC = np.broadcast_to(colmap[cols][:, None, :], w.shape)
            keep = CC >= 0
            return coo_matrix((w[keep], (RR[keep], CC[keep])), shape=(N * R, nfree)).tocsr()

        x_rows = rows(basis[0][:, None, :], idx)
        msel = torch.as_tensor(sel[:-1], device=terms.grid.device)
        mid = terms.mid_basis[msel].detach().cpu().numpy()
        mid_idx = terms.mid_start[msel].detach().cpu().numpy()[:, None] + np.arange(4)[None]
        mid_rows = rows(mid[:, None, :], mid_idx)
        lin_rows = [rows(np.einsum("gdk,kgj->gdj", coef[f], basis), idx) for f in range(F)]
        point_of = np.repeat(np.arange(G), D)
        x_ref = np.maximum(np.asarray(x_ref_grid, float), 1e-9)
        coarse = (np.arange(G) % lp_stride == 0) | (np.arange(G) == G - 1)
        c_best = None
        self._lp_log = []
        for _ in range(rounds):
            A_ub, b_ub, slack, pts = [], [], [], []
            for f in range(F):
                cst = const[f].reshape(-1)
                bnd = np.broadcast_to(bound[f][None], (G, D)).copy()
                if f == 3:  # jerk: root family, linearized at x_ref
                    bnd = bnd / np.sqrt(x_ref)[:, None]
                    bnd[[0, -1]] = 1e9  # ends are not constrained (root_mask)
                bnd = bnd.reshape(-1)
                A_ub += [lin_rows[f], -lin_rows[f]]
                b_ub += [bnd - cst, bnd + cst]
                slack += [bnd, bnd]
                pts += [point_of, point_of]
            A_ub += [-x_rows[1:-1], -mid_rows]
            b_ub += [np.zeros(G - 2), np.zeros(mid_rows.shape[0])]
            slack += [np.full(G - 2, 1e-6), np.full(mid_rows.shape[0], 1e-6)]
            pts += [np.arange(1, G - 1), np.minimum(np.arange(mid_rows.shape[0]), G - 1)]
            A = hstack([vstack(A_ub), coo_matrix(np.concatenate(slack)[:, None])]).tocsr()
            b = np.concatenate(b_ub)
            active = coarse[np.concatenate(pts)]

            def solve(obj, bounds):
                act = active.copy()
                for it in range(max_cuts):
                    # time_limit: HiGHS can cycle indefinitely on degenerate instances (observed);
                    # a timeout is treated as "no Phase-I start" (status != 0).
                    res = linprog(obj, A_ub=A[act], b_ub=b[act], bounds=bounds, method="highs",
                                  options={"time_limit": float(lp_time_limit_s)})
                    if res.status != 0:
                        self._lp_log.append(("lp_status", it, res.status, int(act.sum())))
                        return None
                    if max_cuts == 1:
                        return res.x  # coarse-grid LP only; tiny residuals are left to K107's restore
                    viol = (A @ res.x - b) > 1e-12
                    new = viol & ~act
                    self._lp_log.append(("cut", it, int(act.sum()), int(new.sum())))
                    if not new.any():
                        return res.x
                    act |= new
                return None

            obj2 = np.r_[-np.asarray(mid_rows.sum(axis=0)).ravel(), 0.0]
            z = solve(obj2, [(None, None)] * nfree + [(min_slack, 0.5)]) if single_lp else None
            if z is None:  # two-stage: find the achievable slack first
                obj1 = np.zeros(nfree + 1)
                obj1[-1] = -1.0
                z1 = solve(obj1, [(None, None)] * nfree + [(None, 0.5)])
                if z1 is None or z1[-1] <= 0:
                    return c_best
                t_req = min(0.5 * z1[-1], min_slack)
                z2 = solve(obj2, [(None, None)] * nfree + [(t_req, 0.5)])
                z = z2 if z2 is not None else z1
            c = np.zeros(K)
            c[free] = z[:-1]
            c_best = c
            x_ref = np.maximum(x_rows @ z[:-1], 1e-9)
        return c_best

    def _overload_aware_start(self, terms, ends):
        """Newton start strictly inside the limits WITHOUT slowing down.

        The TOPP-RA feasible set is convex in the squared speed x(s). Instead of
        restoring the start toward x = 0 (infeasible under static gravity
        overload), re-solve TOPP-RA with all limits tightened by r < 1; its law is
        strictly interior w.r.t. the original limits. Largest r first (fastest law);
        stop when TOPP-RA becomes infeasible (the overload band is thinner than r).
        Cost: one TOPP-RA solve (+ fit) per tried r.
        """
        k = self.k
        if self.start_method in ("lp", "lp_then_tighten"):
            from .toppra_audit import toppra_profile

            prof = toppra_profile(terms, 1)
            if prof["feasible"]:
                grid = terms.grid.detach().cpu().numpy()
                c0 = self._phase1_lp_start(terms, np.interp(grid, prof["s"], prof["x"]), **self.lp_options)
                if c0 is not None:
                    try:
                        res = k["polish_timing_law_pd_fast"](
                            terms, torch.as_tensor(c0, device=terms.grid.device), margin=0.995,
                            fixed_ends=True, max_iterations=self.iterations, max_decrease=3,
                            linear_solver="banded", sync_mode="fused", kernel_variant="split",
                            restore_bisect=True,
                        )
                        return res, "phase1_lp"
                    except ValueError:
                        pass
            if self.start_method == "lp":
                return None
        base = terms.limit
        for r in self.inner_margins:
            scale = torch.tensor([r * r, r, r, r], device=base.device, dtype=base.dtype)[:, None]
            tight = dataclasses.replace(terms, limit=base * scale)
            seed, _ = k["toppra_newton_start_v2"](
                tight, ends, margin=0.995, fixed_ends=True, stride=3, jerk_cap=True, fit_k=29,
                batched_scale=True, scale_backend="torch",
            )
            if seed is None:
                return None
            try:
                res = k["polish_timing_law_pd_fast"](
                    terms, seed, margin=0.995, fixed_ends=True, max_iterations=self.iterations,
                    max_decrease=3, linear_solver="banded", sync_mode="fused", kernel_variant="split",
                    restore_bisect=True,
                )
                return res, f"tightened_toppra_r{r:g}"
            except ValueError:
                continue
        return None

    def _torque_expansion(self, terms, ends) -> float:
        """Smallest torque-limit multiplier lambda in [1, 64] for which TOPP-RA succeeds."""
        k = self.k
        base = terms.limit.clone()

        def solvable(lam: float) -> bool:
            lim = base.clone()
            lim[2] = base[2] * lam
            seed, _ = k["toppra_newton_start_v2"](
                dataclasses.replace(terms, limit=lim), ends, margin=0.995, fixed_ends=True,
                stride=3, jerk_cap=True, fit_k=29, batched_scale=False, scale_backend="torch",
            )
            return seed is not None

        lo, hi = 1.0, 2.0
        while not solvable(hi):
            lo, hi = hi, hi * 2.0
            if hi > 64:
                return float("inf")
        for _ in range(self.bisections):
            mid = math.sqrt(lo * hi)
            lo, hi = (lo, mid) if solvable(mid) else (mid, hi)
        return float(hi)

    # ---------------------------------------------------------- sampling
    def sample_trajectory(self, controls, result: OracleResult, *, degree: int = 5,
                          knots: np.ndarray | None = None, n: int = 8193) -> dict:
        """Dense time-uniform samples of the retimed trajectory (t, q, qd, qdd, jerk).

        Integrates t(s) = int ds / sqrt(x) on a fine s grid (event-safe trapezoid with
        sqrt endpoint correction), then resamples uniformly in time. For export and
        independent validation, not for the oracle decision itself.
        """
        if result.coefficients is None:
            raise ValueError("no timing law")
        from scipy.interpolate import BSpline

        K = len(result.coefficients)
        xk = np.r_[np.zeros(3), np.linspace(0, 1, K - 2), np.ones(3)]
        xs = BSpline(xk, np.asarray(result.coefficients, float), 3)
        s = np.linspace(0, 1, 200001)
        x = np.maximum(xs(s), 0.0)
        # t(s): rest endpoints have x ~ s^p; integrate with midpoint rule on sqrt.
        root = np.sqrt(np.maximum(xs(0.5 * (s[1:] + s[:-1])), 1e-300))
        t = np.r_[0.0, np.cumsum(np.diff(s) / root)]
        T = float(t[-1])
        tu = np.linspace(0, T, n)
        su = np.interp(tu, t, s)
        path = self.make_path(controls, degree, knots)
        st = torch.as_tensor(su, device=_DEVICE, dtype=torch.float32)
        q, qs, qss, qsss = (v.detach().cpu().numpy().astype(float) for v in path.evaluate_all(st, 3))
        x = np.maximum(xs(su), 0.0)
        xp = xs.derivative(1)(su)
        xpp = xs.derivative(2)(su)
        r = np.sqrt(x)
        qd = qs * r[:, None]
        qdd = qss * x[:, None] + 0.5 * qs * xp[:, None]
        jerk = r[:, None] * (qsss * x[:, None] + 1.5 * qss * xp[:, None] + 0.5 * qs * xpp[:, None])
        return {"time": tu, "s": su, "q": q, "qd": qd, "qdd": qdd, "jerk": jerk, "duration_s": T}
