# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idea C: alternating path/time refinement guided by the K107 retimer (CUDA env).

Start from a planner path (exact clamped cubic of the selected lane) and iterate:

1. K107 retimes the current geometry -> optimal squared-speed law x*(s), duration T.
2. Ask for a faster law x~ = sigma^2 x* (duration T / sigma). On the current path this
   violates the binding V/A/J/torque rows.
3. Move the free cubic controls (endpoint rest controls fixed) by Adam to minimise
       sum relu(|h(theta; x~)| / bound - 1)^2      (constraint excess at the faster law)
     + w_col * (cuRobo world+self collision cost, payload sphere included)
     + w_lim * joint-limit excess + w_prox * ||theta - theta_k||^2
   h is evaluated exactly on a dense s grid: q, q', q'', q''' from the spline, qd = q' sqrt(x),
   qdd = q'' x + q' x'/2, jerk = sqrt(x)(q''' x + 1.5 q'' x' + 0.5 q' x''), torque by
   differentiable cuRobo RNEA (same payload).
4. Accept theta if the collision screen passes and K107 retimes it FEASIBLY and faster;
   otherwise shrink sigma. Stop after ``iterations`` or when sigma < sigma_min.

The retimer decides the timing every iteration; gradients only shape the geometry,
so every accepted path is a genuine K107-feasible trajectory (independent validation
still required).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.interpolate import BSpline


@dataclass
class RefineConfig:
    iterations: int = 8
    sigma: float = 1.04  # requested speed-up per iteration
    sigma_min: float = 1.003
    inner_steps: int = 40
    lr: float = 2e-3  # rad per Adam step (control points)
    grid: int = 513
    w_col: float = 50.0
    w_lim: float = 100.0
    w_prox: float = 1e-2
    frozen: int = 3  # controls fixed at each end (rest boundary)
    time_budget_s: float | None = None


@dataclass
class RefineLog:
    durations: list = field(default_factory=list)  # accepted K107 durations (first = start)
    accepted: int = 0
    rejected: int = 0
    oracle_calls: int = 0
    wall_s: float = 0.0
    best_controls: np.ndarray | None = None
    best_result: object = None


class JointRefiner:
    def __init__(self, oracle, checker, position_limits: np.ndarray) -> None:
        self.o = oracle
        self.checker = checker
        lim = torch.as_tensor(np.asarray(position_limits, float)[:, :7], device="cuda", dtype=torch.float32)
        self.lo, self.hi = lim[0], lim[1]
        L = oracle.limits
        self.vmax, self.amax, self.jmax, self.tmax = (getattr(L, k).float() for k in
                                                       ("velocity", "acceleration", "jerk", "torque"))

    def _bases(self, knots: np.ndarray, n: int, s: np.ndarray, degree: int = 3):
        out = []
        for k in range(4):
            eye = np.eye(n)
            vals = np.stack([BSpline(knots, eye[i], degree)(s, nu=k) for i in range(n)], axis=1)
            out.append(torch.as_tensor(vals, device="cuda", dtype=torch.float32))
        return out

    def _law(self, result, s: np.ndarray, sigma: float):
        K = len(result.coefficients)
        xk = np.r_[np.zeros(3), np.linspace(0, 1, K - 2), np.ones(3)]
        spl = BSpline(xk, np.asarray(result.coefficients, float), 3)
        f = sigma**2
        return [torch.as_tensor(f * np.maximum(spl(s, nu=k), 0.0) if k == 0 else f * spl(s, nu=k),
                                device="cuda", dtype=torch.float32) for k in range(3)]

    def excess(self, theta, B, law):
        q, q1, q2, q3 = (b @ theta for b in B)
        x, xp, xpp = (v[:, None] for v in law)
        r = torch.sqrt(x.clamp_min(0.0))
        qd = q1 * r
        qdd = q2 * x + 0.5 * q1 * xp
        jerk = r * (q3 * x + 1.5 * q2 * xp + 0.5 * q1 * xpp)
        tau = self.o.model.dynamics.inverse_dynamics(q, qd, qdd)
        terms = [qd.abs() / self.vmax, qdd.abs() / self.amax, jerk[1:-1].abs() / self.jmax, tau.abs() / self.tmax]
        return q, sum((torch.relu(t - 1.0) ** 2).sum() for t in terms), max(float(t.max()) for t in terms)

    def refine(self, controls: np.ndarray, knots: np.ndarray, cfg: RefineConfig, *, start_result=None,
               degree: int = 3) -> RefineLog:
        t0 = time.perf_counter()
        log = RefineLog()
        n = len(controls)
        s = np.linspace(0, 1, cfg.grid)
        B = self._bases(knots, n, s, degree)
        res = start_result or self.o.retime(controls, degree=degree, knots=knots, diagnose_infeasible=False)
        log.oracle_calls += start_result is None
        if not res.feasible:
            log.wall_s = time.perf_counter() - t0
            return log
        log.durations.append(res.duration_s)
        log.best_controls, log.best_result = np.asarray(controls, float), res
        sigma = cfg.sigma
        self.checker.checker.setup_batch_tensors(1, cfg.grid)
        free = torch.zeros(n, 1, device="cuda")
        free[cfg.frozen:n - cfg.frozen] = 1.0
        for _ in range(cfg.iterations):
            if cfg.time_budget_s is not None and time.perf_counter() - t0 > cfg.time_budget_s:
                break
            law = self._law(log.best_result, s, sigma)
            base = torch.as_tensor(log.best_controls, device="cuda", dtype=torch.float32)
            th = base.clone().requires_grad_(True)
            opt = torch.optim.Adam([th], lr=cfg.lr)
            for _ in range(cfg.inner_steps):
                opt.zero_grad()
                q, exc, _ = self.excess(th, B, law)
                dw, ds = self.checker.checker.get_scene_self_collision_distance_from_joints(q[None].contiguous())
                col = dw.sum() + ds.sum()
                lim = (torch.relu(q - self.hi) ** 2 + torch.relu(self.lo - q) ** 2).sum()
                loss = exc + cfg.w_col * col + cfg.w_lim * lim + cfg.w_prox * ((th - base) ** 2).sum()
                loss.backward()
                th.grad *= free
                opt.step()
            cand = th.detach().cpu().numpy().astype(float)
            ok_geom = bool(self.checker.check_paths([cand], degree=degree, knots=knots)[0])
            new = self.o.retime(cand, degree=degree, knots=knots, diagnose_infeasible=False) if ok_geom else None
            log.oracle_calls += new is not None
            if new is not None and new.feasible and new.duration_s < log.durations[-1] - 1e-4:
                log.best_controls, log.best_result = cand, new
                log.durations.append(new.duration_s)
                log.accepted += 1
                sigma = min(cfg.sigma, 1.0 + 1.5 * (sigma - 1.0))  # regain step size after success
            else:
                log.rejected += 1
                sigma = 1.0 + 0.5 * (sigma - 1.0)
                if sigma < cfg.sigma_min:
                    break
        log.wall_s = time.perf_counter() - t0
        return log
