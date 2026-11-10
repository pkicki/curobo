# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TimeOpt-in-the-loop geometric path search (CUDA env).

Arms (identical proposal budget, identical first round, identical oracle):
  B  "blind"    global random detours; the oracle only SCORES candidates.
  C  "ranked"   oracle score selects elites; children are global + elite-local
                perturbations (a TimeOpt-ranked evolutionary search).
  D  "guided"   C plus bottleneck-directed detours: the oracle's bottleneck
                (s*, joint j*) and a gravity-gradient direction decide WHERE and
                HOW to bend the path. A global fraction is always retained
                (local LP-gradient-only guidance failed in the 2026-11-08 study).

Score (lower is better): feasible -> K107 duration; infeasible -> 1000 *
(1 + torque_expansion - 1) so that infeasible candidates are ordered by how much
torque they would additionally need (continuous feedback, not a binary flag).
Collision-invalid proposals consume budget but no oracle call.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .common import rest_controls
from .geometry import directed_proposal, evaluate, global_proposal, local_proposal

ARMS = ("B", "C", "D")


@dataclass
class SearchConfig:
    arm: str = "D"
    max_proposals: int = 128
    batch: int = 16
    elites: int = 4
    global_fraction: float = 0.35  # C/D: share of each later batch from blind global
    directed_fraction: float = 0.4  # D: share of each later batch that is bottleneck-directed
    amplitudes: tuple = (0.3, 0.6, 1.0, 1.5)  # rad, global detours
    local_amplitude: float = 0.08  # rad, elite refinement (decays to 0.25x)
    directed_amplitude: float = 0.4  # rad
    time_budget_s: float | None = None  # full wall time incl. collision + oracle
    stop_on_first_feasible: bool = False
    seed: int = 0

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")


@dataclass
class Candidate:
    index: int
    round: int
    kind: str  # seed | global | local | directed
    parent: int | None
    controls: np.ndarray
    collision_free: bool = False
    result: object = None  # OracleResult
    t_done_s: float = 0.0  # wall time since search start when evaluated

    @property
    def score(self) -> float:
        if not self.collision_free or self.result is None:
            return float("inf")
        return float(self.result.score())

    def to_json(self) -> dict:
        r = self.result
        return {
            "index": self.index, "round": self.round, "kind": self.kind, "parent": self.parent,
            "collision_free": self.collision_free, "t_done_s": self.t_done_s,
            "oracle": None if r is None else r.to_json(),
        }


@dataclass
class SearchLog:
    config: dict
    candidates: list = field(default_factory=list)
    best_index: int | None = None
    wall_s: float = 0.0
    oracle_calls: int = 0
    collision_checks: int = 0
    first_feasible_s: float | None = None
    first_feasible_proposals: int | None = None

    def best(self) -> Candidate | None:
        return None if self.best_index is None else self.candidates[self.best_index]

    def anytime(self) -> list[dict]:
        """Best feasible duration after each proposal (for budget curves)."""
        best, out = float("inf"), []
        for c in self.candidates:
            if c.result is not None and c.result.feasible:
                best = min(best, c.result.duration_s)
            out.append({"proposals": c.index + 1, "t_s": c.t_done_s,
                        "best_duration_s": None if not np.isfinite(best) else best})
        return out

    def to_json(self) -> dict:
        b = self.best()
        return {
            "config": self.config, "wall_s": self.wall_s, "oracle_calls": self.oracle_calls,
            "collision_checks": self.collision_checks, "best_index": self.best_index,
            "best_duration_s": None if b is None or not b.result.feasible else b.result.duration_s,
            "first_feasible_s": self.first_feasible_s,
            "first_feasible_proposals": self.first_feasible_proposals,
            "candidates": [c.to_json() for c in self.candidates],
            "anytime": self.anytime(),
        }


class GuidedSearch:
    """One search instance per (task, payload); oracle and checker are reused."""

    def __init__(self, oracle, checker, position_limits: np.ndarray) -> None:
        self.oracle = oracle
        self.checker = checker
        lo, hi = np.asarray(position_limits, float)[:, :7]
        pad = 0.02  # stay inside the joint position limits
        self.lower, self.upper = lo + pad, hi - pad

    # --------------------------------------------------------------- feedback
    def gravity_gradient(self, q: np.ndarray, joint: int, eps: float = 1e-3) -> np.ndarray:
        """d|g_joint|/dq at configuration q by central differences (cuRobo RNEA)."""
        q = np.asarray(q, float)
        pts = np.concatenate([q[None] + eps * np.eye(7), q[None] - eps * np.eye(7), q[None]])
        g = self.oracle.gravity(pts)[:, joint]
        sign = np.sign(g[-1]) or 1.0
        return sign * (g[:7] - g[7:14]) / (2 * eps)

    def directed(self, parent: Candidate, rng: np.random.Generator, start, goal, amp: float) -> np.ndarray:
        r = parent.result
        info = r.bottleneck() if r is not None else {}
        if not info:
            return global_proposal(parent.controls, rng, amplitude=amp, lower=self.lower, upper=self.upper)
        s_star, joint, util = info["s"], info["joint"], info["util"]
        q_star = evaluate(parent.controls, np.array([s_star]))[0]
        torque_limited = (not r.feasible) or util >= 0.9
        if torque_limited:
            direction = -self.gravity_gradient(q_star, joint)
        else:
            # Kinematically limited: bend toward the chord (shorter path).
            chord = evaluate(rest_controls(start, goal, len(parent.controls)), np.array([s_star]))[0]
            direction = chord - q_star
        if not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-9:
            direction = rng.normal(size=7)
        return directed_proposal(parent.controls, rng, s_center=s_star, direction=direction,
                                 amplitude=amp, lower=self.lower, upper=self.upper)

    # ----------------------------------------------------------------- search
    def run(self, start: np.ndarray, goal: np.ndarray, cfg: SearchConfig,
            seeds: list[np.ndarray] | None = None) -> SearchLog:
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        base = rest_controls(start, goal)
        log = SearchLog(config=asdict(cfg))
        t0 = time.perf_counter()
        shared = np.random.default_rng([cfg.seed, 0])  # identical first round for all arms
        blind = np.random.default_rng([cfg.seed, 1])  # identical blind stream for all arms
        guided = np.random.default_rng([cfg.seed, 2 + ARMS.index(cfg.arm)])
        cand: list[Candidate] = log.candidates

        def add(controls, kind, parent, rnd):
            cand.append(Candidate(len(cand), rnd, kind, parent, np.asarray(controls, float)))

        def blind_global(rng, rnd):
            amp = float(rng.choice(cfg.amplitudes))
            add(global_proposal(base, rng, amplitude=amp, lower=self.lower, upper=self.upper), "global", None, rnd)

        # Round 0: seeds + shared global proposals.
        for s in [base] + list(seeds or []):
            if len(cand) < cfg.batch:
                add(s, "seed", None, 0)
        while len(cand) < min(cfg.batch, cfg.max_proposals):
            blind_global(shared, 0)
        self._evaluate(log, 0, t0, cfg)
        rnd = 0
        while len(cand) < cfg.max_proposals and not self._stop(log, cfg, t0):
            rnd += 1
            n = min(cfg.batch, cfg.max_proposals - len(cand))
            first_new = len(cand)
            if cfg.arm == "B":
                for _ in range(n):
                    blind_global(blind, rnd)
            else:
                ranked = sorted((c for c in cand if c.collision_free and c.result is not None),
                                key=lambda c: c.score)
                elites = ranked[: cfg.elites]
                n_global = int(round(cfg.global_fraction * n))
                n_dir = int(round(cfg.directed_fraction * n)) if cfg.arm == "D" else 0
                for _ in range(n_global):
                    blind_global(blind, rnd)
                progress = len(cand) / cfg.max_proposals
                amp_local = cfg.local_amplitude * (1.0 - 0.75 * progress)
                k = 0
                while len(cand) - first_new < n:
                    if not elites:
                        blind_global(blind, rnd)
                        continue
                    parent = elites[k % len(elites)]
                    k += 1
                    if n_dir > 0:
                        n_dir -= 1
                        amp = cfg.directed_amplitude * float(guided.uniform(0.5, 1.5))
                        add(self.directed(parent, guided, start, goal, amp), "directed", parent.index, rnd)
                    elif parent.result is not None and not parent.result.feasible and guided.random() < 0.5:
                        # Infeasible elites need larger moves than local refinement.
                        amp = float(guided.choice(cfg.amplitudes[:2]))
                        add(global_proposal(parent.controls, guided, amplitude=amp, lower=self.lower,
                                            upper=self.upper), "elite_global", parent.index, rnd)
                    else:
                        add(local_proposal(parent.controls, guided, amplitude=amp_local, lower=self.lower,
                                           upper=self.upper), "local", parent.index, rnd)
            self._evaluate(log, first_new, t0, cfg)
        log.wall_s = time.perf_counter() - t0
        feas = [c for c in cand if c.result is not None and c.result.feasible]
        pool = feas or [c for c in cand if c.collision_free and c.result is not None]
        if pool:
            log.best_index = min(pool, key=lambda c: c.score).index
        return log

    def _stop(self, log: SearchLog, cfg: SearchConfig, t0: float) -> bool:
        if cfg.time_budget_s is not None and time.perf_counter() - t0 >= cfg.time_budget_s:
            return True
        return cfg.stop_on_first_feasible and log.first_feasible_s is not None

    def _evaluate(self, log: SearchLog, first: int, t0: float, cfg: SearchConfig) -> None:
        new = log.candidates[first:]
        ok = self.checker.check_paths([c.controls for c in new])
        log.collision_checks += len(new)
        for c, valid in zip(new, ok):
            if cfg.time_budget_s is not None and time.perf_counter() - t0 >= cfg.time_budget_s:
                c.t_done_s = time.perf_counter() - t0
                continue  # budget exhausted: proposal generated but not evaluated
            c.collision_free = bool(valid)
            if valid:
                c.result = self.oracle.retime(c.controls)
                log.oracle_calls += 1
                if c.result.feasible and log.first_feasible_s is None:
                    log.first_feasible_s = time.perf_counter() - t0
                    log.first_feasible_proposals = c.index + 1
            c.t_done_s = time.perf_counter() - t0
