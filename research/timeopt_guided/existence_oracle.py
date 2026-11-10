# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Existence oracle: constructive certificates that a task x payload is SOLVABLE (CUDA env).

Sufficient condition (quasi-static existence). Let A_rho be the set of configurations
that are collision free with clearance margin (world + self, payload sphere), inside the
joint limits with margin, and whose STATIC gravity torque satisfies
|g_j(q)| <= rho * 0.95 * tau_max_j (rho < 1). If start and goal are connected by a
polygonal joint-space path inside A_rho, executing each segment rest-to-rest with a
quintic time scaling of duration T_i gives velocity ~ 1/T_i, acceleration ~ 1/T_i^2,
jerk ~ 1/T_i^3 and torque -> g(q) <= rho * 0.95 * tau_max as T_i grows, so finite T_i
satisfying ALL Scaled95 constraints exist. The witness is that trajectory.

The search is deliberately independent of every method under evaluation: a lazy PRM
(uniform samples, k-NN, edges checked only when on a candidate shortest path), no
TrajOpt / cuRobo RRT / TimeOpt / K107. Witnesses are certified only by
``independent_validate --gap-check`` (Pinocchio, dense, inter-sample sphere-motion margin).
Failure to connect is UNCERTIFIED, never "infeasible".
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .common import JOINT_NAMES, LIMIT_SCALE, Payload, read_json, write_json


@dataclass
class OracleConfig:
    rhos: tuple = (0.80, 0.90, 0.97)  # gravity ladder, fraction of Scaled95 torque
    samples: tuple = (50_000, 200_000)  # PRM sample ladder per rho
    k: int = 12  # nearest neighbours
    max_edge: float = 1.2  # rad (joint-space L2)
    step: float = 0.01  # rad, max joint step (L_inf) when checking edges
    margins: tuple = (0.015, 0.005)  # collision clearance margin ladder (m)
    joint_margin: float = 0.03  # rad
    endpoint_radius: float = 0.3  # rad (L2): problem constraints only, near start/goal
    shortcut_tries: int = 200
    max_lazy_iters: int = 400
    time_safety: float = 0.9  # V/A/J/tau target fraction of Scaled95 for the time scaling
    seed: int = 0


class Admissibility:
    """Batched A_rho membership on the GPU (cuRobo collision + RNEA gravity)."""

    def __init__(self, scene: str, payload: Payload, cfg: OracleConfig, margin_m: float) -> None:
        import torch
        from curobo._src.robot.dynamics.dynamics import Dynamics
        from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
        from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
        from curobo._src.types.device_cfg import DeviceCfg

        from .collision_screen import PathCollisionChecker
        from .k107_oracle import apply_payload_to_dynamics

        self.cfg = cfg
        self.margin_m = margin_m
        # Edge resolution shrinks with the margin so sphere motion between checks stays below it.
        self.step = cfg.step * margin_m / max(cfg.margins)
        self.checker = PathCollisionChecker(scene, margin_m=margin_m)
        self.checker.set_payload(payload)
        kcfg = KinematicsCfg.from_robot_yaml_file("franka.yml")
        self.kp = kcfg.kinematics_config
        dev = DeviceCfg(device=torch.device("cuda"), dtype=torch.float32)
        self.dyn = Dynamics(DynamicsCfg(kinematics_config=self.kp, device_cfg=dev, gravity=[0.0, 0.0, -9.81]))
        apply_payload_to_dynamics(self.dyn, payload)
        jl = self.kp.joint_limits
        pos = jl.position.detach().cpu().numpy()[:, :7]
        self.lo, self.hi = pos[0] + cfg.joint_margin, pos[1] - cfg.joint_margin
        self.lo_raw, self.hi_raw = pos[0], pos[1]
        self.anchors = (np.full(7, np.inf), np.full(7, np.inf))
        self.tau_max = np.abs(jl.effort.detach().cpu().numpy()[:, :7]).min(axis=0)
        to_np = lambda v: np.abs(v.detach().cpu().numpy()).min(axis=0)[:7] if v.ndim == 2 else np.abs(v.detach().cpu().numpy())[:7]  # noqa: E731
        self.vmax, self.amax, self.jmax = (to_np(getattr(jl, k)) for k in ("velocity", "acceleration", "jerk"))
        self.calls = 0

    def torque(self, q: np.ndarray, qd: np.ndarray | None = None, qdd: np.ndarray | None = None) -> np.ndarray:
        import torch
        from curobo.types import JointState

        out = []
        for i in range(0, len(q), 65536):
            qt = torch.as_tensor(q[i:i + 65536], device="cuda", dtype=torch.float32)
            z = torch.zeros_like(qt)
            v = z if qd is None else torch.as_tensor(qd[i:i + 65536], device="cuda", dtype=torch.float32)
            a = z if qdd is None else torch.as_tensor(qdd[i:i + 65536], device="cuda", dtype=torch.float32)
            self.dyn.setup_batch_size(qt.shape[0])
            js = JointState(position=qt, velocity=v, acceleration=a, joint_names=list(JOINT_NAMES))
            out.append(self.dyn.compute_inverse_dynamics(js).detach().cpu().numpy()[:, :7])
        return np.concatenate(out)

    def gravity_util(self, q: np.ndarray) -> np.ndarray:
        """max_j |g_j(q)| / (0.95 tau_max_j)."""
        return np.max(np.abs(self.torque(q)) / (LIMIT_SCALE * self.tau_max), axis=1)

    def admissible(self, q: np.ndarray, rho: float, relaxed: np.ndarray | None = None) -> np.ndarray:
        """A_rho membership. ``relaxed`` (bool per point): use the PROBLEM's own constraints
        (no clearance margin, raw joint limits) instead of the oracle margins; used only
        within ``endpoint_radius`` of the start/goal, which the bank admitted with smaller
        margins. Gravity <= rho always applies."""
        q = np.asarray(q, float)
        relaxed = np.zeros(len(q), bool) if relaxed is None else np.asarray(relaxed, bool)
        self.calls += len(q)
        lo = np.where(relaxed[:, None], self.lo_raw, self.lo)
        hi = np.where(relaxed[:, None], self.hi_raw, self.hi)
        ok = np.all((q >= lo) & (q <= hi), axis=1)
        out = np.zeros(len(q), bool)
        idx = np.nonzero(ok)[0]
        for i in range(0, len(idx), 32768):
            j = idx[i:i + 32768]
            col = np.empty(len(j), bool)
            r = relaxed[j]
            if (~r).any():
                col[~r] = self.checker.valid_configs(q[j][~r][:, None, :], strict_margin=True)[:, 0]
            if r.any():
                col[r] = self.checker.valid_configs(q[j][r][:, None, :], strict_margin=False)[:, 0]
            grav = self.gravity_util(q[j]) <= rho
            out[j] = col & grav
        return out

    def near_endpoint(self, q: np.ndarray) -> np.ndarray:
        d = np.minimum(np.linalg.norm(q - self.anchors[0], axis=1), np.linalg.norm(q - self.anchors[1], axis=1))
        return d <= self.cfg.endpoint_radius

    def segment_ok(self, a: np.ndarray, b: np.ndarray, rho: float) -> bool:
        n = max(2, int(np.ceil(np.max(np.abs(b - a)) / self.step)) + 1)
        u = np.linspace(0, 1, n)[:, None]
        pts = a[None] + u * (b - a)[None]
        return bool(self.admissible(pts, rho, self.near_endpoint(pts)).all())


def lazy_prm(adm: Admissibility, start: np.ndarray, goal: np.ndarray, rho: float, n_samples: int,
             rng: np.random.Generator) -> dict:
    """Lazy PRM in A_rho. Returns waypoints (start..goal) or None, with statistics."""
    import torch
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    cfg = adm.cfg
    t0 = time.perf_counter()
    adm.anchors = (start, goal)
    if not adm.admissible(np.stack([start, goal]), rho, np.ones(2, bool)).all():
        return {"path": None, "reason": "endpoint_not_admissible"}
    if adm.segment_ok(start, goal, rho):
        return {"path": np.stack([start, goal]), "nodes": 2, "lazy_iters": 0, "wall_s": time.perf_counter() - t0}
    x = rng.uniform(adm.lo, adm.hi, size=(n_samples, 7))
    x = x[adm.admissible(x, rho)]
    nodes = np.concatenate([start[None], goal[None], x])
    n = len(nodes)
    # k-NN (GPU, chunked)
    X = torch.as_tensor(nodes, device="cuda", dtype=torch.float32)
    rows, cols, dist = [], [], []
    for i in range(0, n, 4096):
        d = torch.cdist(X[i:i + 4096], X)
        d[torch.arange(d.shape[0]), torch.arange(i, i + d.shape[0])] = float("inf")
        vals, idx = torch.topk(d, min(cfg.k, n - 1), largest=False)
        rows.append(np.repeat(np.arange(i, i + d.shape[0]), vals.shape[1]))
        cols.append(idx.cpu().numpy().ravel())
        dist.append(vals.cpu().numpy().ravel())
    r, c, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(dist)
    keep = w <= cfg.max_edge
    r, c, w = r[keep], c[keep], w[keep]
    a, b = np.minimum(r, c), np.maximum(r, c)
    key = np.unique(np.stack([a, b], 1), axis=0)
    w = np.linalg.norm(nodes[key[:, 0]] - nodes[key[:, 1]], axis=1)
    status = {}  # (i,j) -> bool, lazily evaluated
    alive = np.ones(len(key), bool)
    index = {(int(i), int(j)): e for e, (i, j) in enumerate(key)}
    for it in range(cfg.max_lazy_iters):
        g = coo_matrix((w[alive], (key[alive, 0], key[alive, 1])), shape=(n, n)).tocsr()
        dist0, pred = dijkstra(g, directed=False, indices=0, return_predecessors=True)
        if not np.isfinite(dist0[1]):
            return {"path": None, "reason": "disconnected", "nodes": n, "lazy_iters": it,
                    "wall_s": time.perf_counter() - t0}
        path = [1]
        while path[-1] != 0:
            path.append(int(pred[path[-1]]))
        path = path[::-1]
        bad = False
        for u, v in zip(path[:-1], path[1:]):
            e = index[(min(u, v), max(u, v))]
            if e not in status:
                status[e] = adm.segment_ok(nodes[u], nodes[v], rho)
            if not status[e]:
                alive[e] = False
                bad = True
        if not bad:
            return {"path": nodes[path], "nodes": n, "lazy_iters": it + 1, "edges_checked": len(status),
                    "wall_s": time.perf_counter() - t0}
    return {"path": None, "reason": "lazy_iteration_budget", "nodes": n, "wall_s": time.perf_counter() - t0}


def shortcut(adm: Admissibility, path: np.ndarray, rho: float, tries: int, rng) -> np.ndarray:
    p = [np.asarray(x) for x in path]
    for _ in range(tries):
        if len(p) <= 2:
            break
        i, j = sorted(rng.choice(len(p), 2, replace=False))
        if j - i < 2:
            continue
        if adm.segment_ok(p[i], p[j], rho):
            p = p[: i + 1] + p[j:]
    return np.stack(p)


# ------------------------------------------------------------------ time scaling
def quintic(u):
    return (u**3 * (10 - 15 * u + 6 * u * u), 30 * u * u * (1 - u) ** 2,
            60 * u - 180 * u * u + 120 * u**3, 60 - 360 * u + 360 * u * u)


S1, S2, S3 = 1.875, 5.7735026919, 60.0  # max |s'|, |s''|, |s'''| of the quintic


def segment_duration(adm: Admissibility, a: np.ndarray, b: np.ndarray, safety: float, n: int = 257,
                     rho: float = 0.0) -> float:
    """Smallest (doubling then bisection) T with V/A/J <= safety * Scaled95 and
    torque <= max(safety, (1 + rho) / 2) * Scaled95 on a dense grid. The torque target
    must lie between rho (gravity bound of the path) and 1, otherwise no T exists."""
    d = np.abs(b - a)
    sc = LIMIT_SCALE * safety
    sc_tau = LIMIT_SCALE * max(safety, 0.5 * (1.0 + rho))
    T = max(float(np.max(d * S1 / (sc * adm.vmax))), float(np.sqrt(np.max(d * S2 / (sc * adm.amax)))),
            float(np.cbrt(np.max(d * S3 / (sc * adm.jmax)))), 1e-3)

    def ok(T):
        u = np.linspace(0, 1, n)
        s, s1, s2, _ = quintic(u)
        q = a[None] + s[:, None] * (b - a)[None]
        qd = s1[:, None] * (b - a)[None] / T
        qdd = s2[:, None] * (b - a)[None] / T**2
        tau = adm.torque(q, qd, qdd)
        return bool(np.all(np.abs(tau) <= sc_tau * adm.tau_max))

    if ok(T):
        return T
    lo, hi = T, 2 * T
    while not ok(hi):
        lo, hi = hi, 2 * hi
        if hi > 1e4:
            raise RuntimeError("no finite time scaling found (gravity not admissible?)")
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


def witness_trajectory(waypoints: np.ndarray, durations: list[float], samples_per_segment: int = 1025) -> dict:
    ts, qs, vs, acs, js = [], [], [], [], []
    t0 = 0.0
    for k, (a, b, T) in enumerate(zip(waypoints[:-1], waypoints[1:], durations)):
        u = np.linspace(0, 1, samples_per_segment)
        if k > 0:
            u = u[1:]
        s, s1, s2, s3 = quintic(u)
        d = (b - a)[None]
        ts.append(t0 + u * T)
        qs.append(a[None] + s[:, None] * d)
        vs.append(s1[:, None] * d / T)
        acs.append(s2[:, None] * d / T**2)
        js.append(s3[:, None] * d / T**3)
        t0 += T
    return {"time": np.concatenate(ts), "q": np.concatenate(qs), "qd": np.concatenate(vs),
            "qdd": np.concatenate(acs), "jerk": np.concatenate(js)}


def certify_cell(scene: str, payload: Payload, start, goal, cfg: OracleConfig, adm_cache: dict) -> dict:
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    tries = []
    t0 = time.perf_counter()
    endpoint_util = None
    for margin in cfg.margins:
        key = (scene, payload.label, margin)
        if key not in adm_cache:
            for k in [k for k in adm_cache if k[:2] != key[:2]]:
                del adm_cache[k]  # keep only the current scene/payload
            adm_cache[key] = Admissibility(scene, payload, cfg, margin)
        adm = adm_cache[key]
        endpoint_util = adm.gravity_util(np.stack([start, goal])).tolist()
        rng = np.random.default_rng(cfg.seed)
        for rho in cfg.rhos:
            for n in cfg.samples:
                r = lazy_prm(adm, start, goal, rho, n, rng)
                tries.append({"margin_m": margin, "rho": rho, "samples": n,
                              **{k: v for k, v in r.items() if k != "path"}})
                if r.get("reason") == "endpoint_not_admissible":
                    break  # more samples cannot help at this rho
                if r["path"] is not None:
                    wp = shortcut(adm, r["path"], rho, cfg.shortcut_tries, rng)
                    try:
                        durs = [segment_duration(adm, a, b, cfg.time_safety, rho=rho)
                                for a, b in zip(wp[:-1], wp[1:])]
                    except RuntimeError as error:
                        tries[-1]["timing_error"] = str(error)
                        continue
                    return {"status": "WITNESS_FOUND", "margin_m": margin, "rho": rho, "samples": n,
                            "waypoints": wp, "durations": durs, "tries": tries,
                            "wall_s": time.perf_counter() - t0, "endpoint_gravity_util": endpoint_util}
    return {"status": "UNCERTIFIED", "tries": tries, "wall_s": time.perf_counter() - t0,
            "endpoint_gravity_util": endpoint_util}


def main() -> None:
    from .run_campaign import export, physical_limits

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cells", type=Path, default=None, help="optional JSON list of {task_id, payload}")
    ap.add_argument("--config", type=str, default="{}")
    a = ap.parse_args()
    cfg = OracleConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(a.config).items()})
    bank = read_json(a.bank / "tasks.json")
    tasks = {t["task_id"]: t for t in bank["tasks"]}
    payloads = {p["label"]: Payload.from_json(p) for p in bank["payloads"]}
    if a.cells:
        todo = [(c["task_id"], c["payload"]) for c in read_json(a.cells)]
    else:
        todo = [(c["task_id"], c["payload"]["label"]) for c in read_json(a.bank / "cells.json") if c["admissible"]]
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg

    lim = physical_limits(KinematicsCfg.from_robot_yaml_file("franka.yml").kinematics_config)
    write_json(a.output / "oracle_config.json", asdict(cfg))
    todo.sort(key=lambda x: (tasks[x[0]]["scene"], x[1], x[0]))  # group by scene/payload
    cache: dict = {}
    for tid, pl in todo:
        d = a.output / "cells" / tid / pl
        if (d / "oracle.json").exists():
            continue
        t = tasks[tid]
        res = certify_cell(t["scene"], payloads[pl], t["start"], t["goal"], cfg, cache)
        rec = {k: v for k, v in res.items() if k not in ("waypoints",)}
        rec.update(task_id=tid, payload=pl, scene=t["scene"], stratum=t["stratum"], split=t["split"])
        if res["status"] == "WITNESS_FOUND":
            wp = res["waypoints"]
            rec["waypoints"] = wp.tolist()
            rec["witness_duration_s"] = float(np.sum(res["durations"]))
            tr = witness_trajectory(wp, res["durations"])
            export(d / "witness", tr, {"payload": payloads[pl].to_json(), "scene": t["scene"],
                                       "start": t["start"], "goal": t["goal"], "limits": lim,
                                       "kind": "existence_witness", "gap_check": True,
                                       "collision_samples": len(tr["q"])})
        write_json(d / "oracle.json", rec)
        print(tid, pl, rec["status"], rec.get("rho"), rec.get("samples"), f"{rec['wall_s']:.1f}s",
              len(rec.get("waypoints", [])), flush=True)


if __name__ == "__main__":
    main()
