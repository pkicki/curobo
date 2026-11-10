# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hard benchmark subsets with certified solvability (CUDA env + Pinocchio certification).

Gravity bottleneck rho*: the smallest achievable PEAK static gravity utilisation
(max_j |g_j(q)| / (0.95 tau_max_j)) over collision-free routes from start to goal,
estimated on a roadmap (an upper bound on the true value: sampling can only miss
better routes). Computed as a minimax path: the minimum spanning tree has the
minimax property; edge weights are the max utilisation along the edge, evaluated
lazily (unchecked edges carry the lower bound max(w_u, w_v)).

Subsets:
  TIGHT_STATIC   0.9 <= rho* < 1 at the primary budget, straight line NOT admissible,
                 and the witness is complex (>= 3 segments or length >= 1.3 x straight).
                 Witness: the existence-oracle construction at rho = rho* (quasi-static,
                 rest at waypoints) -> certified like the main benchmark.
  DYNAMIC        rho* >= 1 at BOTH the primary and the confirmation budget (no
                 quasi-static route found), and a DYNAMIC witness: the minimax route is
                 fitted by a rest-to-rest quintic spline and retimed by K107 with the
                 overload-aware Phase-I start, so the payload swings through the overload.
                 Certified by ``independent_validate`` (gap check). K107 is a dependency
                 shared with methods under test; this is recorded in every record.
Everything else is recorded with its reason (EASY, NO_DYNAMIC_WITNESS, ...).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .common import canonical_hash, payload_grid, rest_controls, task_split, write_json
from .existence_oracle import Admissibility, OracleConfig, segment_duration, shortcut, witness_trajectory
from .geometry import evaluate, fit_quintic

HARD_PAYLOADS = ("m3_offset", "m4_axial", "m4_offset", "m5_axial", "m5_offset")


@dataclass
class HardConfig:
    select_samples: int = 400_000  # selection roadmap (fully checked)
    samples: int = 400_000  # primary verification roadmap budget (fresh samples)
    confirm_samples: int = 1_000_000  # second, independent budget for "no static route"
    dynamic_min: float = 1.02  # rho* must exceed this at BOTH budgets for DYNAMIC
    k: int = 12
    max_edge: float = 1.2
    max_lazy_iters: int = 300
    tight_min: float = 0.90
    min_segments: int = 3
    min_length_ratio: float = 1.3
    endpoint_max: float = 0.9  # endpoint static util (vs 0.95 tau_max)
    straight_min_peak: float = 1.0  # straight line must violate the static limit
    spline_controls: tuple = (30, 60, 100, 16)
    seed: int = 0


def edge_profile(adm: Admissibility, a: np.ndarray, b: np.ndarray) -> float:
    """Max gravity utilisation along a straight edge, or inf if it is not collision free."""
    n = max(2, int(np.ceil(np.max(np.abs(b - a)) / adm.step)) + 1)
    pts = a[None] + np.linspace(0, 1, n)[:, None] * (b - a)[None]
    if not adm.admissible(pts, np.inf, adm.near_endpoint(pts)).all():
        return float("inf")
    return float(adm.gravity_util(pts).max())


def bottleneck_route(adm: Admissibility, start, goal, n_samples: int, cfg: HardConfig, rng) -> dict:
    """Minimax-gravity route on a lazy roadmap of collision-free configurations."""
    import torch
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree

    t0 = time.perf_counter()
    adm.anchors = (start, goal)
    x = rng.uniform(adm.lo, adm.hi, size=(n_samples, 7))
    x = x[adm.admissible(x, np.inf)]
    nodes = np.concatenate([start[None], goal[None], x])
    util = adm.gravity_util(nodes)
    n = len(nodes)
    X = torch.as_tensor(nodes, device="cuda", dtype=torch.float32)
    rows, cols = [], []
    for i in range(0, n, 4096):
        d = torch.cdist(X[i:i + 4096], X)
        d[torch.arange(d.shape[0]), torch.arange(i, i + d.shape[0])] = float("inf")
        vals, idx = torch.topk(d, min(cfg.k, n - 1), largest=False)
        keep = (vals <= cfg.max_edge).cpu().numpy()
        r = np.repeat(np.arange(i, i + d.shape[0]), vals.shape[1]).reshape(keep.shape)
        rows.append(r[keep])
        cols.append(idx.cpu().numpy()[keep])
    r, c = np.concatenate(rows), np.concatenate(cols)
    key = np.unique(np.stack([np.minimum(r, c), np.maximum(r, c)], 1), axis=0)
    w = np.maximum(util[key[:, 0]], util[key[:, 1]])  # lower bound until checked
    checked = np.zeros(len(key), bool)
    index = {(int(a), int(b)): e for e, (a, b) in enumerate(key)}
    for it in range(cfg.max_lazy_iters):
        alive = np.isfinite(w)
        # +1 keeps weights strictly positive for scipy (zero = no edge)
        g = coo_matrix((w[alive] + 1.0, (key[alive, 0], key[alive, 1])), shape=(n, n)).tocsr()
        tree = minimum_spanning_tree(g)
        tree = tree + tree.T
        order, pred = breadth_first_order(tree, 0, directed=False, return_predecessors=True)
        if pred[1] < 0:
            return {"path": None, "reason": "disconnected", "nodes": n, "iters": it,
                    "wall_s": time.perf_counter() - t0}
        path = [1]
        while path[-1] != 0:
            path.append(int(pred[path[-1]]))
        path = path[::-1]
        new = False
        for u, v in zip(path[:-1], path[1:]):
            e = index[(min(u, v), max(u, v))]
            if not checked[e]:
                checked[e] = True
                w[e] = edge_profile(adm, nodes[u], nodes[v])
                new = True
        if not new:
            peak = float(max(w[index[(min(u, v), max(u, v))]] for u, v in zip(path[:-1], path[1:])))
            return {"path": nodes[path], "rho_star": peak, "nodes": n, "iters": it + 1,
                    "edges_checked": int(checked.sum()), "wall_s": time.perf_counter() - t0}
    return {"path": None, "reason": "lazy_iteration_budget", "nodes": n, "wall_s": time.perf_counter() - t0}


def global_roadmap(adm: Admissibility, n_samples: int, cfg: HardConfig, rng, batch_points: int = 600_000) -> dict:
    """Fully checked roadmap over collision-free configurations (no gravity limit).

    Edge weight = max static gravity utilisation along the edge (inf if colliding).
    Returns nodes, node utilisation, the minimum spanning forest (minimax property)."""
    import torch
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    t0 = time.perf_counter()
    adm.anchors = (np.full(7, np.inf), np.full(7, np.inf))
    x = rng.uniform(adm.lo, adm.hi, size=(n_samples, 7))
    nodes = x[adm.admissible(x, np.inf)]
    util = adm.gravity_util(nodes)
    n = len(nodes)
    X = torch.as_tensor(nodes, device="cuda", dtype=torch.float32)
    rows, cols = [], []
    for i in range(0, n, 4096):
        d = torch.cdist(X[i:i + 4096], X)
        d[torch.arange(d.shape[0]), torch.arange(i, i + d.shape[0])] = float("inf")
        vals, idx = torch.topk(d, min(cfg.k, n - 1), largest=False)
        keep = (vals <= cfg.max_edge).cpu().numpy()
        r = np.repeat(np.arange(i, i + d.shape[0]), vals.shape[1]).reshape(keep.shape)
        rows.append(r[keep])
        cols.append(idx.cpu().numpy()[keep])
    r, c = np.concatenate(rows), np.concatenate(cols)
    key = np.unique(np.stack([np.minimum(r, c), np.maximum(r, c)], 1), axis=0)
    length = np.linalg.norm(nodes[key[:, 0]] - nodes[key[:, 1]], axis=1)
    m = np.maximum(2, np.ceil(np.max(np.abs(nodes[key[:, 1]] - nodes[key[:, 0]]), axis=1) / adm.step).astype(int) + 1)
    w = np.empty(len(key))
    order = np.argsort(m)
    i = 0
    while i < len(order):  # batches of edges with similar resolution
        j = i
        tot = 0
        while j < len(order) and tot + m[order[j]] <= batch_points:
            tot += m[order[j]]
            j += 1
        j = max(j, i + 1)
        e = order[i:j]
        mm = int(m[e].max())
        u = np.linspace(0, 1, mm)
        a, b = nodes[key[e, 0]], nodes[key[e, 1]]
        pts = (a[:, None] + u[None, :, None] * (b - a)[:, None]).reshape(-1, 7)
        ok = adm.admissible(pts, np.inf).reshape(len(e), mm).all(1)
        gu = adm.gravity_util(pts).reshape(len(e), mm).max(1)
        w[e] = np.where(ok, gu, np.inf)
        i = j
    alive = np.isfinite(w)
    g = coo_matrix((w[alive] + 1.0, (key[alive, 0], key[alive, 1])), shape=(n, n)).tocsr()
    tree = minimum_spanning_tree(g)
    tree = (tree + tree.T).tocsr()
    return {"nodes": nodes, "util": util, "tree": tree, "edges": int(alive.sum()), "length": length,
            "key": key, "w": w,
            "wall_s": time.perf_counter() - t0}


def select_pairs(rm: dict, cfg: HardConfig, rng, *, n_tight: int, n_dynamic: int, max_overload: float = 1.35,
                 min_distance: float = 1.0, tries: int = 4000) -> list[dict]:
    """Pick endpoint pairs from roadmap nodes by their minimax (MST path max) value."""
    from scipy.sparse.csgraph import breadth_first_order

    nodes, util, tree = rm["nodes"], rm["util"], rm["tree"]
    ok_end = np.nonzero(util <= cfg.endpoint_max)[0]
    out, need = [], {"TIGHT": n_tight, "DYNAMIC": n_dynamic}
    for _ in range(tries):
        if all(v <= 0 for v in need.values()) or len(ok_end) < 2:
            break
        s = int(rng.choice(ok_end))
        order, pred = breadth_first_order(tree, s, directed=False, return_predecessors=True)
        # minimax value from s to every reachable node along the tree
        mm = np.full(len(nodes), np.inf)
        mm[s] = util[s]
        tree_coo = tree.tocsr()
        for v in order[1:]:
            u = pred[v]
            mm[v] = max(mm[u], tree_coo[u, v] - 1.0, util[v])
        cand = [t for t in ok_end if t != s and np.isfinite(mm[t]) and np.linalg.norm(nodes[t] - nodes[s]) >= min_distance]
        if not cand:
            continue
        cand = np.asarray(cand)
        tight = cand[(mm[cand] >= 0.95) & (mm[cand] < 1.0)]
        dyn = cand[(mm[cand] >= 1.0) & (mm[cand] <= max_overload)]
        for kind, pool in (("TIGHT", tight), ("DYNAMIC", dyn)):
            if need[kind] > 0 and len(pool):
                t = int(rng.choice(pool))
                out.append({"kind": kind, "start": nodes[s], "goal": nodes[t], "selection_minimax": float(mm[t])})
                need[kind] -= 1
    return out


def polyline_length(p: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def static_witness(adm: Admissibility, route: np.ndarray, rho: float, cfg_o: OracleConfig, rng) -> dict:
    wp = shortcut(adm, route, rho, cfg_o.shortcut_tries, rng)
    for a, b in zip(wp[:-1], wp[1:]):
        if not adm.segment_ok(a, b, rho):
            return {"ok": False, "reason": "shortcut_invalid"}
    durs = [segment_duration(adm, a, b, cfg_o.time_safety, rho=rho) for a, b in zip(wp[:-1], wp[1:])]
    return {"ok": True, "waypoints": wp, "durations": durs}


def dynamic_witness(adm: Admissibility, oracle, checker, route: np.ndarray, cfg: HardConfig,
                    rho_route: float, rng) -> dict:
    """Shortcut the minimax route (keeping its bottleneck), fit a rest-to-rest quintic
    spline, and retime it with K107 (overload-aware Phase-I start)."""
    route = shortcut(adm, route, rho_route + 1e-3, 200, rng)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    s_way = arc / arc[-1]
    dense_s = np.linspace(0, 1, 1025)
    dense = np.stack([np.interp(dense_s, s_way, route[:, j]) for j in range(7)], 1)
    attempts = []
    for n_ctrl in cfg.spline_controls:
        c = fit_quintic(dense_s, dense, n=n_ctrl, smooth=1e-8)
        q = evaluate(c, np.linspace(0, 1, 2049))
        relaxed = adm.near_endpoint(q)
        if not adm.admissible(q, np.inf, relaxed).all():
            attempts.append({"controls": n_ctrl, "reason": "spline_collision"})
            continue
        peak = float(adm.gravity_util(q).max())
        res = oracle.retime(c, diagnose_infeasible=False)
        attempts.append({"controls": n_ctrl, "status": res.status, "peak_static_util": peak,
                         "duration_s": res.duration_s, "route": res.timings_ms.get("start_route")})
        if res.feasible:
            return {"ok": True, "controls": c, "result": res, "peak_static_util": peak, "attempts": attempts}
    return {"ok": False, "attempts": attempts}


def straight_peak(adm: Admissibility, a, b) -> float:
    q = evaluate(rest_controls(a, b), np.linspace(0, 1, 129))
    return float(adm.gravity_util(q).max())


def main() -> None:
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg

    from .collision_screen import PathCollisionChecker
    from .k107_oracle import K107Oracle
    from .run_campaign import export, physical_limits

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--scenes", nargs="*", default=["free", "table", "cluttered"])
    ap.add_argument("--payloads", nargs="*", default=list(HARD_PAYLOADS))
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--max-candidates", type=int, default=30, help="per kind, per scene x payload")
    ap.add_argument("--config", default="{}", help="JSON overrides of HardConfig")
    ap.add_argument("--kinds", nargs="*", default=["TIGHT", "DYNAMIC"])
    a = ap.parse_args()
    cfg = HardConfig(seed=a.seed, **json.loads(a.config))
    cfg_o = OracleConfig(seed=a.seed)
    lim = physical_limits(KinematicsCfg.from_robot_yaml_file("franka.yml").kinematics_config)
    write_json(a.output / "hard_config.json", {"hard": asdict(cfg), "oracle": asdict(cfg_o), "args": vars(a)})
    payloads = {p.label: p for p in payload_grid()}
    rng = np.random.default_rng(a.seed)
    for scene in a.scenes:
        pool_checker = PathCollisionChecker(scene)
        for pl in a.payloads:
            p = payloads[pl]
            adm = Admissibility(scene, p, cfg_o, cfg_o.margins[0])
            oracle = K107Oracle(p, overload_aware_start=True, start_method="lp")
            # 1) selection on a fully checked roadmap (minimax along its spanning tree)
            rm = global_roadmap(adm, cfg.select_samples, cfg, np.random.default_rng([a.seed, 7]))
            pairs = select_pairs(rm, cfg, rng, n_tight=a.max_candidates,
                                 n_dynamic=a.max_candidates if "DYNAMIC" in a.kinds else 0)
            cand = []
            for pr in pairs:
                q0, q1 = pr["start"], pr["goal"]
                if not pool_checker_ok(adm, q0, q1):
                    continue
                eu = adm.gravity_util(np.stack([q0, q1]))
                sp = straight_peak(adm, q0, q1)
                if pr["kind"] == "TIGHT" and sp < cfg.straight_min_peak:
                    continue  # straight line must not be statically admissible
                cand.append((q0, q1, eu, sp, pr))
            print(f"{scene} {pl}: roadmap {len(rm['nodes'])} nodes {rm['edges']} edges {rm['wall_s']:.0f}s; "
                  f"selected {sum(c[4]['kind'] == 'TIGHT' for c in cand)} tight / "
                  f"{sum(c[4]['kind'] == 'DYNAMIC' for c in cand)} dynamic", flush=True)
            for q0, q1, eu, sp, pr in cand:
                core = {"scene": scene, "payload": pl, "start": q0.tolist(), "goal": q1.tolist()}
                cid = f"{scene}_{pl}_{canonical_hash(core)[:10]}"
                d = a.output / "cells" / cid
                if (d / "hard.json").exists():
                    continue
                rec = core | {"cell_id": cid, "split": task_split(cid), "endpoint_util": eu.tolist(),
                              "selected_as": pr["kind"], "selection_minimax": pr["selection_minimax"],
                              "straight_peak_util": sp, "straight_length": float(np.linalg.norm(q1 - q0))}
                r1 = bottleneck_route(adm, q0, q1, cfg.samples, cfg, np.random.default_rng([a.seed, 1]))
                rec["primary"] = {k: v for k, v in r1.items() if k != "path"}
                label, reason = None, None
                if r1["path"] is None:
                    label, reason = "UNRESOLVED", r1["reason"]
                elif r1["rho_star"] < 1.0:
                    rho = min(max(r1["rho_star"], float(eu.max())) + 1e-3, 0.999)
                    if r1["rho_star"] >= cfg.tight_min:
                        # robust tightness: re-estimate at the larger budget, keep the lower bound
                        r2 = bottleneck_route(adm, q0, q1, cfg.confirm_samples, cfg, np.random.default_rng([a.seed, 2]))
                        rec["confirm"] = {k: v for k, v in r2.items() if k != "path"}
                        if r2["path"] is not None and r2["rho_star"] < r1["rho_star"]:
                            r1 = r2
                            rho = min(max(r2["rho_star"], float(eu.max())) + 1e-3, 0.999)
                    rec["rho_star_hat"] = r1["rho_star"]
                    if r1["rho_star"] < cfg.tight_min:
                        label, reason = "EASY", "rho_star_below_tight_min"
                    else:
                        sw = static_witness(adm, r1["path"], rho, cfg_o, np.random.default_rng(a.seed))
                        if not sw["ok"]:
                            label, reason = "UNRESOLVED", sw["reason"]
                        else:
                            segs = len(sw["waypoints"]) - 1
                            ratio = polyline_length(sw["waypoints"]) / rec["straight_length"]
                            rec.update(witness_kind="static", witness_rho=rho, segments=segs, length_ratio=ratio,
                                       waypoints=sw["waypoints"].tolist(),
                                       witness_duration_s=float(np.sum(sw["durations"])))
                            if segs >= cfg.min_segments or ratio >= cfg.min_length_ratio:
                                label = "TIGHT_STATIC"
                            else:
                                label, reason = "EASY", "simple_witness"
                            tr = witness_trajectory(sw["waypoints"], sw["durations"])
                            export(d / "witness", tr, {"payload": p.to_json(), "scene": scene, "start": rec["start"],
                                                        "goal": rec["goal"], "limits": lim, "gap_check": True,
                                                        "kind": "static_witness"})
                else:
                    r2 = bottleneck_route(adm, q0, q1, cfg.confirm_samples, cfg, np.random.default_rng([a.seed, 2]))
                    rec["confirm"] = {k: v for k, v in r2.items() if k != "path"}
                    if r2["path"] is not None and r2["rho_star"] < 1.0:
                        label, reason = "UNRESOLVED", "static_route_found_at_confirm_budget"
                        rec["confirm_static_route"] = True
                    elif min(r1["rho_star"], r2["rho_star"] if r2["path"] is not None else np.inf) < cfg.dynamic_min:
                        label, reason = "UNRESOLVED", "static_bottleneck_within_dynamic_margin"
                    else:
                        use2 = r2["path"] is not None and r2["rho_star"] < r1["rho_star"]
                        route, rr = (r2["path"], r2["rho_star"]) if use2 else (r1["path"], r1["rho_star"])
                        dw = dynamic_witness(adm, oracle, pool_checker, route, cfg, rr,
                                             np.random.default_rng(a.seed))
                        rec["dynamic_attempts"] = dw["attempts"]
                        if dw["ok"]:
                            label = "DYNAMIC"
                            res = dw["result"]
                            rec.update(witness_kind="dynamic_k107_phase1", peak_static_util=dw["peak_static_util"],
                                       witness_duration_s=res.duration_s,
                                       witness_dependency="K107 TOPP-RA+Newton with Phase-I start (also used by tested methods)")
                            d.mkdir(parents=True, exist_ok=True)
                            np.save(d / "witness_controls.npy", dw["controls"])
                            tr = oracle.sample_trajectory(dw["controls"], res, n=8193)
                            export(d / "witness", tr, {"payload": p.to_json(), "scene": scene, "start": rec["start"],
                                                        "goal": rec["goal"], "limits": lim, "gap_check": True,
                                                        "kind": "dynamic_witness"})
                        else:
                            label, reason = "NO_DYNAMIC_WITNESS", "k107_or_spline_failed"
                rec["label"], rec["reason"] = label, reason
                write_json(d / "hard.json", rec)
                print(cid, label, reason, round(r1.get("rho_star", float("nan")), 3) if r1["path"] is not None else None,
                      f"{r1['wall_s']:.1f}s", flush=True)


def pool_checker_ok(adm: Admissibility, q0, q1) -> bool:
    """Endpoints collision free with THIS payload under the problem's own constraints."""
    adm.anchors = (q0, q1)
    return bool(adm.admissible(np.stack([q0, q1]), np.inf, np.ones(2, bool)).all())


if __name__ == "__main__":
    main()
