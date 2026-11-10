# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DETOUR and SWING benchmark subsets (CUDA env + Pinocchio certification).

Path lengths are joint-space L2 lengths of polygonal routes on roadmaps:
  L_free    shortest collision-free route (no gravity limit)
  L_static  shortest route through static gravity utilisation <= RHO_STATIC (< 1)
  L_swing   shortest route with peak static utilisation <= RHO_SWING (> 1)
All are roadmap estimates (upper bounds on the true shortest lengths); routes are
shortcut under the same constraint before measuring.

  DETOUR  L_static >= DETOUR_RATIO * L_free and the shortest collision-free route is
          statically infeasible (peak >= SHORT_OVERLOAD). Certified by a quasi-static
          witness on the static route (existence_oracle construction).
  SWING   DETOUR and L_swing <= L_static / DETOUR_RATIO, with peak > 1 on the swing
          route, and a DYNAMIC witness on it: rest-to-rest quintic spline retimed by K107
          (overload-aware Phase-I start), certified independently. The static route is
          also retimed by K107 (same spline construction) for a fair duration comparison.
Selection runs on a fully checked roadmap (Dijkstra under gravity thresholds); every
selected pair is re-verified on FRESH lazy roadmaps before it is labelled.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .common import canonical_hash, payload_grid, task_split, write_json
from .existence_oracle import Admissibility, OracleConfig, lazy_prm, segment_duration, shortcut, witness_trajectory
from .hard_bank import HARD_PAYLOADS, HardConfig, dynamic_witness, global_roadmap, polyline_length


@dataclass
class DetourConfig:
    rho_static: float = 0.97
    rho_swing: float = 1.30
    swing_ladder: tuple = (1.02, 1.05, 1.10, 1.20, 1.30)
    swing_ratio: float = 1.5  # SWING: L_static >= swing_ratio * L_swing
    short_overload: float = 1.05
    detour_ratio: float = 2.0
    endpoint_max: float = 0.9
    min_free_length: float = 0.8  # rad
    select_samples: int = 400_000
    verify_samples: int = 400_000
    pairs_per_combo: int = 25
    starts_per_combo: int = 400
    seed: int = 20261113


def select_detour_pairs(rm: dict, cfg: DetourConfig, rng) -> list[dict]:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    nodes, util = rm["nodes"], rm["util"]
    key, w, length = rm["key"], rm["w"], rm["length"]
    n = len(nodes)

    def graph(mask):
        return coo_matrix((length[mask], (key[mask, 0], key[mask, 1])), shape=(n, n)).tocsr()

    alive = np.isfinite(w)
    g_free, g_static, g_swing = graph(alive), graph(alive & (w <= cfg.rho_static)), graph(alive & (w <= cfg.rho_swing))
    ok_end = np.nonzero(util <= cfg.endpoint_max)[0]
    out = []
    for s in rng.choice(ok_end, size=min(cfg.starts_per_combo, len(ok_end)), replace=False):
        d = dijkstra(g_free, directed=False, indices=int(s))
        ds = dijkstra(g_static, directed=False, indices=int(s))
        dw = dijkstra(g_swing, directed=False, indices=int(s))
        t = ok_end[(d[ok_end] >= cfg.min_free_length) & np.isfinite(ds[ok_end])]
        if not len(t):
            continue
        ratio = ds[t] / d[t]
        good = t[ratio >= cfg.detour_ratio * 1.1]  # selection margin above the label threshold
        if not len(good):
            continue
        # prefer pairs with a short swing route too (SWING candidates)
        swing_ok = good[dw[good] <= ds[good] / (cfg.detour_ratio * 1.1)]
        pick = swing_ok if len(swing_ok) else good
        g = int(rng.choice(pick))
        out.append({"start": nodes[s], "goal": nodes[g], "sel_L_free": float(d[g]), "sel_L_static": float(ds[g]),
                    "sel_L_swing": float(dw[g])})
        if len(out) >= cfg.pairs_per_combo:
            break
    return out


def route(adm, q0, q1, rho, n, seed):
    r = lazy_prm(adm, q0, q1, rho, n, np.random.default_rng(seed))
    if r["path"] is None:
        return None, r
    p = shortcut(adm, r["path"], rho, 300, np.random.default_rng(seed))
    return p, r


def peak_util(adm, path: np.ndarray, step: float = 0.01) -> float:
    pts = []
    for a, b in zip(path[:-1], path[1:]):
        m = max(2, int(np.ceil(np.max(np.abs(b - a)) / step)) + 1)
        pts.append(a[None] + np.linspace(0, 1, m)[:, None] * (b - a)[None])
    return float(adm.gravity_util(np.concatenate(pts)).max())


def main() -> None:
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg

    from .k107_oracle import K107Oracle
    from .run_campaign import export, physical_limits

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--scenes", nargs="*", default=["free", "table", "cluttered"])
    ap.add_argument("--payloads", nargs="*", default=list(HARD_PAYLOADS))
    ap.add_argument("--config", default="{}")
    a = ap.parse_args()
    cfg = DetourConfig(**json.loads(a.config))
    cfg_o, cfg_h = OracleConfig(seed=cfg.seed), HardConfig(seed=cfg.seed)
    lim = physical_limits(KinematicsCfg.from_robot_yaml_file("franka.yml").kinematics_config)
    write_json(a.output / "detour_config.json", {"detour": asdict(cfg), "oracle": asdict(cfg_o), "args": vars(a)})
    payloads = {p.label: p for p in payload_grid()}
    rng = np.random.default_rng(cfg.seed)
    for scene in a.scenes:
        for pl in a.payloads:
            p = payloads[pl]
            adm = Admissibility(scene, p, cfg_o, cfg_o.margins[0])
            oracles = [K107Oracle(p, overload_aware_start=True, start_method="lp"),
                       K107Oracle(p, overload_aware_start=True, start_method="lp", warmup=False,
                                  lp_options={"lp_stride": 1, "min_slack": 0.001, "rounds": 2})]
            oracle = oracles[0]

            def dyn(path, rho_path):
                for o in oracles:  # default Phase-I LP first, then the full-resolution LP
                    w = dynamic_witness(adm, o, None, path, cfg_h, rho_path, np.random.default_rng(cfg.seed))
                    if w["ok"]:
                        return w, o
                return w, None
            rm = global_roadmap(adm, cfg.select_samples, cfg_h, np.random.default_rng([cfg.seed, 7]))
            pairs = select_detour_pairs(rm, cfg, rng)
            print(f"{scene} {pl}: roadmap {len(rm['nodes'])} nodes, {len(pairs)} selected", flush=True)
            for pr in pairs:
                q0, q1 = pr["start"], pr["goal"]
                core = {"scene": scene, "payload": pl, "start": q0.tolist(), "goal": q1.tolist()}
                cid = f"{scene}_{pl}_{canonical_hash(core)[:10]}"
                d = a.output / "cells" / cid
                if (d / "detour.json").exists():
                    continue
                rec = core | {"cell_id": cid, "split": task_split(cid), "selection": {k: v for k, v in pr.items() if k.startswith("sel_")},
                              "endpoint_util": adm.gravity_util(np.stack([q0, q1])).tolist()}
                seed = [cfg.seed, 11]
                pf, _ = route(adm, q0, q1, np.inf, cfg.verify_samples, seed)
                ps, _ = route(adm, q0, q1, cfg.rho_static, cfg.verify_samples, seed)
                pw, _ = route(adm, q0, q1, cfg.rho_swing, cfg.verify_samples, seed)
                if pf is None or ps is None:
                    rec.update(label="UNRESOLVED", reason="no_free_or_static_route")
                    write_json(d / "detour.json", rec)
                    continue
                Lf, Ls = polyline_length(pf), polyline_length(ps)
                rec.update(L_free=Lf, L_static=Ls, detour_ratio=Ls / Lf, peak_free=peak_util(adm, pf),
                           peak_static=peak_util(adm, ps))
                if pw is not None:
                    rec.update(L_swing=polyline_length(pw), peak_swing=peak_util(adm, pw))
                if rec["detour_ratio"] < cfg.detour_ratio or rec["peak_free"] < cfg.short_overload:
                    rec.update(label="NOT_DETOUR", reason="ratio_or_short_route_static")
                    write_json(d / "detour.json", rec)
                    print(cid, "NOT_DETOUR", round(rec["detour_ratio"], 2), round(rec["peak_free"], 3), flush=True)
                    continue
                # static witness (certificate) on the long route
                durs = [segment_duration(adm, x, y, cfg_o.time_safety, rho=cfg.rho_static) for x, y in zip(ps[:-1], ps[1:])]
                tr = witness_trajectory(ps, durs)
                meta = {"payload": p.to_json(), "scene": scene, "start": rec["start"], "goal": rec["goal"],
                        "limits": lim, "gap_check": True}
                export(d / "static_witness", tr, meta | {"kind": "static_witness"})
                rec.update(label="DETOUR", static_witness_duration_s=float(np.sum(durs)),
                           static_waypoints=ps.tolist(), free_waypoints=pf.tolist())
                # fair K107 timing of the static route (same spline construction)
                sw, so = dyn(ps, cfg.rho_static)
                if sw["ok"]:
                    rec["static_route_k107_s"] = sw["result"].duration_s
                    export(d / "static_route_k107", so.sample_trajectory(sw["controls"], sw["result"], n=8193),
                           meta | {"kind": "static_route_k107"})
                # swing (dynamic) witness: least-overloaded short route first (rho ladder)
                attempts = []
                for rho_sw in cfg.swing_ladder:
                    pr_sw, _ = route(adm, q0, q1, rho_sw, cfg.verify_samples, [cfg.seed, 13])
                    if pr_sw is None:
                        attempts.append({"rho": rho_sw, "reason": "no_route"})
                        continue
                    L, pk = polyline_length(pr_sw), peak_util(adm, pr_sw)
                    if L > Ls / cfg.swing_ratio:
                        attempts.append({"rho": rho_sw, "reason": "too_long", "L": L})
                        continue
                    if pk <= 1.0:
                        attempts.append({"rho": rho_sw, "reason": "not_overloaded", "peak": pk})
                        continue
                    dw, do = dyn(pr_sw, pk)
                    attempts.append({"rho": rho_sw, "L": L, "peak": pk, "k107": dw["attempts"]})
                    if dw["ok"]:
                        rec.update(L_swing=L, peak_swing=pk, swing_k107_s=dw["result"].duration_s,
                                   swing_peak_static_util=dw["peak_static_util"], swing_waypoints=pr_sw.tolist(),
                                   swing_dependency="K107 TOPP-RA+Newton with Phase-I start (also used by tested methods)",
                                   swing_candidate=True)
                        d.mkdir(parents=True, exist_ok=True)
                        np.save(d / "swing_controls.npy", dw["controls"])
                        rec["swing_ratio"] = Ls / L
                        export(d / "swing_witness", do.sample_trajectory(dw["controls"], dw["result"], n=8193),
                               meta | {"kind": "swing_witness"})
                        break
                rec["swing_attempts"] = attempts
                write_json(d / "detour.json", rec)
                print(cid, rec["label"], "ratio %.2f" % rec["detour_ratio"], "peak_free %.3f" % rec["peak_free"],
                      "swing" if rec.get("swing_candidate") else "", rec.get("swing_k107_s"),
                      rec.get("static_route_k107_s"), flush=True)


if __name__ == "__main__":
    main()
