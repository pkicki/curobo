# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Best-known solutions (BKS) for frozen benchmark problems (CUDA env; offline, quality first).

Global multi-candidate search + long local joint path-time optimisation (in the spirit of
Shiller & Dubowsky: search geometry with a time-optimal retimer inside the loop):

1. Candidate routes
   * roadmap routes (one LazyRoadmap per problem) with peak static gravity utilisation
     <= each threshold in ``THRESHOLDS`` (free space .. strictly static), plus
     ``ALTERNATIVES`` edge-penalised alternatives at selected thresholds (other homotopies);
   * planner trajectories from an evaluation run (native/hybrid/B/C), if available;
   * the certified oracle witnesses (main-benchmark witness, hard static/swing routes).
2. Every candidate -> rest-to-rest quintic spline fits (arc-length) -> collision screen ->
   K107 retiming (overload-aware Phase-I start, default then full-resolution LP).
3. The ``TOP`` fastest distinct candidates -> ``JointRefiner`` (alternating K107 law /
   gradient path update, quintic) with basin-hopping restarts (smooth random perturbation,
   accepted only if K107-feasible and faster).
4. The best ``EXPORT`` trajectories are exported with ``gap_check`` for strict independent
   certification; ``summarise`` picks the fastest CERTIFIED one as the best-known solution.
Durations are upper bounds on the optimum (no optimality claim).
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .common import payload_grid, read_json, write_json
from .geometry import fit_quintic

THRESHOLDS = (np.inf, 1.3, 1.15, 1.05, 0.999, 0.97, 0.9)
ALTERNATIVES = {np.inf: 2, 0.97: 2}


@dataclass
class BKSConfig:
    roadmap_samples: int = 300_000
    spline_controls: tuple = (16, 30)
    top: int = 4
    restarts: int = 2
    perturb_rad: float = 0.06
    refine_iterations: int = 40
    refine_inner_steps: int = 60
    export: int = 3
    refine_budget_s: float = 600.0  # per refinement run (guard against pathological solver stalls)
    seed: int = 20261115


def resample_arclength(q: np.ndarray, n: int = 1025) -> tuple[np.ndarray, np.ndarray]:
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arc) > 1e-9]
    arc, q = arc[keep], q[keep]
    s = np.linspace(0, 1, n)
    return s, np.stack([np.interp(s, arc / arc[-1], q[:, j]) for j in range(q.shape[1])], 1)


def roadmap_routes(rm, adm, shortcut) -> list[tuple[str, np.ndarray]]:
    out = []
    for thr in THRESHOLDS:
        r = rm.shortest(thr)
        if r["path"] is None:
            continue
        out.append((f"roadmap_thr{thr}", shortcut(adm, r["path"], thr)))
        n_alt = ALTERNATIVES.get(thr, 0)
        if n_alt:
            saved = rm.length.copy()
            path = r["path"]
            for k in range(n_alt):
                idx = [int(np.argmin(np.linalg.norm(rm.nodes - p, axis=1))) for p in path]
                rm.length[rm.edge_ids(idx)] *= 4.0  # penalise the previous route's edges
                ra = rm.shortest(thr)
                if ra["path"] is None:
                    break
                path = ra["path"]
                out.append((f"roadmap_thr{thr}_alt{k + 1}", shortcut(adm, path, thr)))
            rm.length[:] = saved
    return out


def main() -> None:
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg

    from .collision_screen import PathCollisionChecker
    from .common import quintic_knots
    from .existence_oracle import Admissibility, LazyRoadmap, OracleConfig, shortcut
    from .joint_refine import JointRefiner, RefineConfig
    from .k107_oracle import K107Oracle
    from .run_campaign import export, physical_limits

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--eval-run", type=Path, default=None, help="evaluation run with planner trajectories")
    ap.add_argument("--eval-tags", nargs="*", default=["native_R1", "hybrid2_R1", "B2_R1", "C2_R1"])
    ap.add_argument("--oracle-run", type=Path, default=None, help="existence-oracle run (main benchmark witnesses)")
    ap.add_argument("--hard-manifest", type=Path, default=None, help="hard manifest (static/swing witnesses)")
    ap.add_argument("--split", choices=("development", "confirmation", "all"), default="all")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--config", default="{}")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = BKSConfig(**json.loads(a.config))
    cfg_o = OracleConfig(seed=cfg.seed)
    lim = physical_limits(KinematicsCfg.from_robot_yaml_file("franka.yml").kinematics_config)
    write_json(a.output / "bks_config.json", {"bks": asdict(cfg), "args": {k: str(v) for k, v in vars(a).items()}})
    bank = read_json(a.bank / "tasks.json")
    tasks = {t["task_id"]: t for t in bank["tasks"]}
    payloads = {p.label: p for p in payload_grid()}
    hard = {p["cell_id"]: p for p in read_json(a.hard_manifest)["problems"]} if a.hard_manifest else {}
    cells = [(c["task_id"], c["payload"]["label"]) for c in read_json(a.bank / "cells.json") if c["admissible"]
             and (a.split == "all" or tasks[c["task_id"]]["split"] == a.split)]
    cells.sort(key=lambda x: (tasks[x[0]]["scene"], x[1], x[0]))
    cells = [c for i, c in enumerate(cells) if i % a.nshards == a.shard][: a.limit]
    cache: dict = {}
    for tid, pl in cells:
        d = a.output / "cells" / tid / pl
        if (d / "bks.json").exists():
            continue
        t, p = tasks[tid], payloads[pl]
        q0, q1 = np.asarray(t["start"], float), np.asarray(t["goal"], float)
        key = (t["scene"], pl)
        if cache.get("key") != key:
            cache.clear()
            cache["key"] = key
            cache["adm"] = Admissibility(t["scene"], p, cfg_o, cfg_o.margins[0])
            cache["oracles"] = [K107Oracle(p, overload_aware_start=True, start_method="lp"),
                                K107Oracle(p, overload_aware_start=True, start_method="lp", warmup=False,
                                           lp_options={"lp_stride": 1, "min_slack": 0.001, "rounds": 2})]
            ch = PathCollisionChecker(t["scene"])
            ch.set_payload(p)
            cache["checker"] = ch
        adm, oracles, checker = cache["adm"], cache["oracles"], cache["checker"]
        t0 = time.perf_counter()
        rng = np.random.default_rng([cfg.seed, zlib.crc32(f"{tid}/{pl}".encode())])
        # 1) candidate routes
        cands: list[tuple[str, np.ndarray]] = []
        rm = LazyRoadmap(adm, q0, q1, cfg.roadmap_samples, rng)
        if rm.endpoints_ok:
            cands += roadmap_routes(rm, adm, shortcut)
        if a.eval_run is not None:
            for tag in a.eval_tags:
                f = a.eval_run / "cells" / tid / pl / f"{tag}_traj.npz"
                if f.exists():
                    z = np.load(f)
                    if np.isfinite(z["q"]).all():
                        cands.append((f"planner_{tag}", z["q"]))
        if a.oracle_run is not None:
            o = a.oracle_run / "cells" / tid / pl / "oracle.json"
            if o.exists() and read_json(o).get("waypoints"):
                cands.append(("oracle_witness", np.asarray(read_json(o)["waypoints"])))
        if tid in hard:
            src = Path(hard[tid]["source"]) / "cells" / tid
            rec = read_json(src / "detour.json") if (src / "detour.json").exists() else read_json(src / "hard.json")
            for k in ("static_waypoints", "swing_waypoints", "waypoints", "free_waypoints"):
                if rec.get(k):
                    cands.append((f"hard_{k}", np.asarray(rec[k])))
        # 2) spline fits + K107
        scored = []
        for name, route in cands:
            route = np.asarray(route, float)
            if len(route) < 2 or np.linalg.norm(route[-1] - route[0]) < 1e-9 and len(route) < 3:
                continue
            s, dense = resample_arclength(route)
            for n_ctrl in cfg.spline_controls:
                c = fit_quintic(s, dense, n=n_ctrl, smooth=1e-8)
                c[:3], c[-3:] = q0, q1
                if not checker.check_paths([c])[0]:
                    continue
                for oi, o in enumerate(oracles):
                    r = o.retime(c, diagnose_infeasible=False)
                    if r.feasible:
                        scored.append({"name": name, "controls": c, "result": r, "oracle": oi,
                                       "duration_s": r.duration_s, "stage": "candidate"})
                        break
        rec = {"task_id": tid, "payload": pl, "scene": t["scene"], "split": t["split"],
               "candidates": len(cands), "feasible_candidates": len(scored),
               "candidate_durations": sorted(({"name": x["name"], "n": len(x["controls"]),
                                               "duration_s": x["duration_s"]} for x in scored),
                                              key=lambda x: x["duration_s"])}
        if not scored:
            rec.update(status="NO_FEASIBLE_CANDIDATE", wall_s=time.perf_counter() - t0)
            write_json(d / "bks.json", rec)
            print(tid, pl, "NO_FEASIBLE_CANDIDATE", flush=True)
            continue
        # 3) refine the TOP distinct candidates (by route name) with restarts
        scored.sort(key=lambda x: x["duration_s"])
        seeds, seen = [], set()
        for x in scored:
            if x["name"] not in seen:
                seeds.append(x)
                seen.add(x["name"])
            if len(seeds) >= cfg.top:
                break
        rcfg = RefineConfig(iterations=cfg.refine_iterations, inner_steps=cfg.refine_inner_steps, sigma=1.05,
                            sigma_min=1.002, time_budget_s=cfg.refine_budget_s)
        finals = list(scored)
        for x in seeds:
            o = oracles[x["oracle"]]
            refiner = JointRefiner(o, checker, np.asarray(lim["position"]))
            knots = quintic_knots(len(x["controls"]))
            best_c, best_r = x["controls"], x["result"]
            for rs in range(cfg.restarts + 1):
                start = best_c
                if rs > 0:  # basin hopping: smooth random perturbation of the interior
                    n = len(start)
                    bump = np.sin(np.pi * np.linspace(0, 1, n))[:, None]
                    start = start + cfg.perturb_rad * bump * rng.normal(size=(1, 7))
                    start[:3], start[-3:] = q0, q1
                    if not checker.check_paths([start])[0]:
                        continue
                    r0 = o.retime(start, diagnose_infeasible=False)
                    if not r0.feasible:
                        continue
                else:
                    r0 = best_r
                log = refiner.refine(start, knots, rcfg, start_result=r0, degree=5)
                if log.best_result is not None and log.best_result.duration_s < best_r.duration_s:
                    best_c, best_r = log.best_controls, log.best_result
            finals.append({"name": x["name"] + "+refined", "controls": best_c, "result": best_r,
                           "oracle": x["oracle"], "duration_s": best_r.duration_s, "stage": "refined"})
        finals.sort(key=lambda x: x["duration_s"])
        exported = []
        for k, x in enumerate(finals[: cfg.export]):
            o = oracles[x["oracle"]]
            tr = o.sample_trajectory(x["controls"], x["result"], n=8193)
            export(d / f"bks_{k}", tr, {"payload": p.to_json(), "scene": t["scene"], "start": t["start"],
                                        "goal": t["goal"], "limits": lim, "gap_check": True,
                                        "kind": "best_known_candidate", "source": x["name"]})
            np.save(d / f"bks_{k}_controls.npy", x["controls"])
            exported.append({"rank": k, "name": x["name"], "stage": x["stage"], "duration_s": x["duration_s"],
                             "k107_duration_s": x["result"].duration_s})
        rec.update(status="EXPORTED", exported=exported, wall_s=time.perf_counter() - t0,
                   best_candidate_s=scored[0]["duration_s"], best_final_s=finals[0]["duration_s"])
        write_json(d / "bks.json", rec)
        import torch

        from .k107_oracle import clear_k107_caches

        cleared = clear_k107_caches()
        print(tid, pl, "cand %.3f -> final %.3f" % (scored[0]["duration_s"], finals[0]["duration_s"]),
              finals[0]["name"], "%.0fs" % rec["wall_s"],
              "gpu_reserved %.1fGB peak %.1fGB" % (torch.cuda.memory_reserved() / 1e9,
                                                   torch.cuda.max_memory_reserved() / 1e9),
              "cleared", cleared, flush=True)


if __name__ == "__main__":
    main()
