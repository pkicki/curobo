# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate planners on a frozen benchmark (export: CUDA env; summarise: any env).

``export``     writes every planner trajectory (``<tag>_traj.npz`` from mine_native, or a
               B/C export) to ``<run>/validation/<task>/<payload>/<tag>.traj.{npz,json}``.
               Then run ``independent_validate --root <run>/validation`` (Pinocchio env).
``summarise``  success = planner success AND independent validation passes (dynamics within
               Scaled95 with a 0.5% sampling tolerance, collision free, endpoints/rest).
               Every benchmark problem is certified solvable, so the denominator is all
               evaluated problems. Reports strata (payload, scene, stratum/subset), latency
               and paired duration ratios vs a reference tag.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import read_json, write_json

TOL = 1.005


def validated(f: Path) -> bool:
    if not f.exists():
        return False
    x = read_json(f)
    u = x.get("utilization_scaled95") or {}
    if not u or x.get("error"):
        return False
    geo = (x["world_clearance_m"] is None or x["world_clearance_m"] >= 0) and x["self_clearance_m"] >= 0
    end = x["start_error_rad"] <= 1e-3 and x["goal_error_rad"] <= 1e-3 and x["end_speed"] <= 1e-2
    return bool(geo and end and x["position_violation_rad"] <= 1e-6
                and all(np.isfinite(v) and v <= TOL for v in u.values()))


def export(run: Path, bank: Path, tags: list[str], gap_check: bool = False) -> None:
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg

    from .run_campaign import export as write
    from .run_campaign import physical_limits

    lim = physical_limits(KinematicsCfg.from_robot_yaml_file("franka.yml").kinematics_config)
    tasks = {t["task_id"]: t for t in read_json(bank / "tasks.json")["tasks"]}
    n = 0
    for tag in tags:
        for f in run.glob(f"cells/*/*/{tag}_traj.npz"):
            rec = read_json(f.with_name(f"{tag}.json"))
            if not (rec["success"] and rec["duration_s"] is not None and np.isfinite(rec["duration_s"])):
                continue
            t = tasks[rec["task_id"]]
            out = run / "validation" / rec["task_id"] / rec["payload"]["label"] / tag
            if Path(str(out) + ".traj.json").exists():
                continue
            write(out, dict(np.load(f)), {"payload": rec["payload"], "scene": t["scene"], "start": t["start"],
                                          "goal": t["goal"], "limits": lim, "variant": tag,
                                          "gap_check": gap_check})
            n += 1
    print("exported", n)


def summarise(run: Path, bank: Path, tags: list[str], reference: str, manifest: Path | None) -> dict:
    tasks = {t["task_id"]: t for t in read_json(bank / "tasks.json")["tasks"]}
    hard = {}
    if manifest is not None:
        hard = {p["cell_id"]: p for p in read_json(manifest)["problems"]}
    rows = []
    for f in sorted(run.glob(f"cells/*/*/{reference}.json")):
        cell = f.parent
        tid, pl = cell.parent.name, cell.name
        t = tasks[tid]
        row = {"task_id": tid, "payload": pl, "scene": t["scene"], "stratum": t["stratum"], "split": t["split"]}
        for tag in tags:
            g = cell / f"{tag}.json"
            if not g.exists():
                row[tag] = None
                continue
            r = read_json(g)
            ok = bool(r["success"] and r["duration_s"] is not None and np.isfinite(r["duration_s"]))
            v = validated(run / "validation" / tid / pl / f"{tag}.validation.json") if ok else False
            row[tag] = {"planned": ok, "valid": v, "duration_s": r["duration_s"] if ok else None,
                        "wall_s": r["wall_s"], "route": r.get("route")}
        if tid in hard:
            h = hard[tid]
            row["hard"] = {k: h.get(k) for k in ("labels", "swing_k107_s", "static_route_k107_s",
                                                  "static_witness_duration_s", "detour_ratio", "rho_star_hat")}
        rows.append(row)

    def strata(key):
        out = defaultdict(dict)
        groups = defaultdict(list)
        for r in rows:
            ks = r[key].split("+") if key == "stratum" else [r[key]]
            for k in ks:
                groups[k].append(r)
        for k, rs in sorted(groups.items()):
            for tag in tags:
                xs = [r[tag] for r in rs if r.get(tag)]
                if xs:
                    out[k][tag] = f"{sum(x['valid'] for x in xs)}/{len(xs)}"
        return dict(out)

    res = {"cells": len(rows), "tags": tags, "reference": reference}
    for tag in tags:
        xs = [r[tag] for r in rows if r.get(tag)]
        if not xs:
            continue
        walls_ok = [x["wall_s"] for x in xs if x["planned"]]
        walls_fail = [x["wall_s"] for x in xs if not x["planned"]]
        res[tag] = {"n": len(xs), "planned": sum(x["planned"] for x in xs), "valid": sum(x["valid"] for x in xs),
                    "valid_rate": sum(x["valid"] for x in xs) / len(xs),
                    "planned_but_invalid": sum(x["planned"] and not x["valid"] for x in xs),
                    "wall_median_success_s": float(np.median(walls_ok)) if walls_ok else None,
                    "wall_median_failure_s": float(np.median(walls_fail)) if walls_fail else None,
                    "wall_p95_s": float(np.percentile([x["wall_s"] for x in xs], 95))}
        if tag != reference:
            ratio = [r[tag]["duration_s"] / r[reference]["duration_s"] for r in rows
                     if r.get(tag) and r.get(reference) and r[tag]["valid"] and r[reference]["valid"]]
            res[tag]["duration_ratio_vs_ref"] = (None if not ratio else
                                                 {"n": len(ratio), "median": float(np.median(ratio)),
                                                  "mean": float(np.mean(ratio)),
                                                  "frac_5pct_shorter": float(np.mean(np.asarray(ratio) <= 0.95))})
            res[tag]["rescued_ref_failures"] = sum(1 for r in rows if r.get(tag) and r.get(reference)
                                                   and r[tag]["valid"] and not r[reference]["valid"])
            res[tag]["lost_ref_successes"] = sum(1 for r in rows if r.get(tag) and r.get(reference)
                                                 and not r[tag]["valid"] and r[reference]["valid"])
    for key in ("payload", "scene", "stratum"):
        res[f"by_{key}"] = strata(key)
    sw = [r for r in rows if r.get("hard") and "SWING" in (r["hard"]["labels"] or [])]
    if sw:
        res["swing"] = {tag: {"valid": sum(bool(r.get(tag) and r[tag]["valid"]) for r in sw), "n": len(sw),
                              "median_duration_vs_swing_witness": (lambda x: float(np.median(x)) if x else None)(
                                  [r[tag]["duration_s"] / r["hard"]["swing_k107_s"] for r in sw
                                   if r.get(tag) and r[tag]["valid"]])} for tag in tags}
    res["rows"] = rows
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("export", "summarise"))
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--tags", nargs="+", default=["native_R1", "hybrid_R1"])
    ap.add_argument("--reference", default="native_R1")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--gap-check", action="store_true", help="export for strict (gap-checked) certification")
    a = ap.parse_args()
    if a.mode == "export":
        export(a.run, a.bank, a.tags, a.gap_check)
        return
    res = summarise(a.run, a.bank, a.tags, a.reference, a.manifest)
    write_json(a.run / "evaluation.json", res)
    for k, v in res.items():
        if k != "rows":
            print(k, v)


if __name__ == "__main__":
    main()
