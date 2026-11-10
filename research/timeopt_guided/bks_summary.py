# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze best-known solutions (fastest CERTIFIED export per problem) and compare methods (any env).

Run after ``best_known`` and ``independent_validate --root <bks run>``. The BKS of a problem is
the fastest of its exported candidates whose independent certification passed (Scaled95 with no
tolerance, sphere collisions with the inter-sample gap check, endpoints/rest). Problems without
a certified export are reported. Optionally compares an evaluation run's methods with the BKS
(duration ratio method / BKS on problems the method solves, validated).
"""

from __future__ import annotations

import argparse
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import ROOT, canonical_hash, read_json, sha256_file, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bks", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--evaluation", type=Path, default=None, help="evaluation.json of a method run")
    ap.add_argument("--benchmark-manifest", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit(f"{a.output} exists: never overwritten")
    rows, missing = [], Counter()
    for f in sorted(a.bks.glob("cells/*/*/bks.json")):
        r = read_json(f)
        d = f.parent
        best = None
        for e in r.get("exported", []):
            v = d / f"bks_{e['rank']}.validation.json"
            if v.exists() and read_json(v).get("certified"):
                if best is None or e["duration_s"] < best["duration_s"]:
                    best = e | {"sha256": sha256_file(d / f"bks_{e['rank']}.traj.npz")}
        if best is None:
            missing[r["status"] if r["status"] != "EXPORTED" else "NO_CERTIFIED_EXPORT"] += 1
            continue
        rows.append({"task_id": r["task_id"], "payload": r["payload"], "scene": r["scene"], "split": r["split"],
                     "bks_duration_s": best["duration_s"], "bks_source": best["name"], "bks_stage": best["stage"],
                     "bks_rank": best["rank"], "bks_sha256": best["sha256"],
                     "best_raw_candidate_s": r["best_candidate_s"], "wall_s": r["wall_s"]})
    out = {"bks_dir": str(a.bks.resolve().relative_to(ROOT)),
           "benchmark_manifest_sha256": read_json(a.benchmark_manifest).get("manifest_sha256"),
           "curobo_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                         text=True).stdout.strip(),
           "config": read_json(a.bks / "bks_config.json"), "problems_with_bks": len(rows),
           "problems_without_bks": dict(missing),
           "refinement_gain_median": float(np.median([r["bks_duration_s"] / r["best_raw_candidate_s"] for r in rows]))
           if rows else None,
           "bks_source_counts": dict(Counter(r["bks_source"].split("+")[0].split("_alt")[0] for r in rows)),
           "problems": rows}
    if a.evaluation is not None:
        ev = read_json(a.evaluation)
        bks = {(r["task_id"], r["payload"]): r["bks_duration_s"] for r in rows}
        comp = defaultdict(list)
        for r in ev["rows"]:
            b = bks.get((r["task_id"], r["payload"]))
            if b is None:
                continue
            for tag in ev["tags"]:
                x = r.get(tag)
                if x and x["valid"]:
                    comp[tag].append(x["duration_s"] / b)
        out["methods_vs_bks"] = {t: {"n_solved_with_bks": len(v), "median_ratio": float(np.median(v)),
                                     "mean_ratio": float(np.mean(v)),
                                     "frac_within_5pct": float(np.mean(np.asarray(v) <= 1.05)),
                                     "frac_method_faster_than_bks": float(np.mean(np.asarray(v) < 0.999))}
                                 for t, v in comp.items() if v}
    out["manifest_sha256"] = canonical_hash(out)
    write_json(a.output, out, overwrite=False)
    print({k: v for k, v in out.items() if k not in ("problems", "config")})


if __name__ == "__main__":
    main()
