# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze the hard benchmark subsets (TIGHT / DETOUR / SWING) into one manifest (any env).

Run after ``hard_bank`` / ``detour_bank`` and ``independent_validate --root`` on every
generation directory, BEFORE evaluating methods. Only problems whose witness passed
independent certification (Scaled95 dynamics, sphere collisions with the inter-sample
gap check, endpoints/rest) are included:
  TIGHT_STATIC    hard_bank label TIGHT_STATIC + certified quasi-static witness
  DETOUR          detour_bank label DETOUR + certified quasi-static witness on the long route
  SWING           DETOUR + certified dynamic (K107) witness on a short route whose peak static
                  utilisation is >= SWING_MIN_PEAK (a genuine overload)
  SWING_MARGINAL  as SWING but 1 < peak < SWING_MIN_PEAK (listed, not counted as SWING)
A problem can be DETOUR and SWING at once (``labels`` list). The manifest records the
generation directories, configurations, source hashes and every witness hash, and is
never overwritten.
"""

from __future__ import annotations

import argparse
import subprocess
from collections import Counter
from pathlib import Path

from .common import ROOT, canonical_hash, read_json, sha256_file, write_json

SWING_MIN_PEAK = 1.01
SOURCES = ("hard_bank.py", "detour_bank.py", "existence_oracle.py", "independent_validate.py", "k107_oracle.py",
           "geometry.py", "common.py", "collision_screen.py")


def certified(path: Path) -> bool:
    f = path.with_name(path.name + ".validation.json")
    return f.exists() and bool(read_json(f).get("certified"))


def witness_hash(path: Path) -> str:
    return sha256_file(path.with_name(path.name + ".traj.npz"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tight", type=Path, nargs="*", default=[])
    ap.add_argument("--detour", type=Path, nargs="*", default=[])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    a.tight = [d.resolve() for d in a.tight]
    a.detour = [d.resolve() for d in a.detour]
    if a.output.exists():
        raise SystemExit(f"{a.output} exists: a frozen manifest is never overwritten")
    problems: dict[str, dict] = {}
    for d in a.tight:
        for f in sorted(d.glob("cells/*/hard.json")):
            r = read_json(f)
            w = f.parent / "witness"
            if r["label"] != "TIGHT_STATIC" or not certified(w):
                continue
            problems[r["cell_id"]] = {
                "cell_id": r["cell_id"], "labels": ["TIGHT_STATIC"], "source": str(d.relative_to(ROOT)),
                "scene": r["scene"], "payload": r["payload"], "split": r["split"], "start": r["start"],
                "goal": r["goal"], "rho_star_hat": r["rho_star_hat"], "tightness_driver": r.get("tightness_driver"),
                "endpoint_util": r["endpoint_util"], "straight_peak_util": r["straight_peak_util"],
                "segments": r["segments"], "length_ratio": r["length_ratio"],
                "witness_duration_s": r["witness_duration_s"], "witness_sha256": witness_hash(w)}
    for d in a.detour:
        for f in sorted(d.glob("cells/*/detour.json")):
            r = read_json(f)
            w = f.parent / "static_witness"
            if r["label"] != "DETOUR" or not certified(w):
                continue
            p = {"cell_id": r["cell_id"], "labels": ["DETOUR"], "source": str(d.relative_to(ROOT)),
                 "scene": r["scene"], "payload": r["payload"], "split": r["split"], "start": r["start"],
                 "goal": r["goal"], "endpoint_util": r["endpoint_util"], "L_free": r["L_free"],
                 "L_static": r["L_static"], "detour_ratio": r["detour_ratio"], "peak_free": r["peak_free"],
                 "static_witness_sha256": witness_hash(w), "static_witness_duration_s": r["static_witness_duration_s"],
                 "static_route_k107_s": r.get("static_route_k107_s")}
            sw = f.parent / "swing_witness"
            if r.get("swing_candidate") and certified(sw):
                p.update(L_swing=r["L_swing"], swing_ratio=r["swing_ratio"], peak_swing=r["peak_swing"],
                         swing_k107_s=r["swing_k107_s"], swing_witness_sha256=witness_hash(sw),
                         swing_dependency=r.get("swing_dependency"))
                p["labels"].append("SWING" if r["peak_swing"] >= SWING_MIN_PEAK else "SWING_MARGINAL")
            problems[r["cell_id"]] = p
    rows = sorted(problems.values(), key=lambda x: x["cell_id"])
    src = Path(__file__).resolve().parent
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    counts = Counter(lab for r in rows for lab in r["labels"])
    manifest = {
        "curobo_head": head, "note": a.note, "swing_min_peak": SWING_MIN_PEAK,
        "generation_dirs": {"tight": [str(d.relative_to(ROOT)) for d in a.tight],
                            "detour": [str(d.relative_to(ROOT)) for d in a.detour]},
        "generation_configs": {str(d.relative_to(ROOT)): read_json(next(iter(d.glob("*_config.json"))))
                               for d in a.tight + a.detour},
        "source_sha256": {f: sha256_file(src / f) for f in SOURCES},
        "label_counts": dict(counts),
        "label_counts_by_split": {s: dict(Counter(lab for r in rows if r["split"] == s for lab in r["labels"]))
                                  for s in ("development", "confirmation")},
        "problems": rows,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    write_json(a.output, manifest, overwrite=False)
    print(manifest["label_counts"], manifest["label_counts_by_split"])


if __name__ == "__main__":
    main()
