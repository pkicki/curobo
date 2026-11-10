# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze a benchmark of task x payload cells with existence labels (any env).

Run AFTER ``existence_oracle`` and ``independent_validate --root <oracle out>`` and BEFORE
evaluating any planner on the bank. Labels:
  CERTIFIED_SOLVABLE    oracle witness passed independent certification (Scaled95 dynamics,
                        sphere collisions with inter-sample gap check, endpoints/rest)
  WITNESS_REJECTED      oracle produced a witness that FAILED certification (oracle defect;
                        treated as uncertified, listed for inspection)
  UNCERTIFIED           no rho < 1 connection found within the oracle budget (NOT infeasible)
  ENDPOINT_INADMISSIBLE endpoint static gravity / payload collision (excluded by the bank)
Difficulty features (method independent): rho at which the witness was found
(gravity tightness), number of witness segments, certified clearance slacks,
endpoint gravity utilisation, stratum, scene, payload, split.
The manifest stores SHA256 of the bank, oracle config, oracle/validator sources and every
witness, and is never overwritten.
"""

from __future__ import annotations

import argparse
import subprocess
from collections import Counter
from pathlib import Path

from .common import ROOT, canonical_hash, read_json, sha256_file, write_json

SOURCES = ("existence_oracle.py", "independent_validate.py", "collision_screen.py", "common.py", "task_bank.py")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--oracle", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="manifest JSON path (must not exist)")
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit(f"{a.output} exists: a frozen manifest is never overwritten")
    bank = read_json(a.bank / "tasks.json")
    tasks = {t["task_id"]: t for t in bank["tasks"]}
    cells = read_json(a.bank / "cells.json")
    rows, missing = [], 0
    for c in cells:
        tid, pl = c["task_id"], c["payload"]["label"]
        t = tasks[tid]
        row = {"task_id": tid, "payload": pl, "scene": t["scene"], "stratum": t["stratum"], "split": t["split"],
               "mass_kg": c["payload"]["mass_kg"], "endpoint_gravity_util_bank": c["endpoint_gravity_util"]}
        if not c["admissible"]:
            row["label"] = "ENDPOINT_INADMISSIBLE"
            row["reason"] = c["reason"]
            rows.append(row)
            continue
        d = a.oracle / "cells" / tid / pl
        if not (d / "oracle.json").exists():
            missing += 1
            row["label"] = "NOT_RUN"
            rows.append(row)
            continue
        o = read_json(d / "oracle.json")
        if o["status"] != "WITNESS_FOUND":
            row["label"] = "UNCERTIFIED"
            row["oracle_tries"] = o["tries"]
        else:
            v = read_json(d / "witness.validation.json") if (d / "witness.validation.json").exists() else {}
            row["label"] = "CERTIFIED_SOLVABLE" if v.get("certified") else "WITNESS_REJECTED"
            row.update(rho=o["rho"], margin_m=o.get("margin_m"), samples=o["samples"], segments=len(o["waypoints"]) - 1,
                       witness_duration_s=o["witness_duration_s"],
                       witness_sha256=sha256_file(d / "witness.traj.npz"),
                       gap_check=v.get("gap_check"), utilization_scaled95=v.get("utilization_scaled95"))
        row["oracle_wall_s"] = o["wall_s"]
        rows.append(row)
    if missing:
        raise SystemExit(f"{missing} admissible cells have no oracle result: refusing to freeze a partial set")
    src = Path(__file__).resolve().parent
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    manifest = {
        "curobo_head": head,
        "bank_sha256": sha256_file(a.bank / "tasks.json"),
        "cells_sha256": sha256_file(a.bank / "cells.json"),
        "oracle_config": read_json(a.oracle / "oracle_config.json"),
        "source_sha256": {f: sha256_file(src / f) for f in SOURCES},
        "label_counts": dict(Counter(r["label"] for r in rows)),
        "label_counts_by_split": {s: dict(Counter(r["label"] for r in rows if r["split"] == s))
                                  for s in ("development", "confirmation")},
        "cells": rows,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    write_json(a.output, manifest, overwrite=False)
    print(manifest["label_counts"], manifest["label_counts_by_split"])


if __name__ == "__main__":
    main()
