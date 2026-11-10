# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze the search algorithm (source hashes + SearchConfig) before held-out runs.

``python -m research.timeopt_guided.freeze --config cfg.json --output freeze.json``
``run_campaign --split confirmation`` refuses to run unless ``--freeze`` matches the
current source files and the configuration it is given.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .common import ROOT, canonical_hash, read_json, sha256_file, write_json

FROZEN_FILES = ("common.py", "geometry.py", "k107_oracle.py", "collision_screen.py", "guided_search.py",
                "run_campaign.py", "independent_validate.py")


def source_hashes() -> dict:
    d = Path(__file__).resolve().parent
    return {f: sha256_file(d / f) for f in FROZEN_FILES}


def check(freeze_file: Path, config: dict) -> None:
    fz = read_json(freeze_file)
    if fz["source_sha256"] != source_hashes():
        raise SystemExit("frozen source hashes differ from the working tree: refuse held-out run")
    if fz["config_sha256"] != canonical_hash(config):
        raise SystemExit("search configuration differs from the frozen one: refuse held-out run")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="JSON of SearchConfig overrides")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--toc-commit", required=True, help="time_optimal_curobo commit used for K107")
    a = ap.parse_args()
    cfg = read_json(a.config)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    write_json(a.output, {"curobo_head": head, "toc_commit": a.toc_commit, "config": cfg,
                          "config_sha256": canonical_hash(cfg), "source_sha256": source_hashes()},
               overwrite=False)


if __name__ == "__main__":
    main()
