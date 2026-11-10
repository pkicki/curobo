# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write a planner-runner bank (tasks.json + cells.json) from a frozen hard manifest.

Each hard problem becomes one task with exactly one admissible cell (its own payload).
``stratum`` carries the problem's subset labels joined by '+', e.g. "DETOUR+SWING".
The bank records the source manifest SHA so evaluation results are tied to it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import payload_grid, read_json, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    m = read_json(a.manifest)
    payloads = {p.label: p for p in payload_grid()}
    tasks, cells = [], []
    for p in m["problems"]:
        tasks.append({"task_id": p["cell_id"], "scene": p["scene"], "stratum": "+".join(p["labels"]),
                      "split": p["split"], "start": p["start"], "goal": p["goal"]})
        cells.append({"task_id": p["cell_id"], "payload": payloads[p["payload"]].to_json(), "admissible": True,
                      "reason": None, "endpoint_gravity_util": max(p["endpoint_util"])})
    write_json(a.output / "tasks.json", {"source_manifest_sha256": m["manifest_sha256"],
                                         "payloads": [p.to_json() for p in payloads.values()], "tasks": tasks})
    write_json(a.output / "cells.json", cells)
    print(len(tasks), "tasks")


if __name__ == "__main__":
    main()
