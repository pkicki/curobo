# SPDX-License-Identifier: Apache-2.0
"""Prepare a separate immutable 3-case native pre-TimeOpt diagnostic arm.

The full cuRobo native instrumented planner must not overwrite the historical
32-restart negative-control evidence. Copy ONLY the frozen task/profile JSON
from the two source experiments, verify their source receipts, and write an
isolated input root for the opt-in probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cohort_retime import ROOT
from .geometry_feasibility_core import sha256_file
from .torque_active import read_profile
from .torque_guided_rescue import ROOTS, SCENARIO

OUT = ROOT / "research/joint_time_oracle/results/native_pretimeopt_probe_20261108"


def freeze(out: Path) -> dict:
    arm = out / SCENARIO
    arm.mkdir(parents=True, exist_ok=True)
    tasks = []
    records = {}
    for case, (source_name, experiment_name) in ROOTS.items():
        source = ROOT / "research/joint_time_oracle/results" / source_name
        experiment = ROOT / "research/joint_time_oracle/results" / experiment_name
        rows = json.loads((source / "tasks.json").read_text())
        task = next((t for t in rows if t["case_id"] == case), None)
        if task is None:
            raise ValueError("source task missing " + case)
        prev = experiment / SCENARIO / case / "torque_profile.json"
        prof = read_profile(prev)
        if prof["source_task_sha256"] != task["task_sha256"]:
            raise ValueError("source task and derated limit SHA mismatch")
        original = (
            experiment / SCENARIO / "native_full" / case / "native_candidates.json"
        )
        old = json.loads(original.read_text())
        if not all(
            old.get(f"F{i:03d}", {}).get("status") == "FAILED" for i in range(32)
        ):
            raise ValueError("historical native failure budget not complete: " + case)
        tasks.append(task)
        dest = arm / case
        dest.mkdir(parents=True, exist_ok=True)
        profdest = dest / "torque_profile.json"
        if profdest.exists() and profdest.read_bytes() != prev.read_bytes():
            raise ValueError("source profile modified under frozen probe")
        if not profdest.exists():
            profdest.write_bytes(prev.read_bytes())
        records[case] = {
            "task_sha256": task["task_sha256"],
            "torque_profile_sha256": prof["profile_sha256"],
            "source_task_inventory_sha256": sha256_file(source / "tasks.json"),
            "source_native_failure_sha256": sha256_file(original),
            "source_torque_profile_file_sha256": sha256_file(prev),
        }
    registry = {
        "schema_version": 1,
        "scenario": SCENARIO,
        "cases": records,
        "probe": "research-only pre-TimeOpt snapshot, NOT latency-qualified",
        "interpretation": (
            "compare recorded stage/candidate failures, not claim global impossibility"
        ),
    }
    path = arm / "probe_source_manifest.json"
    if path.exists() and json.loads(path.read_text()) != registry:
        raise ValueError("probe source manifest is frozen")
    path.write_text(json.dumps(registry, indent=2) + "\n")
    tasksfile = arm / "tasks.json"
    if tasksfile.exists() and json.loads(tasksfile.read_text()) != tasks:
        raise ValueError("probe task inventory changed")
    tasksfile.write_text(json.dumps(tasks, indent=2) + "\n")
    return registry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUT)
    a = p.parse_args()
    print(json.dumps(freeze(a.output), indent=2))


if __name__ == "__main__":
    main()
