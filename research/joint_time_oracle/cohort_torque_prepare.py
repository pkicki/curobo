# SPDX-License-Identifier: Apache-2.0
"""Freeze two torque-active stress regimes from historical A40 trajectories.

Requires the isolated Pinocchio environment. No GPU planning, tuning from
new B/C/D outputs, or mutation of historical result files occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from .cohort_retime import dynamics
from .torque_active import make_profile, write_profile, canonical_hash

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research/joint_time_oracle/results/cohort_pilot_isolated_20261106"
DEFAULT_OUT = ROOT / "research/joint_time_oracle/results/torque_stress_20261107"
NOMINAL = ROOT / "research/joint_time_oracle/results/native_reconstruction_20261106/baseline_validation_states.npz"
REGIMES = (("moderate", 0.90), ("severe", 1.10))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def endpoint_gravity(task: dict, model, data, state, qi, vi) -> np.ndarray:
    out = []
    zero = np.zeros(model.nv)
    for endpoint in ("start", "goal"):
        q = state.copy()
        q[qi] = task[endpoint]
        out.append(pin.rnea(model, data, q, zero, zero)[vi].copy())
    return np.stack(out)


def prepare(source: Path, output: Path, *, targets=REGIMES) -> dict:
    tasks = json.loads((source / "tasks.json").read_text())
    if len({task["case_id"] for task in tasks}) != len(tasks):
        raise ValueError("duplicate source task IDs")
    model, data, q, qi, vi = dynamics()
    with np.load(NOMINAL) as base:
        limits = np.asarray(base["torque_bounds"], dtype=np.float64)
    summary = {"protocol": "frozen A40-based synthetic actuator derating",
               "source_tasks_sha256": sha(source / "tasks.json"),
               "nominal_bounds_sha256": sha(NOMINAL),
               "source_task_count": len(tasks), "regimes": {},
               "model": "existing model-aligned Pinocchio Panda; NO payload change",
               "independence": "both regimes reuse the SAME source tasks; do not pool as independent observations",
               "selection": "calibrated exclusively from historical A40 sampled torque and endpoint static gravity"}
    output.mkdir(parents=True, exist_ok=True)
    for regime, target in targets:
        out = output / regime
        out.mkdir(parents=True, exist_ok=True)
        admitted, profiles, failures = [], {}, []
        for task in tasks:
            case = task["case_id"]
            src = source / case / "a40_pin_torque.npz"
            if not src.exists():
                failures.append({"source_case_id": case, "reason": "missing frozen A40 Pinocchio torques"})
                continue
            with np.load(src) as archive:
                torques = np.asarray(archive["torque"], dtype=np.float64)
            gravity = endpoint_gravity(task, model, data, q, qi, vi)
            profile = make_profile(
                case_id=f"{regime}::{case}", source_case_id=case,
                regime=regime, nominal_bounds=limits,
                baseline_torques=torques, endpoint_gravity_torques=gravity,
                target=target, source_task_sha256=task["task_sha256"],
                source_torque_sha256=sha(src))
            write_profile(out / case / "torque_profile.json", profile)
            profiles[case] = profile.to_json()
            if profile.admitted:
                admitted.append(task)
            else:
                failures.append({"source_case_id": case, "reason": profile.rejection_reason,
                                 "baseline_utilization": profile.observed_baseline_utilization,
                                 "endpoint_gravity_utilization": profile.endpoint_gravity_utilization})
        (out / "tasks.json").write_text(json.dumps(admitted, indent=2) + "\n")
        (out / "all_profiles.json").write_text(json.dumps(profiles, indent=2) + "\n")
        (out / "rejections.json").write_text(json.dumps(failures, indent=2) + "\n")
        summary["regimes"][regime] = {"target": target, "attempted": len(tasks),
                                      "admitted": len(admitted), "rejected": len(failures),
                                      "regime_tasks_sha256": sha(out / "tasks.json")}
    summary["manifest_sha256"] = canonical_hash(summary)
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
