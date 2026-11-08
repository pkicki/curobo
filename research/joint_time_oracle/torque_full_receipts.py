# SPDX-License-Identifier: Apache-2.0
"""Issue explicit adaptive sampled receipts for every full-planner restart.

No continuous-time guarantee. Every receipt is tied to exact trajectory
bytes, frozen source case, profile SHA and independently evaluated diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .cohort_torque_report import ENDPOINT_TOL, TOLERANCES
from .torque_active import read_profile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "research/joint_time_oracle/results/torque_stress_final_20261107"


def sha(path: Path) -> str:
    """Return the SHA256 digest of a candidate artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_adaptive_candidate(
    arm: Path, case: str, method: str, diagnostics: dict
) -> dict:
    """Validate one saved native trajectory against independent adaptive evidence."""
    source = arm / "native_full" / case
    profile = read_profile(arm / case / "torque_profile.json")
    matched = method.startswith("R")
    recordfile = source / (
        "matched_retime_candidates.json" if matched else "native_candidates.json"
    )
    record = (
        json.loads(recordfile.read_text()).get(method, {})
        if recordfile.exists()
        else {}
    )
    base = source / (method.lower() + "_dense.npz")
    adaptive = source / (method.lower() + "_adaptive_dense.npz")
    diag = diagnostics.get(case + ":" + method + "_adaptive")
    reason = []
    replay_file = source / (
        "matched_adaptive_replay.json" if matched else "native_adaptive_replay.json"
    )
    replay = (
        json.loads(replay_file.read_text()).get(method, {})
        if replay_file.exists()
        else {}
    )
    if replay.get("status") != "REPLAYED":
        reason.append("independent adaptive native replay did not pass")
    if record.get("status") != ("COMPLETE" if matched else "SUCCESS"):
        reason.append("native/matched retiming candidate not successful")
    if not base.exists():
        reason.append("native base trajectory missing")
    if not adaptive.exists():
        reason.append("native adaptive replay missing")
    if not isinstance(diag, dict):
        reason.append("independent adaptive diagnostic missing")
    if reason:
        return {
            "physical": False,
            "scaled95": False,
            "collision": False,
            "endpoint": False,
            "torque_profile_sha256": profile["profile_sha256"],
            "reasons": reason,
        }
    if record.get("torque_profile_sha256") != profile["profile_sha256"]:
        reason.append("profile mismatch with native candidate")
    if record.get("dense_sha256") != sha(base):
        reason.append("native base archive SHA mismatch")
    if diag.get("torque_profile_sha256") != profile["profile_sha256"]:
        reason.append("profile mismatch with independent diagnostic")
    with np.load(base) as data:
        base_times = np.asarray(data["time"], dtype=float)
    with np.load(adaptive) as data:
        t = np.asarray(data["time"], dtype=float)
    if diag.get("samples") != len(t):
        reason.append("adaptive diagnostic sample-count mismatch")
    if len(t) < 8193 or len(t) < len(base_times):
        reason.append("insufficient adaptive samples")
    if len(t) > 1 and np.any(np.diff(t) <= 0):
        reason.append("nonmonotonic adaptive time grid")
    endpoints = {}
    for key, tolerance in ENDPOINT_TOL.items():
        val = diag.get("endpoints", {}).get(key)
        endpoints[key] = (
            isinstance(val, (int, float)) and np.isfinite(val) and abs(val) <= tolerance
        )
    endpoint = all(endpoints.values()) and not reason
    world = diag.get("world_min_clearance_m")
    self_distance = diag.get("self_min_clearance_m")
    collision = (
        (world is None or world >= -1e-7)
        and (self_distance is not None and self_distance >= -1e-7)
        and not reason
    )
    families = {"physical": {}, "scaled95": {}}
    for family, tolerance in TOLERANCES.items():
        for name in families:
            value = (
                diag.get("families", {}).get(family, {}).get(name, {}).get("violation")
            )
            families[name][family] = (
                isinstance(value, (int, float))
                and np.isfinite(value)
                and value <= tolerance
            )
    physical = bool(endpoint and collision and all(families["physical"].values()))
    scaled95 = bool(physical and all(families["scaled95"].values()))
    return {
        "physical": physical,
        "scaled95": scaled95,
        "collision": bool(collision),
        "endpoint": bool(endpoint),
        "torque_profile_sha256": profile["profile_sha256"],
        "base_trajectory_sha256": sha(base),
        "adaptive_trajectory_sha256": sha(adaptive),
        "independent_adaptive_samples": len(t),
        "added_samples": max(0, len(t) - len(base_times)),
        "family_checks": families,
        "endpoint_checks": endpoints,
        "reasons": reason,
        "scope": "dense and adaptive independently sampled; not a continuous-time certificate",
    }


def generate(arm: Path) -> dict:
    """Issue receipts for every saved native full-planner candidate."""
    native = arm / "native_full"
    file = native / "independent_adaptive.json"
    diagnostics = json.loads(file.read_text()) if file.exists() else {}
    tasks = json.loads((arm / "tasks.json").read_text())
    out = {}
    for task in tasks:
        case = task["case_id"]
        candidates = native / case / "native_candidates.json"
        rows = json.loads(candidates.read_text()) if candidates.exists() else {}
        matched = native / case / "matched_retime_candidates.json"
        if matched.exists():
            rows.update(json.loads(matched.read_text()))
        for method in sorted(rows):
            out[case + ":" + method] = validate_adaptive_candidate(
                arm, case, method, diagnostics
            )
    (native / "adaptive_receipts.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def main():
    """Generate physical and Scaled95 receipt summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--regime", required=True)
    a = parser.parse_args()
    out = generate(a.root / a.regime)
    print(
        json.dumps(
            {
                "total": len(out),
                "physical_pass": sum(x.get("physical") is True for x in out.values()),
                "adaptive_receipts": str(
                    a.root / a.regime / "native_full" / "adaptive_receipts.json"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
