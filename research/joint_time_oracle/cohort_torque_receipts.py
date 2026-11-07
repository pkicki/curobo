# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create explicit sampled adaptive-verification receipts for torque cohorts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cohort_torque_report import ENDPOINT_TOL, TOLERANCES
from .torque_active import read_profile


def receipts(arm: Path, output: Path) -> dict:
    """Pair exact replay/audit artifacts by source and torque-profile checksum."""
    diagnostic_path = arm / "independent_adaptive.json"
    diagnostics = json.loads(diagnostic_path.read_text())
    tasks = json.loads((arm / "tasks.json").read_text())
    out: dict[str, dict] = {}
    for task in tasks:
        case = task["case_id"]
        folder = arm / case
        profile = read_profile(folder / "torque_profile.json")
        for method in ("A10", "A40", "B", "C", "D"):
            diagnostic = diagnostics.get(f"{case}:{method}_adaptive")
            trajectory = folder / f"{method.lower()}_adaptive_dense.npz"
            if diagnostic is None or not trajectory.exists():
                continue
            with np.load(trajectory) as data:
                samples = int(len(data["time"]))
            checks: dict[str, bool] = {}
            for family, tolerance in ENDPOINT_TOL.items():
                value = diagnostic.get("endpoints", {}).get(family)
                checks[family] = (isinstance(value, (float, int)) and
                                  np.isfinite(value) and abs(value) <= tolerance)
            endpoint = bool(all(checks.values()))
            clearance_world = diagnostic.get("world_min_clearance_m")
            clearance_self = diagnostic.get("self_min_clearance_m")
            collision = ((clearance_world is None or clearance_world >= -1e-7) and
                         clearance_self is not None and clearance_self >= -1e-7)
            family_checks: dict[str, dict[str, bool]] = {"physical": {}, "scaled95": {}}
            for family, tolerance in TOLERANCES.items():
                for contract in ("physical", "scaled95"):
                    residual = diagnostic.get("families", {}).get(family, {}).get(contract, {}).get("violation")
                    family_checks[contract][family] = (residual is not None and
                        np.isfinite(residual) and residual <= tolerance)
            checksum_ok = diagnostic.get("torque_profile_sha256") == profile["profile_sha256"]
            enough_samples = samples >= 8193
            physical = bool(checksum_ok and enough_samples and endpoint and collision and
                            all(family_checks["physical"].values()))
            scaled = bool(physical and all(family_checks["scaled95"].values()))
            with np.load(folder / f"{method.lower()}_dense.npz") as base:
                base_samples = int(len(base["time"]))
            out[f"{case}:{method}"] = {
                "physical": physical,
                "scaled95": scaled,
                "collision": bool(collision and checksum_ok and enough_samples),
                "endpoint": bool(endpoint and checksum_ok and enough_samples),
                "torque_profile_sha256": profile["profile_sha256"],
                "adaptive_samples": samples,
                "base_samples": base_samples,
                "added_samples": max(0, samples - base_samples),
                "adaptive_refinement_passes": 1,
                "family_checks": family_checks,
                "endpoints": checks,
                "world_clearance_m": clearance_world,
                "self_clearance_m": clearance_self,
                "validation_contract": "sampled exact-curve replay with added neighborhoods around independently observed extrema; not a continuous-time certificate",
            }
    output.write_text(json.dumps(out, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    target = args.output or args.arm / "adaptive_receipts.json"
    result = receipts(args.arm, target)
    print(target, "receipts", len(result), "physical", sum(x["physical"] for x in result.values()),
          "Scaled95", sum(x["scaled95"] for x in result.values()))


if __name__ == "__main__":
    main()
