# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check frozen B/D timing-rerun parity before interpreting IPOPT multipliers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CASES = ("pilot_free_00", "pilot_table_03", "pilot_cluttered_03", "pilot_table_07")


def qualify(historical: Path, rerun: Path) -> dict:
    """Compare rerun duration, exact geometry, convergence, and KKT evidence."""
    result = {"duration_tolerance_s": 1e-9, "geometry_tolerance_rad": 1e-8, "cases": []}
    for case in CASES:
        entry = {"case_id": case, "methods": {}}
        for method in ("b", "d"):
            old_json = json.loads(
                (historical / case / f"{method}_grid513.json").read_text()
            )
            new_json = json.loads((rerun / case / f"{method}_grid513.json").read_text())
            with np.load(historical / case / f"{method}_grid513.npz") as old_data:
                old_q = np.asarray(old_data["q"])
                old_s = np.asarray(old_data["progress"])
            with np.load(rerun / case / f"{method}_grid513.npz") as new_data:
                new_q = np.asarray(new_data["q"])
                new_s = np.asarray(new_data["progress"])
            old_interp = np.stack(
                [
                    np.interp(new_s, old_s, old_q[:, joint])
                    for joint in range(old_q.shape[1])
                ],
                axis=-1,
            )
            geometry_error = float(np.max(np.abs(new_q - old_interp)))
            duration_error = float(abs(new_json["duration_s"] - old_json["duration_s"]))
            item = {
                "historical_duration_s": old_json["duration_s"],
                "rerun_duration_s": new_json["duration_s"],
                "duration_abs_difference_s": duration_error,
                "geometry_max_abs_difference_rad": geometry_error,
                "ipopt_status": new_json.get("status"),
                "solver_converged": new_json.get("solver_converged"),
                "stationarity_inf": new_json.get("stationarity_inf"),
                "primal_min_margin": new_json.get("primal_min_margin"),
                "constraint_multipliers": new_json.get("constraint_multipliers"),
                "refinement_passes": new_json.get("refinement_passes", []),
                "parity_pass": duration_error <= 1e-9 and geometry_error <= 1e-8,
            }
            entry["methods"][method.upper()] = item
        entry["parity_pass"] = all(
            method["parity_pass"] for method in entry["methods"].values()
        )
        result["cases"].append(entry)
    result["all_parity_pass"] = all(row["parity_pass"] for row in result["cases"])
    (rerun / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Compare the separate multiplier reruns with frozen historical outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(qualify(args.historical, args.rerun), indent=2))


if __name__ == "__main__":
    main()
