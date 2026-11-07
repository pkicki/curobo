# SPDX-License-Identifier: Apache-2.0
"""Adaptive exact-geometry replay of matched Full-R IPOPT timing profiles.

Runs in the Pinocchio environment. Reconstructs the same original native
cubic geometry and IPOPT squared-speed spline, adding samples around
independently observed extrema. Does not alter frozen base candidates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cohort_geometry import NativeGeometry
from .cohort_retime import states


def _near_times(diag: dict, duration: float, samples: np.ndarray) -> np.ndarray:
    """Independent CPU implementation of extrema-neighborhood sample selection."""
    centers = []
    for family in diag.get("families", {}).values():
        for contract in ("physical", "scaled95"):
            t = family.get(contract, {}).get("time_s")
            if t is not None:
                centers.append(float(t))
    centers.extend(
        float(diag[k])
        for k in ("world_min_time_s", "self_min_time_s")
        if diag.get(k) is not None
    )
    offsets = np.array([-2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0]) * duration / 8192
    extras = [np.clip(t + offsets, 0, duration) for t in centers]
    return (
        np.unique(np.concatenate([samples, *extras])) if extras else np.asarray(samples)
    )


ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "research/joint_time_oracle/results/torque_stress_final_20261107"


def replay(arm: Path, *, case_filter: str | None = None) -> dict:
    folder_root = arm / "native_full"
    diagnostic_file = folder_root / "independent_sampled.json"
    diagnostics = (
        json.loads(diagnostic_file.read_text()) if diagnostic_file.exists() else {}
    )
    tasks = json.loads((arm / "tasks.json").read_text())
    output = {}
    for task in tasks:
        case = task["case_id"]
        if case_filter and case != case_filter:
            continue
        folder = folder_root / case
        folder.mkdir(parents=True, exist_ok=True)
        summary_file = folder / "matched_retime_candidates.json"
        records = json.loads(summary_file.read_text()) if summary_file.exists() else {}
        output[case] = {}
        for name, record in records.items():
            if record.get("status") != "COMPLETE":
                continue
            src = folder / (name.lower() + "_dense.npz")
            diag = diagnostics.get(case + ":" + name)
            if not src.exists() or not isinstance(diag, dict):
                output[case][name] = {"status": "MISSING_SOURCE_OR_DIAGNOSTICS"}
                continue
            try:
                with np.load(src) as z:
                    base = {k: np.asarray(z[k]) for k in z.files}
                if not all(
                    key in base for key in ("time", "progress", "controls", "knots")
                ):
                    raise ValueError("missing IPOPT timing geometry reconstruction")
                native_name = record["source_native_candidate"]
                parent = native_name.split("L", 1)[0]
                with np.load(folder / (parent.lower() + "_buffers.npz")) as z:
                    buffers = {k: np.asarray(z[k]) for k in z.files}
                geometry = NativeGeometry(buffers, int(record["lane"]))
                query = _near_times(diag, float(base["time"][-1]), base["time"])
                progress = np.clip(
                    np.interp(query, base["time"], base["progress"]), 0, 1
                )
                # Preserve the exact original progress coordinates at every
                # base physical-time sample. Interpolating time→progress at
                # those same abscissae can otherwise shift them by ~1e-4.
                base_indices = np.searchsorted(query, base["time"])
                if np.any(base_indices >= len(query)) or not np.array_equal(
                    query[base_indices], base["time"]
                ):
                    raise ValueError(
                        "base physical-time samples missing from adaptive query"
                    )
                is_base_sample = np.zeros(len(query), dtype=bool)
                is_base_sample[base_indices] = True
                adaptive_progress = progress[~is_base_sample]
                # Keep every frozen base progress value exactly. Nearby
                # interpolated adaptive points are redundant at this scale
                # and can cause a sorter to discard a base value.
                insertion = np.searchsorted(base["progress"], adaptive_progress)
                insertion = np.minimum(insertion, len(base["progress"]) - 1)
                distance = np.abs(base["progress"][insertion] - adaptive_progress)
                adaptive_progress = adaptive_progress[distance > 1e-12]
                progress = np.r_[base["progress"], adaptive_progress]
                dense = states(
                    geometry.evaluate_trimmed,
                    base["controls"],
                    knots=base["knots"],
                    extra=progress,
                )
                if np.any(np.diff(dense["time"]) <= 0):
                    raise ValueError("non-increasing retimed physical times")
                # Exact geometry, independently retimed; no geometric interpolation.
                original_grid = base["progress"]
                replay_grid = dense["progress"]
                indices = np.searchsorted(replay_grid, original_grid)
                indices = np.minimum(indices, len(replay_grid) - 1)
                distance = float(np.max(np.abs(replay_grid[indices] - original_grid)))
                if distance > 2e-8:
                    raise ValueError(
                        f"original progress grid not recovered: {distance}"
                    )
                pos_error = float(np.max(np.abs(dense["q"][indices] - base["q"])))
                if pos_error > 1e-6:
                    raise ValueError(
                        f"matched retimed geometry replay diverged: {pos_error}"
                    )
                np.savez_compressed(
                    folder / (name.lower() + "_adaptive_dense.npz"), **dense
                )
                output[case][name] = {
                    "status": "REPLAYED",
                    "samples": len(dense["time"]),
                    "added_samples": len(dense["time"]) - len(base["time"]),
                    "position_replay_error_rad": pos_error,
                }
            except Exception as exc:
                output[case][name] = {"status": "ERROR", "reason": repr(exc)}
        (folder / "matched_adaptive_replay.json").write_text(
            json.dumps(output[case], indent=2) + "\n"
        )
    (folder_root / "matched_adaptive_replay_summary.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT)
    p.add_argument("--regime", required=True, choices=("moderate", "severe"))
    p.add_argument("--case")
    a = p.parse_args()
    result = replay(a.root / a.regime, case_filter=a.case)
    print(
        json.dumps(
            {
                "cases": len(result),
                "replayed": sum(
                    x.get("status") == "REPLAYED"
                    for cases in result.values()
                    for x in cases.values()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
