# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure exact cubic-path diversity across torque-stressed native restarts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .cohort_geometry import NativeGeometry
from .torque_active import read_profile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "research/joint_time_oracle/results/torque_stress_final_20261107"


def _geometry_hash(buffers: dict[str, np.ndarray], lane: int) -> str:
    digest = hashlib.sha256()
    for key in ("start_boundary", "controls", "goal"):
        value = np.ascontiguousarray(buffers[key][lane])
        digest.update(key.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _arc_length_path(
    geometry: NativeGeometry, samples: int = 2001
) -> tuple[np.ndarray, float]:
    """Return path positions at uniform normalized joint-space arc length."""
    geometry.nonstationary_interval()
    progress = np.linspace(0.0, 1.0, samples)
    q = np.asarray(geometry.evaluate_trimmed(progress)[0])
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    length = float(cumulative[-1])
    if not np.isfinite(length) or length <= 1e-10:
        raise ValueError("native path is stationary or invalid")
    unique_length, indices = np.unique(cumulative, return_index=True)
    target = np.linspace(0.0, length, 201)
    aligned = np.column_stack(
        [
            np.interp(target, unique_length, q[indices, joint])
            for joint in range(q.shape[1])
        ]
    )
    return aligned, length


def evaluate_case(arm: Path, case: str, *, tolerance_rad: float = 1e-4) -> dict:
    """Backfill exact control hashes and calculate pairwise path distances."""
    folder = arm / "native_full" / case
    candidates_file = folder / "native_candidates.json"
    candidates = (
        json.loads(candidates_file.read_text()) if candidates_file.exists() else {}
    )
    profile = read_profile(arm / case / "torque_profile.json")
    path_cache: dict[tuple[str, int], tuple[np.ndarray, float]] = {}
    records = []
    for name, record in sorted(candidates.items()):
        if record.get("status") != "SUCCESS":
            continue
        parent = str(record.get("parent_restart_tag", name))
        if not parent.startswith("F") or len(parent) != 4:
            continue
        buffer_path = folder / (parent.lower() + "_buffers.npz")
        with np.load(buffer_path) as archive:
            buffers = {key: np.asarray(archive[key]) for key in archive.files}
        lane = int(record["lane"])
        exact_hash = _geometry_hash(buffers, lane)
        path_key = (parent, lane)
        if path_key not in path_cache:
            path_cache[path_key] = _arc_length_path(NativeGeometry(buffers, lane))
        path, length = path_cache[path_key]
        record["geometry_sha256"] = exact_hash
        records.append(
            {
                "candidate": name,
                "restart": int(record.get("restart", 0)),
                "lane": lane,
                "parent_restart_tag": parent,
                "duration_s": record.get("duration_s"),
                "geometry_sha256": exact_hash,
                "joint_path_length_rad": length,
                "arc_aligned_path": path,
            }
        )
    candidates_file.write_text(json.dumps(candidates, indent=2) + "\n")

    pairwise = []
    for i, left in enumerate(records):
        for right in records[i + 1 :]:
            delta = left["arc_aligned_path"] - right["arc_aligned_path"]
            pairwise.append(
                {
                    "left": left["candidate"],
                    "right": right["candidate"],
                    "rms_deviation_rad": float(np.sqrt(np.mean(delta * delta))),
                    "max_deviation_rad": float(np.max(np.abs(delta))),
                    "same_geometry_hash": left["geometry_sha256"]
                    == right["geometry_sha256"],
                }
            )
    clusters: list[list[dict]] = []
    for candidate in records:
        for cluster in clusters:
            representative = cluster[0]
            delta = candidate["arc_aligned_path"] - representative["arc_aligned_path"]
            if float(np.max(np.abs(delta))) <= tolerance_rad:
                cluster.append(candidate)
                break
        else:
            clusters.append([candidate])
    output = {
        "case_id": case,
        "regime": profile["regime"],
        "torque_profile_sha256": profile["profile_sha256"],
        "native_candidate_count": len(records),
        "independent_restart_count": len({row["restart"] for row in records}),
        "exact_geometry_count": len({row["geometry_sha256"] for row in records}),
        "arc_aligned_unique_geometry_count": len(clusters),
        "geometric_diversity_tolerance_rad": tolerance_rad,
        "candidates": [
            {key: value for key, value in row.items() if key != "arc_aligned_path"}
            for row in records
        ],
        "pairwise_deviation": pairwise,
        "clusters": [[row["candidate"] for row in cluster] for cluster in clusters],
        "metric": (
            "RMS/max joint-position deviation after uniform joint-space "
            "arc-length alignment"
        ),
    }
    output["task_sha256"] = next(
        (
            task["task_sha256"]
            for task in json.loads((arm / "tasks.json").read_text())
            if task["case_id"] == case
        ),
        None,
    )
    target = folder / "geometry_diversity.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    return output


def main() -> None:
    """Measure geometric diversity in frozen native candidate paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--regime", choices=("moderate", "severe"), required=True)
    parser.add_argument("--cases", help="comma-separated IDs, otherwise all tasks")
    parser.add_argument("--tolerance-rad", type=float, default=1e-4)
    args = parser.parse_args()
    arm = args.root / args.regime
    tasks = json.loads((arm / "tasks.json").read_text())
    selected = set(args.cases.split(",")) if args.cases else None
    results = []
    for task in tasks:
        if selected is not None and task["case_id"] not in selected:
            continue
        results.append(
            evaluate_case(arm, task["case_id"], tolerance_rad=args.tolerance_rad)
        )
    print(
        json.dumps(
            {
                "cases": len(results),
                "native_candidates": sum(
                    row["native_candidate_count"] for row in results
                ),
                "output_root": str(arm / "native_full"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
