# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a SHA256 inventory for frozen torque-mechanism study artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "reports/joint_time_oracle/torque_mechanisms/artifact_manifest.json"
)
DEFAULT_INPUTS = (
    ROOT / "research/joint_time_oracle/results/torque_stress_final_20261107",
    ROOT / "research/joint_time_oracle/results/torque_mechanisms_additional_20261108",
    ROOT / "research/joint_time_oracle/results/torque_multiplier_rerun_20261107",
    ROOT
    / "research/joint_time_oracle/results/native_reconstruction_20261106/cuda_derivative_sweep.json",
    ROOT / "reports/joint_time_oracle/torque_mechanisms/index.html",
    ROOT / "reports/joint_time_oracle/torque_mechanisms/summary.json",
    ROOT / "reports/joint_time_oracle/torque_mechanisms/assets",
)


def file_sha256(path: Path) -> str:
    """Hash a file in bounded-memory chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(inputs: tuple[Path, ...]) -> dict:
    """Hash every file under the supplied frozen evidence roots."""
    records = []
    missing = []
    for root in inputs:
        if not root.exists():
            missing.append(str(root.relative_to(ROOT)))
            continue
        paths = (
            [root]
            if root.is_file()
            else sorted(p for p in root.rglob("*") if p.is_file())
        )
        for path in paths:
            records.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return {
        "schema": 1,
        "purpose": "Torque mechanism/full-planner study raw artifact inventory; generated evidence is immutable.",
        "missing_inputs": missing,
        "artifacts": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--input", type=Path, action="append", dest="inputs")
    args = parser.parse_args()
    inputs = tuple(args.inputs) if args.inputs else DEFAULT_INPUTS
    result = inventory(inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_count": len(result["artifacts"]),
                "missing_inputs": result["missing_inputs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
