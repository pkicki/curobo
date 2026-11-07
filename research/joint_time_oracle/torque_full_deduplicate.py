# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hard-link byte-identical full-planner NPZ aliases without changing hashes."""

from __future__ import annotations

import argparse
import hashlib
import os
from collections import defaultdict
from pathlib import Path

SUFFIXES = (
    "_dense.npz",
    "_adaptive_dense.npz",
    "_pin_torque.npz",
    "_adaptive_pin_torque.npz",
)


def deduplicate(root: Path, *, apply: bool) -> dict[str, int]:
    """Link only exact-byte duplicates within a task directory.

    Existing candidate paths and SHA256 values remain intact; aliases share
    disk blocks only when the complete compressed archive bytes match.
    """
    saved_bytes = 0
    duplicates = 0
    for task_dir in root.iterdir():
        if not task_dir.is_dir():
            continue
        for suffix in SUFFIXES:
            groups: dict[str, list[Path]] = defaultdict(list)
            for path in task_dir.glob(f"*{suffix}"):
                if not (path.name.startswith("f") or path.name.startswith("rf")):
                    continue
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                groups[digest].append(path)
            for paths in groups.values():
                if len(paths) < 2:
                    continue
                canonical = paths[0]
                for duplicate in paths[1:]:
                    if os.path.samefile(canonical, duplicate):
                        continue
                    duplicates += 1
                    saved_bytes += duplicate.stat().st_size
                    if apply:
                        duplicate.unlink()
                        os.link(canonical, duplicate)
    return {"duplicate_paths": duplicates, "potential_bytes_saved": saved_bytes}


def main() -> None:
    """Summarize duplicates or replace them with exact hard links."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(deduplicate(args.root, apply=args.apply))


if __name__ == "__main__":
    main()
