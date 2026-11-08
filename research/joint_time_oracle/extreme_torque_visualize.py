# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render an evidence-linked qualitative view of one extreme-torque case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .torque_active import torque_utilization

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _draw_scene(ax, scene_path: Path | None) -> None:
    if scene_path is None or not scene_path.exists():
        return
    scene = yaml.safe_load(scene_path.read_text()) or {}
    for item in (scene.get("cuboid") or {}).values():
        dims = np.asarray(item["dims"], dtype=float)
        center = np.asarray(item["pose"][:3], dtype=float)
        lo, hi = center - dims / 2, center + dims / 2
        vertices = np.array(
            [
                [x, y, z]
                for x in (lo[0], hi[0])
                for y in (lo[1], hi[1])
                for z in (lo[2], hi[2])
            ]
        )
        faces = [
            [vertices[j] for j in face]
            for face in (
                [0, 1, 3, 2],
                [4, 5, 7, 6],
                [0, 1, 5, 4],
                [2, 3, 7, 6],
                [0, 2, 6, 4],
                [1, 3, 7, 5],
            )
        ]
        ax.add_collection3d(
            Poly3DCollection(faces, alpha=0.16, facecolor="gray", edgecolor="gray")
        )
    for item in (scene.get("sphere") or {}).values():
        p = np.asarray(
            item.get("position", item.get("pose", [0, 0, 0])[:3]), dtype=float
        )
        radius = float(item["radius"])
        ax.scatter(*p, s=max(10, radius * 180), color="gray", alpha=0.35)


def render(root: Path, source: Path, regime: str, case: str, output: Path) -> dict:
    """Plot exact frozen end-effector and Pinocchio torque audits."""
    arm = root / regime
    folder = arm / case
    native = arm / "native_full" / case
    witness = _load(folder / "w_adaptive_dense.npz")
    witness_tau = _load(folder / "w_adaptive_pin_torque.npz")
    native_dense = _load(native / "f000_adaptive_dense.npz")
    native_tau = _load(native / "f000_adaptive_pin_torque.npz")
    original_dir = source.parent.parent / "cohort_pilot_isolated_20261106" / case
    original = _load(original_dir / "a10_dense.npz")
    original_ee = _load(folder / "original_geometry_ee.npz")
    original_audit = _load(original_dir / "a10_pin_torque.npz")
    profile = json.loads((folder / "torque_profile.json").read_text())
    lp = json.loads((folder / "path_torque_lp.json").read_text())
    candidate = json.loads((native / "native_candidates.json").read_text())["F000"]
    wmeta = json.loads((folder / "w_source.json").read_text())
    task = next(
        x for x in json.loads((arm / "tasks.json").read_text()) if x["case_id"] == case
    )
    scene = (
        ROOT / "curobo/content/configs/scene" / task["scene"]
        if task.get("scene")
        else None
    )

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax3 = fig.add_subplot(grid[0, 0], projection="3d")
    axq = fig.add_subplot(grid[0, 1])
    axt = fig.add_subplot(grid[1, :])
    for ee, label, style in (
        (original_ee["end_effector"], "Original cuRobo geometry", "--"),
        (witness_tau["end_effector"], "Feasible D witness geometry", "-"),
        (native_tau["end_effector"], "Fresh full-planner geometry", ":"),
    ):
        ax3.plot(ee[:, 0], ee[:, 1], ee[:, 2], style, linewidth=2, label=label)
    _draw_scene(ax3, scene)
    ax3.scatter(*original_ee["end_effector"][0], color="green", s=35, label="start")
    ax3.scatter(*original_ee["end_effector"][-1], color="black", s=35, label="goal")
    ax3.set(
        xlabel="x (m)",
        ylabel="y (m)",
        zlabel="z (m)",
        title="End-effector geometry and configured obstacles",
    )
    ax3.legend(fontsize=8)

    for joint in range(7):
        axq.plot(
            original["time"],
            original["q"][:, joint],
            "--",
            alpha=0.55,
            label="original" if joint == 0 else None,
        )
        axq.plot(
            witness["time"],
            witness["q"][:, joint],
            linewidth=1,
            label="witness" if joint == 0 else None,
        )
        axq.plot(
            native_dense["time"],
            native_dense["q"][:, joint],
            ":",
            alpha=0.8,
            label="fresh native" if joint == 0 else None,
        )
    axq.set(
        xlabel="physical time (s)",
        ylabel="joint position (rad)",
        title="Joint positions",
    )
    axq.legend(fontsize=8)

    lo, hi = np.asarray(profile["effort_lower"]), np.asarray(profile["effort_upper"])
    for joint in range(7):
        color = f"C{joint}"
        axt.plot(
            original_audit["time"],
            original_audit["torque"][:, joint],
            color=color,
            linestyle="--",
            alpha=0.7,
        )
        axt.plot(
            witness_tau["time"],
            witness_tau["torque"][:, joint],
            color=color,
            linestyle="-",
        )
        axt.plot(
            native_tau["time"],
            native_tau["torque"][:, joint],
            color=color,
            linestyle=":",
            alpha=0.8,
        )
        axt.axhline(hi[joint], color=color, linewidth=0.6, alpha=0.3)
        axt.axhline(lo[joint], color=color, linewidth=0.6, alpha=0.3)
        axt.axhline(
            profile["nominal_effort_upper"][joint],
            color="0.55",
            linewidth=0.45,
            linestyle="--",
            alpha=0.35,
        )
        axt.axhline(
            profile["nominal_effort_lower"][joint],
            color="0.55",
            linewidth=0.45,
            linestyle="--",
            alpha=0.35,
        )
    for joint in range(7):
        axt.axhline(hi[joint], color=f"C{joint}", linewidth=0.6, alpha=0.45)
        axt.axhline(lo[joint], color=f"C{joint}", linewidth=0.6, alpha=0.45)
    axt.set(
        xlabel="physical time (s)",
        ylabel="Pinocchio torque (Nm)",
        title=(
            "Independent torque: solid W, dotted fresh native; "
            "horizontal bounds use joint color"
        ),
    )
    method_legend = axt.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                label="original geometry / native timing",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="-",
                label="D geometry / dilated timing",
            ),
            Line2D([0], [0], color="black", linestyle=":", label="fresh full planner"),
            Line2D(
                [0], [0], color="0.55", linestyle="--", label="nominal effort limits"
            ),
        ],
        loc="upper left",
        fontsize=8,
    )
    axt.add_artist(method_legend)
    axt.legend(
        handles=[
            Line2D([0], [0], color=f"C{j}", label=f"joint {j+1}") for j in range(7)
        ],
        loc="upper right",
        ncol=7,
        fontsize=7,
    )

    receipt = json.loads((arm / "w_adaptive_receipts.json").read_text()).get(
        case + ":W", {}
    )
    native_receipt_file = arm / "native_full" / "adaptive_receipts.json"
    native_receipts = (
        json.loads(native_receipt_file.read_text())
        if native_receipt_file.exists()
        else {}
    )
    native_physical = sum(
        1
        for key, value in native_receipts.items()
        if key.startswith(case + ":") and value.get("physical") is True
    )
    witness_validation = json.loads(
        (arm / "independent_adaptive.json").read_text()
    ).get(case + ":W_adaptive", {})
    native_validation = json.loads(
        (arm / "native_full/independent_adaptive.json").read_text()
    ).get(case + ":F000_adaptive", {})
    w_tau_util = float(
        np.max(torque_utilization(witness_tau["torque"], np.stack((lo, hi))))
    )
    n_tau_util = float(
        np.max(torque_utilization(native_tau["torque"], np.stack((lo, hi))))
    )
    clearance_w = (
        f"{witness_validation.get('world_min_clearance_m')}/"
        f"{witness_validation.get('self_min_clearance_m')}m"
    )
    clearance_n = (
        f"{native_validation.get('world_min_clearance_m')}/"
        f"{native_validation.get('self_min_clearance_m')}m"
    )
    lp_expansion = lp["original_fixed_curobo"].get(
        "max_required_torque_bound_expansion"
    )
    title = (
        f"{case} · {regime}\n"
        f"D witness: rho={wmeta.get('witness_slowdown')}×, "
        f"T={wmeta.get('witness_duration_s'):.4f}s; "
        f"fresh native T={candidate['duration_s']:.4f}s\n"
        f"W physical/Scaled95={receipt.get('physical')}/{receipt.get('scaled95')}; "
        f"native qualified lanes={native_physical}; "
        f"torque utilization W/native={w_tau_util:.3f}/{n_tau_util:.3f}\n"
        f"clearance world/self W={clearance_w}, native={clearance_n}; "
        f"original-path LP expansion={lp_expansion} Nm (zero is inconclusive)"
    )
    fig.suptitle(title, fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return {
        "regime": regime,
        "case_id": case,
        "output": str(output),
        "task_sha256": task["task_sha256"],
        "torque_profile_sha256": profile["profile_sha256"],
        "native_candidate_geometry_sha256": candidate.get("geometry_sha256"),
        "witness_duration_s": wmeta.get("witness_duration_s"),
        "witness_slowdown": wmeta.get("witness_slowdown"),
        "native_duration_s": candidate.get("duration_s"),
        "witness_receipt": receipt,
        "original_path_lp": lp["original_fixed_curobo"],
    }


def main() -> None:
    """Parse rendering options and write the requested evidence figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--regime", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            render(args.root, args.source, args.regime, args.case, args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
