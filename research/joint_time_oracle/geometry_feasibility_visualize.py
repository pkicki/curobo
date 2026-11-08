# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot a frozen original-path obstruction and its qualified alternative."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin

from .cohort_geometry import QuinticGeometry
from .geometry_feasibility_discover import source_native, source_tasks
from .geometry_feasibility_scene import IndependentGeometryAuditor


def _frame_points(auditor: IndependentGeometryAuditor, q: np.ndarray) -> np.ndarray:
    """Return link-origin positions for a single seven-joint configuration."""
    auditor.q[auditor.qi] = q
    pin.framesForwardKinematics(auditor.model, auditor.data, auditor.q)
    names = [f"panda_link{i}" for i in range(1, 8)] + ["panda_hand"]
    return np.asarray(
        [
            auditor.data.oMf[auditor.model.getFrameId(n, pin.BODY)].translation
            for n in names
        ]
    )


def create_figure(
    root: Path, source: Path, scenario: str, case: str, output: Path
) -> Path:
    """Create path, robot-pose, and all-joint torque panels from frozen data."""
    source_tasks_list, rows = source_tasks(source)
    task = next(item for item in source_tasks_list if item["case_id"] == case)
    arm = root / scenario
    folder = arm / case
    native = source_native(source, task, rows[case])
    lp = json.loads((folder / "refined_original_path_lp.json").read_text())
    receipt = json.loads((arm / "w_adaptive_receipts.json").read_text())[case + ":W"]
    geometry_data = json.loads((folder / "w_geometry.json").read_text())
    alternative = QuinticGeometry(np.asarray(geometry_data["controls"], dtype=float))
    progress = float(lp["worst_progress"])
    original_point = np.asarray(native.evaluate_trimmed(np.array([progress]))[0])[0]
    alt_s = np.linspace(0.0, 1.0, 4097)
    alt_q = np.asarray(alternative.evaluate(alt_s)[0])
    original_s = np.linspace(0.0, 1.0, 4097)
    original_q = np.asarray(native.evaluate_trimmed(original_s)[0])

    def normalized_arc(q: np.ndarray) -> np.ndarray:
        lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        return lengths / lengths[-1]

    original_arc = normalized_arc(original_q)
    marked_arc = float(np.interp(progress, original_s, original_arc))
    alternative_mark = np.asarray(
        [np.interp(marked_arc, normalized_arc(alt_q), alt_q[:, j]) for j in range(7)]
    )

    with np.load(source / case / "a10_pin_torque.npz") as data:
        base_time = np.asarray(data["time"], dtype=float)
        base_torque = np.asarray(data["torque"], dtype=float)
        base_ee = np.asarray(data["end_effector"], dtype=float)
    with np.load(folder / "w_pin_torque.npz") as data:
        witness_time = np.asarray(data["time"], dtype=float)
        witness_torque = np.asarray(data["torque"], dtype=float)
        witness_ee = np.asarray(data["end_effector"], dtype=float)
    profile = json.loads((folder / "torque_profile.json").read_text())
    lower = np.asarray(profile["effort_lower"], dtype=float)
    upper = np.asarray(profile["effort_upper"], dtype=float)
    auditor = IndependentGeometryAuditor(task["scene"])
    original_pose = _frame_points(auditor, original_point)
    alternative_pose = _frame_points(auditor, alternative_mark)
    auditor.q[auditor.qi] = original_point
    pin.framesForwardKinematics(auditor.model, auditor.data, auditor.q)
    obstruction_ee = auditor.data.oMf[
        auditor.model.getFrameId("panda_hand", pin.BODY)
    ].translation.copy()

    figure = plt.figure(figsize=(17, 10), constrained_layout=True)
    layout = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.2))
    ax_path = figure.add_subplot(layout[0, 0], projection="3d")
    ax_robot = figure.add_subplot(layout[0, 1], projection="3d")
    ax_tau = figure.add_subplot(layout[1, :])
    ax_path.plot(
        *base_ee.T, color="#586575", lw=1.4, label="Original native path (A10)"
    )
    ax_path.plot(
        *witness_ee.T, color="#087e8b", lw=2.0, label="Feasible quintic alternative"
    )
    ax_path.scatter(*base_ee[0], color="black", s=25, label="Start")
    ax_path.scatter(*base_ee[-1], color="black", s=25, marker="x", label="Goal")
    ax_path.scatter(
        *obstruction_ee,
        color="#d1495b",
        marker="*",
        s=100,
        label="Original-path LP point",
    )
    ax_path.set_title("End-effector paths in the frozen scene")
    ax_path.set_xlabel("x (m)")
    ax_path.set_ylabel("y (m)")
    ax_path.set_zlabel("z (m)")
    ax_path.legend(loc="upper left", fontsize=8)

    for pose, color, label in (
        (original_pose, "#d1495b", f"Original at LP obstruction s={progress:.3f}"),
        (alternative_pose, "#087e8b", "Alternative at matched arc progress"),
    ):
        ax_robot.plot(*pose.T, "o-", color=color, lw=2, ms=4, label=label)
    for name, center, rotation, half in auditor.cuboids:
        signs = np.asarray(list(product((-1.0, 1.0), repeat=3)))
        corners = signs * half
        corners = rotation.apply(corners) + center
        for i, corner in enumerate(corners):
            for j in range(i + 1, len(corners)):
                if np.count_nonzero(signs[i] != signs[j]) == 1:
                    ax_robot.plot(
                        *np.stack((corner, corners[j])).T,
                        color="#9c755f",
                        lw=0.8,
                        alpha=0.7,
                    )
        ax_robot.text(*center, name, fontsize=7)
    for name, center, radius in auditor.spheres:
        ax_robot.scatter(*center, s=max(20, radius * 700), color="#9c755f", alpha=0.5)
        ax_robot.text(*center, name, fontsize=7)
    ax_robot.set_title("Robot configurations at the geometric divergence")
    ax_robot.set_xlabel("x (m)")
    ax_robot.set_ylabel("y (m)")
    ax_robot.set_zlabel("z (m)")
    ax_robot.legend(loc="upper left", fontsize=8)

    for joint in range(7):
        color = plt.cm.tab10(joint)
        ax_tau.plot(
            base_time,
            base_torque[:, joint],
            color=color,
            alpha=0.65,
            lw=0.9,
            label=f"A10 J{joint + 1}",
        )
        ax_tau.plot(
            witness_time,
            witness_torque[:, joint],
            color=color,
            lw=1.5,
            ls="--",
            label=f"Alternative J{joint + 1}",
        )
        ax_tau.axhline(lower[joint], color=color, lw=0.6, ls=":")
        ax_tau.axhline(upper[joint], color=color, lw=0.6, ls=":")
    certificate = lp["numerical_dual_certificate"]
    ax_tau.set_title(
        "Independent Pinocchio torque: original native timing vs. feasible alternative\n"
        f"Derated asymmetric bounds shown dotted; original-path LP dual lower bound "
        f"ε={certificate['dual_lower_bound']:.4f}; alternative T={witness_time[-1]:.2f} s"
    )
    ax_tau.set_xlabel("Physical time (s)")
    ax_tau.set_ylabel("Joint torque (Nm)")
    ax_tau.grid(alpha=0.2)
    ax_tau.legend(ncol=7, fontsize=7, loc="upper center")
    figure.suptitle(
        f"{case} / {scenario} — source and witness checksummed; "
        f"sampled physical + Scaled95 receipt: {receipt['physical']} / {receipt['scaled95']}",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def main() -> None:
    """Render a frozen, qualified original-path obstruction example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(create_figure(args.root, args.source, args.scenario, args.case, args.output))


if __name__ == "__main__":
    main()
