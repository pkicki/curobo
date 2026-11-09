# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a reproducible, offline report for fixed-geometry margin recovery."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline

from .cohort_retime import ROOT
from .geometry_feasibility_core import sha256_file
from .geometry_feasibility_manifest import generate as generate_manifest
from .torque_margin_core import CASE_IDS, TIGHTENING_FACTORS, family_bounds
from .torque_margin_prepare import OUTPUT, PREVIOUS
from .torque_margin_report import arms, collect

REPORT = ROOT / "reports/joint_time_oracle/torque_margin_retiming"
SOURCE_ARM = "no_lp"
NATIVE_PROBE = (
    ROOT
    / "research/joint_time_oracle/results/native_timeopt_phase_probe_20261109/j1234_f0300/native_full"
)
NATIVE_CONTROL = (
    ROOT
    / "research/joint_time_oracle/results/native_timeopt_phase_control_20261109/j1234_f0300/native_full"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _limits() -> dict[str, np.ndarray]:
    with np.load(
        ROOT
        / "research/joint_time_oracle/results/native_reconstruction_20261106/baseline_validation_states.npz"
    ) as archive:
        return {
            name: np.asarray(archive[name + "_bounds"], dtype=float)
            for name in ("position", "velocity", "acceleration", "jerk")
        }


def _utilization(
    values: np.ndarray, bounds: np.ndarray, *, torque: bool = False
) -> np.ndarray:
    low, high = np.asarray(bounds, dtype=float)
    values = np.asarray(values, dtype=float)
    if torque:
        result = np.where(
            values >= 0,
            values / np.maximum(high, 1e-12),
            values / np.minimum(low, -1e-12),
        )
    else:
        center = (low + high) / 2
        half = np.maximum((high - low) / 2, 1e-12)
        result = np.abs((values - center) / half)
    return np.maximum(result, 0)


def _peak_metrics(
    values: np.ndarray, bounds: np.ndarray, times: np.ndarray, *, torque: bool = False
) -> dict:
    utilization = _utilization(values, bounds, torque=torque)
    sample, joint = np.unravel_index(int(np.argmax(utilization)), utilization.shape)
    value = float(values[sample, joint])
    low, high = (float(bounds[0, joint]), float(bounds[1, joint]))
    active_limit = high if value >= 0 or not torque else low
    return {
        "max_utilization": float(utilization[sample, joint]),
        "joint_index_zero_based": int(joint),
        "time_s": float(times[sample]),
        "observed_value": value,
        "active_limit": active_limit,
        "sample_index": int(sample),
    }


def _task_row(root: Path, case: str, arm: str, source: dict, limits: dict) -> dict:
    folder = root / arm / case
    screen = _read_json(folder / "screen_result.json")
    receipt = _read_json(root / arm / "r_adaptive_receipts.json").get(case + ":R", {})
    sampled = _read_json(root / arm / "independent_sampled.json").get(case + ":R", {})
    adaptive = _read_json(root / arm / "independent_adaptive.json").get(
        case + ":R_adaptive", {}
    )
    registration = (
        _read_json(folder / "r_registration.json")
        if (folder / "r_registration.json").exists()
        else {}
    )
    if registration:
        expected_files = {
            "r_dense.npz": registration.get("r_sha256"),
            "r_geometry.json": registration.get("r_geometry_sha256"),
        }
        for filename, expected in expected_files.items():
            if not expected or sha256_file(folder / filename) != expected:
                raise ValueError(
                    f"R artifact checksum mismatch: {case}/{arm}/{filename}"
                )
        profile_document = _read_json(folder / "torque_profile.json")
        if profile_document.get("profile_sha256") != registration.get("profile_sha256"):
            raise ValueError(f"torque-profile checksum mismatch: {case}/{arm}")
        if receipt.get("r_sha256") != registration.get("r_sha256"):
            raise ValueError(f"qualification receipt references stale R: {case}/{arm}")
        if receipt.get("r_geometry_sha256") != registration.get("r_geometry_sha256"):
            raise ValueError(
                f"qualification receipt references stale geometry: {case}/{arm}"
            )
        if receipt.get("profile_sha256") != registration.get("profile_sha256"):
            raise ValueError(
                f"qualification receipt references stale profile: {case}/{arm}"
            )
        adaptive_path = folder / "r_adaptive_dense.npz"
        adaptive_replay = _read_json(folder / "r_adaptive_replay.json")
        controls_receipt = _read_json(folder / "r_controls_receipt.json")
        if sha256_file(adaptive_path) != receipt.get("r_adaptive_sha256"):
            raise ValueError(f"adaptive trajectory checksum mismatch: {case}/{arm}")
        if adaptive_replay.get("r_adaptive_sha256") != sha256_file(adaptive_path):
            raise ValueError(f"adaptive replay receipt is stale: {case}/{arm}")
        if adaptive_replay.get("source_r_sha256") != registration.get("r_sha256"):
            raise ValueError(f"adaptive replay uses stale source R: {case}/{arm}")
        if controls_receipt.get("r_dense_sha256") != registration.get("r_sha256"):
            raise ValueError(f"control receipt uses stale source R: {case}/{arm}")
        if controls_receipt.get("r_speed_controls_sha256") != sha256_file(
            folder / "r_speed_controls.npz"
        ):
            raise ValueError(f"squared-speed control receipt is stale: {case}/{arm}")
    result = {
        "case_id": case,
        "arm": arm,
        "W_duration_s": source["source_W_duration_s"],
        "P_duration_s": source["source_P_duration_s"],
        "R_duration_s": registration.get("r_duration_s"),
        "improvement_vs_W_pct": None,
        "solver": (
            "analytic finite dilation grid"
            if arm == "dilation"
            else "IPOPT local retiming"
        ),
        "solver_status": (
            "SCREENED"
            if arm == "dilation"
            else screen.get("solver", {}).get("message", screen.get("reason"))
        ),
        "solver_converged": (
            None
            if arm == "dilation"
            else screen.get("solver", {}).get("solver_converged")
        ),
        "iterations": (
            None if arm == "dilation" else screen.get("solver", {}).get("iterations")
        ),
        "solve_or_screen_wall_s": (
            screen.get("total_cpu_wall_s")
            if arm == "dilation"
            else screen.get("solver", {}).get("elapsed_s")
        ),
        "ipopt_solve_wall_s": (
            None if arm == "dilation" else screen.get("solver", {}).get("elapsed_s")
        ),
        "dilation_total_wall_s": (
            screen.get("total_cpu_wall_s") if arm == "dilation" else None
        ),
        "dilation_rnea_wall_s": (
            screen.get("baseline_rnea_wall_s") if arm == "dilation" else None
        ),
        "dilation_screen_wall_s": (
            screen.get("screen_wall_s") if arm == "dilation" else None
        ),
        "rho": screen.get("selected_rho"),
        "beta": screen.get("beta"),
        "physical": receipt.get("physical") is True,
        "scaled95": receipt.get("scaled95") is True,
        "physical_samples_dense": receipt.get("base_samples"),
        "physical_samples_adaptive": receipt.get("adaptive_samples"),
        "receipt_reasons": receipt.get("reason", []),
        "trajectory_sha256": registration.get("r_sha256"),
        "adaptive_sha256": receipt.get("r_adaptive_sha256"),
        "speed_control_sha256": (folder / "r_speed_controls.npz").is_file()
        and sha256_file(folder / "r_speed_controls.npz"),
        "geometry_sha256": registration.get("r_geometry_sha256"),
        "source_P_controls_sha256": registration.get("source_files_sha256", {}).get(
            "p_timing_controls.npz"
        ),
        "source_geometry_sha256": registration.get("source_files_sha256", {}).get(
            "w_geometry.json"
        ),
        "max_world_clearance_m": (
            min(
                x.get("world_min_clearance_m", float("inf"))
                for x in (sampled, adaptive)
                if x.get("world_min_clearance_m") is not None
            )
            if any(
                x.get("world_min_clearance_m") is not None for x in (sampled, adaptive)
            )
            else None
        ),
        "max_self_clearance_m": (
            min(
                x.get("self_min_clearance_m", float("inf"))
                for x in (sampled, adaptive)
                if x.get("self_min_clearance_m") is not None
            )
            if any(
                x.get("self_min_clearance_m") is not None for x in (sampled, adaptive)
            )
            else None
        ),
        "source_W_receipt_physical": source["source_W_receipt"].get("physical"),
        "source_W_receipt_scaled95": source["source_W_receipt"].get("scaled95"),
        "source_P_scaled95_qualified": source.get("source_P_scaled95_qualified", False),
        "family_margins": {},
    }
    duration = result["R_duration_s"]
    if duration is not None:
        result["improvement_vs_W_pct"] = 100 * (1 - duration / result["W_duration_s"])

    if (
        folder.joinpath("r_adaptive_dense.npz").exists()
        and folder.joinpath("r_adaptive_pin_torque.npz").exists()
    ):
        states = _load_npz(folder / "r_adaptive_dense.npz")
        torque = _load_npz(folder / "r_adaptive_pin_torque.npz")["torque"]
        dynamic_limits = dict(limits)
        profile = _read_json(folder / "torque_profile.json")
        dynamic_limits["torque"] = np.stack(
            (profile["effort_lower"], profile["effort_upper"])
        )
        arrays = {
            "position": states["q"],
            "velocity": states["qd"],
            "acceleration": states["qdd"],
            "jerk": states["jerk"],
            "torque": torque,
        }
        for family, values in arrays.items():
            physical = dynamic_limits[family]
            strict = family_bounds(physical, family, scaled95=(family != "position"))
            physical_peak = _peak_metrics(
                values, physical, states["time"], torque=family == "torque"
            )
            scaled_peak = _peak_metrics(
                values, strict, states["time"], torque=family == "torque"
            )
            result["family_margins"][family] = {
                "physical_max_utilization": physical_peak["max_utilization"],
                "scaled95_max_utilization": scaled_peak["max_utilization"],
                "physical_min_normalized_margin": 1 - physical_peak["max_utilization"],
                "scaled95_min_normalized_margin": 1 - scaled_peak["max_utilization"],
                "physical_limiting_point": physical_peak,
                "scaled95_limiting_point": scaled_peak,
                "dense_violation": sampled.get("families", {})
                .get(family, {})
                .get("physical", {})
                .get("violation"),
                "adaptive_violation": adaptive.get("families", {})
                .get(family, {})
                .get("physical", {})
                .get("violation"),
                "scaled95_dense_violation": sampled.get("families", {})
                .get(family, {})
                .get("scaled95", {})
                .get("violation"),
                "scaled95_adaptive_violation": adaptive.get("families", {})
                .get(family, {})
                .get("scaled95", {})
                .get("violation"),
            }
        result["scaled95_limiting_family"] = max(
            result["family_margins"],
            key=lambda family: result["family_margins"][family][
                "scaled95_max_utilization"
            ],
        )
        result["scaled95_limiting_joint_zero_based"] = result["family_margins"][
            result["scaled95_limiting_family"]
        ]["scaled95_limiting_point"]["joint_index_zero_based"]
    return result


def _native_summary(native_arm: Path, control_arm: Path) -> dict:
    case = "confirm_geom_cluttered_02"
    case_arm = native_arm / case if (native_arm / case).is_dir() else native_arm
    case_control = control_arm / case if (control_arm / case).is_dir() else control_arm
    traced = _read_json(case_arm / "native_candidates.json").get("F000", {})
    control = _read_json(case_control / "native_candidates.json").get("F000", {})
    traces = sorted(case_arm.glob("*_timeopt_phase_trace.json"))
    if not traces:
        raise FileNotFoundError("native phase trace JSONs missing")
    lane_rows = []
    min_hi, max_lo = [], []
    raw_receipts = []
    for path in traces:
        doc = _read_json(path)
        raw_path = Path(doc["raw_buffers_file"])
        if not raw_path.is_absolute():
            raw_path = ROOT / raw_path
        raw_sha = sha256_file(raw_path)
        if raw_sha != doc.get("raw_buffers_sha256"):
            raise ValueError("native phase NPZ checksum mismatch: " + str(raw_path))
        raw_receipts.append(
            {
                "trace": path.name,
                "trace_sha256": sha256_file(path),
                "buffers": str(raw_path),
                "buffers_sha256": raw_sha,
                "bytes": raw_path.stat().st_size,
            }
        )
        with np.load(raw_path) as archive:
            count = len(archive["input_valid"])
            seed = np.asarray(archive["solver_seed_bounds"])
            # fused_seed reduces per-tile hi by min and lo by max.
            hi = np.min(seed[:, :, 0], axis=1)
            lo = np.max(seed[:, :, 1], axis=1)
            min_hi.extend(hi.tolist())
            max_lo.extend(lo.tolist())
            for lane in range(count):
                lane_rows.append(
                    {
                        "trace": path.name,
                        "lane": lane,
                        "input_valid": bool(archive["input_valid"][lane]),
                        "seed_feasible": bool(archive["solver_seed_feasible"][lane]),
                        "solver_numerical_ok": bool(
                            archive["solver_numerical_ok"][lane]
                        ),
                        "seed_interval_high": float(hi[lane]),
                        "seed_interval_low": float(lo[lane]),
                        "seed_interval_gap_high_minus_low": float(hi[lane] - lo[lane]),
                        "solver_output_timing_finite": bool(
                            np.isfinite(archive["solver_output_timing"][lane]).all()
                        ),
                        "finished_controls_finite": bool(
                            np.isfinite(
                                archive["finished_squared_speed_controls"][lane]
                            ).all()
                        ),
                        "finish_ok": bool(archive["finish_optimization_ok"][lane]),
                        "uniform_duration": float(
                            archive["finished_uniform_duration"][lane]
                        ),
                        "integrated_times_finite": bool(
                            np.isfinite(archive["integrated_times"][lane]).all()
                        ),
                        "timeopt_success": bool(
                            archive["native_timeopt_success"][lane]
                        ),
                        "finish_status": int(archive["finish_status"][lane]),
                    }
                )
    return {
        "case_id": case,
        "instrumented": traced,
        "uninstrumented_control": control,
        "outcome_matches": traced.get("status") == control.get("status")
        and traced.get("planning_attempts") == control.get("planning_attempts"),
        "captured_invocations": len(traces),
        "captured_lanes": len(lane_rows),
        "lane_rows": lane_rows,
        "raw_checksums": raw_receipts,
        "counts": {
            "input_valid_true": sum(x["input_valid"] for x in lane_rows),
            "seed_feasible_true": sum(x["seed_feasible"] for x in lane_rows),
            "numerical_ok_true": sum(x["solver_numerical_ok"] for x in lane_rows),
            "finite_solver_timing": sum(
                x["solver_output_timing_finite"] for x in lane_rows
            ),
            "finite_finished_controls": sum(
                x["finished_controls_finite"] for x in lane_rows
            ),
            "finish_ok": sum(x["finish_ok"] for x in lane_rows),
            "finite_integrated_times": sum(
                x["integrated_times_finite"] for x in lane_rows
            ),
            "native_success": sum(x["timeopt_success"] for x in lane_rows),
            "seed_interval_empty_hi_lt_lo": sum(
                x["seed_interval_high"] < x["seed_interval_low"] for x in lane_rows
            ),
            "finish_status_counts": {
                str(k): sum(x["finish_status"] == k for x in lane_rows)
                for k in sorted({x["finish_status"] for x in lane_rows})
            },
        },
        "seed_interval_high_min": float(np.min(min_hi)),
        "seed_interval_low_max": float(np.max(max_lo)),
        "interpretation": (
            "Observed seed-feasibility flag is false with empty common uniform-scale intervals; "
            "solver numerical flags and raw timing controls remain finite. The false initialization "
            "is followed by finish rejection, infinite uniform duration, and NaN times. The native "
            "finish kernel intentionally writes NaN sentinel states on unsuccessful lanes. This "
            "narrows the observed failure to the seed-feasibility/initialization gate, but does not "
            "prove why the geometric candidates lack an admissible seed scale or rule out graph/buffer effects."
        ),
        "source_semantics": {
            "fused_coefficients_and_seed_bounds": "computes path-wide scalar squared-speed scaling interval intersecting velocity, acceleration, jerk, and signed torque bounds; zero dynamic torque term additionally requires gravity within limits",
            "fused_seed": "reduces upper bounds with min and lower bounds with max; seed_feasible requires a nonempty positive interval (or valid fixed-start reference)",
            "fused_output_status": "requires initialized and input_valid; unsuccessful duration becomes infinity",
            "fused_output_finalize": "writes NaN to integrated time and states when success is false",
            "qualification": "observational post-finalizer values; not causal proof or continuous-time validation",
        },
    }


def _source_provenance(root: Path, source_root: Path) -> dict:
    registration = _read_json(root / "margin_registration.json")
    tasks = _read_json(root / "source_tasks.json")
    scenes = {}
    for task in tasks:
        scene = task.get("scene")
        if scene:
            path = ROOT / "curobo/content/configs/scene" / scene
            if not path.is_file():
                raise FileNotFoundError("scene config missing: " + str(path))
            scenes[scene] = {"path": str(path), "sha256": sha256_file(path)}
    source_arm = source_root / registration["source_arm"]
    for row in registration["sources"]:
        case = row["case_id"]
        for name, digest in row["source_files_sha256"].items():
            if sha256_file(source_arm / case / name) != digest:
                raise ValueError("frozen source artifact changed: " + case + "/" + name)
    for name, digest in registration["model_inputs_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("frozen model input changed: " + name)
    return {
        "registration_sha256": sha256_file(root / "margin_registration.json"),
        "source_tasks_sha256": registration["source_tasks_sha256"],
        "source_W_receipts_sha256": registration["source_w_receipts_sha256"],
        "source_task_checksums": {
            row["case_id"]: row["task_sha256"] for row in registration["sources"]
        },
        "source_W_sha256": {
            row["case_id"]: row["source_files_sha256"]["w_dense.npz"]
            for row in registration["sources"]
        },
        "source_P_sha256": {
            row["case_id"]: row["source_files_sha256"]["p_refined_dense.npz"]
            for row in registration["sources"]
        },
        "source_P_controls_sha256": {
            row["case_id"]: row["source_files_sha256"]["p_timing_controls.npz"]
            for row in registration["sources"]
        },
        "robot_model_sha256": registration["model_inputs_sha256"],
        "scene_configuration_sha256": scenes,
        "scene_hash_note": "Scene hashes supplement the frozen registration: its task hashes identify scene names but the original registration did not separately enumerate scene-file checksums.",
    }


def _make_plots(
    root: Path, source_root: Path, report: Path, rows: list[dict], limits: dict
) -> list[str]:
    assets = report / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    names = arms()
    colors = {
        "W": "#555555",
        "P": "#888888",
        "dilation": "#1874b4",
        "tight_b980": "#37a66b",
        "tight_b960": "#b4a225",
        "tight_b940": "#de7b28",
        "tight_b920": "#bb3d55",
    }
    # Duration overview with the slow qualified W and unqualified-P reference.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, case in zip(axes, CASE_IDS):
        source = next(
            x
            for x in _read_json(root / "margin_registration.json")["sources"]
            if x["case_id"] == case
        )
        durations = [source["source_W_duration_s"], source["source_P_duration_s"]]
        labels = ["W\nqualified", "P\nphysical only"]
        bars_color = [colors["W"], colors["P"]]
        for arm in names:
            record = _read_json(root / arm / case / "r_registration.json")
            durations.append(record["r_duration_s"])
            labels.append(arm.replace("tight_b", "β=").replace("dilation", "dilation"))
            bars_color.append(colors[arm])
        ax.bar(np.arange(len(durations)), durations, color=bars_color)
        ax.set_xticks(
            np.arange(len(labels)), labels, rotation=30, ha="right", fontsize=8
        )
        ax.set_title(case)
        ax.set_ylabel("Execution duration (s)")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Frozen W, original IPOPT P, and new fixed-geometry R timings")
    fig.savefig(assets / "duration_comparison.png", dpi=160)
    plt.close(fig)

    # For each task, show the exact three trajectory references and the
    # shortest qualified R, with utilization in physical time.
    plot_files = ["duration_comparison.png"]
    for case in CASE_IDS:
        src = source_root / SOURCE_ARM / case
        profile = _read_json(src / "torque_profile.json")
        candidates = []
        for arm in names:
            rec = _read_json(root / arm / case / "r_registration.json")
            receipt = _read_json(root / arm / "r_adaptive_receipts.json")[case + ":R"]
            if receipt.get("physical") and receipt.get("scaled95"):
                candidates.append((rec["r_duration_s"], arm))
        best_duration, best_arm = min(candidates)
        w = _load_npz(src / "w_dense.npz")
        p = _load_npz(src / "p_refined_dense.npz")
        rfolder = root / best_arm / case
        r = _load_npz(rfolder / "r_adaptive_dense.npz")
        wt = _load_npz(src / "w_pin_torque.npz")["torque"]
        pt = _load_npz(src / "p_refined_pin_torque.npz")["torque"]
        rt = _load_npz(rfolder / "r_adaptive_pin_torque.npz")["torque"]
        torque_bounds = np.stack((profile["effort_lower"], profile["effort_upper"]))
        tq_limits = family_bounds(torque_bounds, "torque", scaled95=True)
        v_bounds = family_bounds(limits["velocity"], "velocity", scaled95=True)
        a_bounds = family_bounds(limits["acceleration"], "acceleration", scaled95=True)
        j_bounds = family_bounds(limits["jerk"], "jerk", scaled95=True)

        fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
        ax = axes[0, 0]
        for label, file, color in (
            ("P", src / "p_timing_controls.npz", colors["P"]),
            (best_arm, rfolder / "r_speed_controls.npz", colors[best_arm]),
        ):
            controls = _load_npz(file)["squared_speed_controls"]
            basis = BSpline(
                np.r_[np.zeros(3), np.linspace(0, 1, 30), np.ones(3)], controls, 3
            )
            s = np.linspace(0, 1, 500)
            ax.plot(s, basis(s), label=label, color=color)
        ax.set_title("Squared path speed x(s)")
        ax.set_xlabel("Normalized path progress")
        ax.set_ylabel("(progress/s)²")
        ax.legend()
        ax.grid(alpha=0.25)

        for ax, key, bounds, title in (
            (axes[0, 1], "qd", v_bounds, "Max joint velocity utilization (Scaled95)"),
            (
                axes[0, 2],
                "qdd",
                a_bounds,
                "Max joint acceleration utilization (Scaled95)",
            ),
            (axes[1, 0], "jerk", j_bounds, "Max joint jerk utilization (Scaled95)"),
        ):
            for label, data, color in (
                ("W", w, colors["W"]),
                ("P", p, colors["P"]),
                (best_arm, r, colors[best_arm]),
            ):
                value = _utilization(data[key], bounds)
                ax.plot(
                    data["time"],
                    np.max(value, axis=1),
                    label=label,
                    color=color,
                    linewidth=1.1,
                )
            ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8)
            ax.set_title(title)
            ax.set_xlabel("Physical time (s)")
            ax.set_ylabel("Maximum utilization")
            ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        for label, data, tau, color in (
            ("W", w, wt, colors["W"]),
            ("P", p, pt, colors["P"]),
            (best_arm, r, rt, colors[best_arm]),
        ):
            util = _utilization(tau, tq_limits, torque=True)
            # Distinct joint line styles are grouped by trajectory; the
            # seventh joint is highlighted with a heavier stroke.
            for joint in range(7):
                ax.plot(
                    data["time"],
                    util[:, joint],
                    color=color,
                    linewidth=1.3 if joint == 6 else 0.6,
                    alpha=0.9 if joint == 6 else 0.45,
                    label=label if joint == 6 else None,
                )
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8)
        ax.set_title("Signed actuator torque utilization (Scaled95)")
        ax.set_xlabel("Physical time (s)")
        ax.set_ylabel("Max-side |τ| / limit")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

        ax = axes[1, 2]
        for label, file, color in (
            ("W", src / "w_pin_torque.npz", colors["W"]),
            ("P", src / "p_refined_pin_torque.npz", colors["P"]),
            (best_arm, rfolder / "r_adaptive_pin_torque.npz", colors[best_arm]),
        ):
            ee = _load_npz(file)["end_effector"]
            ax.plot(ee[:, 0], ee[:, 1], label=label, color=color)
        ax.set_title("End-effector XY projection")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.suptitle(
            f"{case}: shortest independently qualified R = {best_arm}, {best_duration:.3f} s"
        )
        fig.savefig(assets / f"profiles_{case}.png", dpi=150)
        plt.close(fig)
        plot_files.append(f"profiles_{case}.png")

    # Registered dilation attempts and four IPOPT tightening arms.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, case in zip(axes, CASE_IDS):
        dil = _read_json(root / "dilation" / case / "screen_result.json")
        attempts = dil.get("attempts", [])
        ax.plot(
            [x["rho"] for x in attempts],
            [x["duration_s"] for x in attempts],
            marker="o",
            label="analytic dilation",
        )
        passed = [x for x in attempts if x["conservative_screen_pass"]]
        ax.scatter(
            [x["rho"] for x in passed],
            [x["duration_s"] for x in passed],
            marker="o",
            facecolors="none",
            edgecolors="#1676b8",
            label="preliminary pass",
        )
        betas, durations = [], []
        for beta in TIGHTENING_FACTORS:
            arm = f"tight_b{round(beta*1000):03d}"
            reg = _read_json(root / arm / case / "r_registration.json")
            betas.append(beta)
            durations.append(reg["r_duration_s"])
        ax.plot(betas, durations, marker="s", label="tightened IPOPT (β)")
        ax.axhline(
            _read_json(root / "margin_registration.json")["sources"][
                CASE_IDS.index(case)
            ]["source_W_duration_s"],
            color="gray",
            linestyle="--",
            label="W duration",
        )
        ax.set_title(case)
        ax.set_xlabel("ρ for dilation / β for IPOPT")
        ax.set_ylabel("Candidate duration (s)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle(
        "Dilation grid is discrete; IPOPT tightening is a separate continuation family"
    )
    fig.savefig(assets / "registered_parameter_sweeps.png", dpi=160)
    plt.close(fig)
    plot_files.append("registered_parameter_sweeps.png")
    return plot_files


def publish(
    root: Path, source_root: Path, report: Path, native_arm: Path, native_control: Path
) -> dict:
    data = collect(root, native_arm=native_arm)
    limits = _limits()
    registration = _read_json(root / "margin_registration.json")
    source = {item["case_id"]: item for item in registration["sources"]}
    rows = [
        _task_row(root, case, arm, source[case], limits)
        for case in CASE_IDS
        for arm in arms()
    ]
    native = _native_summary(native_arm, native_control)
    provenance = _source_provenance(root, source_root)
    plots = _make_plots(root, source_root, report, rows, limits)
    report.mkdir(parents=True, exist_ok=True)
    (report / "source_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    (report / "native_phase_summary.json").write_text(
        json.dumps(native, indent=2) + "\n"
    )
    decision = {
        "RESEARCH_GO_MARGIN_RECOVERY": "GO_MARGIN_RECOVERY",
        "PARTIAL_MARGIN_ONLY": "PARTIAL",
        "NO_GO_MARGIN_RECOVERY": "NO_GO",
        "NOT_READY_UNAUDITED": "BLOCKED",
    }.get(data["decision"], data["decision"])
    summary = {
        "schema_version": 1,
        "status": "COMPLETE_SAMPLED_QUALIFICATION",
        "decision": decision,
        "scientific_scope": "three deliberately selected synthetic-torque cases; no population CI or global optimality claim",
        "source_W_status": "independently sampled physical+Scaled95 qualified in frozen source study",
        "source_P_status": "independently sampled physical qualified but Scaled95 unqualified; not an attainable strict-contract optimum reference",
        "qualification": "Independent Pinocchio and sphere-model dense+adaptive sampled physical+Scaled95; not continuous-time torque/collision certification",
        "material_improvement_threshold_pct": data[
            "material_duration_gain_threshold_pct"
        ],
        "rows": rows,
        "source_provenance": provenance,
        "native_phase": native,
        "plots": plots,
        "historical_summary": data,
        "counts": {
            "tasks": len(CASE_IDS),
            "task_method_cells": len(rows),
            "qualified_physical_scaled95": sum(
                row["physical"] and row["scaled95"] for row in rows
            ),
            "qualified_with_at_least_10pct_gain": sum(
                row["physical"]
                and row["scaled95"]
                and row["improvement_vs_W_pct"] >= 10
                for row in rows
            ),
            "all_native_traced_lanes": native["captured_lanes"],
        },
    }
    (report / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    native_files = native["raw_checksums"]
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            + "".join(
                "<td>" + html.escape(str(row.get(key, "—"))) + "</td>"
                for key in (
                    "case_id",
                    "arm",
                    "W_duration_s",
                    "P_duration_s",
                    "R_duration_s",
                    "improvement_vs_W_pct",
                    "solver_converged",
                    "iterations",
                    "ipopt_solve_wall_s",
                    "dilation_screen_wall_s",
                    "dilation_total_wall_s",
                    "physical",
                    "scaled95",
                    "scaled95_limiting_family",
                    "scaled95_limiting_joint_zero_based",
                    "max_world_clearance_m",
                    "max_self_clearance_m",
                )
            )
            + "</tr>"
        )
    sections = []
    for case in CASE_IDS:
        case_rows = [row for row in rows if row["case_id"] == case]
        margin_rows = []
        for row in case_rows:
            for family, metrics in row["family_margins"].items():
                physical_peak = metrics["physical_limiting_point"]
                scaled_peak = metrics["scaled95_limiting_point"]
                fields = (
                    row["arm"],
                    family,
                    f"{metrics['physical_max_utilization']:.6f}",
                    f"{physical_peak['joint_index_zero_based'] + 1} / {physical_peak['time_s']:.6f}",
                    f"{metrics['scaled95_max_utilization']:.6f}",
                    f"{scaled_peak['joint_index_zero_based'] + 1} / {scaled_peak['time_s']:.6f}",
                )
                margin_rows.append(
                    "<tr>"
                    + "".join(
                        "<td>" + html.escape(str(value)) + "</td>" for value in fields
                    )
                    + "</tr>"
                )
        sections.append(
            f"<section><h3>{html.escape(case)}</h3><img src='assets/profiles_{html.escape(case)}.png' alt='Measured trajectory profile plot for {html.escape(case)}'>"
            "<h4>Independent adaptive-sample utilization and limiting points</h4>"
            "<div style='overflow:auto;max-height:500px'><table><thead><tr><th>Method</th><th>Family</th><th>Physical max utilization</th><th>Physical joint / time (s)</th><th>Scaled95 max utilization</th><th>Scaled95 joint / time (s)</th></tr></thead><tbody>"
            + "".join(margin_rows)
            + "</tbody></table></div></section>"
        )
    native_table = (
        "<table><thead><tr><th>native attempts</th><th>trace status</th><th>input valid</th><th>seed feasible</th><th>solver numeric OK</th><th>finite solver controls</th><th>finish success</th><th>finite integrated times</th><th>empty seed intervals</th></tr></thead><tbody><tr>"
        + "".join(
            f"<td>{html.escape(str(x))}</td>"
            for x in (
                native["instrumented"].get("planning_attempts"),
                native["instrumented"].get("status"),
                native["counts"]["input_valid_true"],
                native["counts"]["seed_feasible_true"],
                native["counts"]["numerical_ok_true"],
                native["counts"]["finite_solver_timing"],
                native["counts"]["finish_ok"],
                native["counts"]["finite_integrated_times"],
                native["counts"]["seed_interval_empty_hi_lt_lo"],
            )
        )
        + "</tr></tbody></table>"
    )
    page = f"""<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Margin-aware fixed-geometry retiming and native TimeOpt diagnostics</title>
<style>body{{font:15px/1.55 system-ui;max-width:1400px;margin:2rem auto;padding:1rem;color:#18232e}}h1,h2{{line-height:1.2}}.callout{{background:#e9f6ee;border-left:5px solid #24814e;padding:1rem}}.warning{{background:#fff1dc;border-left:5px solid #b87911;padding:1rem}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:.38rem;border-bottom:1px solid #dce3e8;text-align:left}}th{{background:#edf2f9;position:sticky;top:0}}img{{max-width:100%;height:auto}}section{{border-top:1px solid #ccd4dc;margin-top:1.5rem}}input{{padding:.5rem;width:24rem}}pre{{overflow:auto;background:#f5f7f9;padding:1rem}}code{{background:#f0f2f4;padding:.1rem .25rem}}</style>
<h1>Margin-aware fixed-geometry retiming</h1>
<p class='callout'><b>{html.escape(decision)}:</b> {summary['counts']['qualified_physical_scaled95']} of {len(rows)} new candidates passed independent physical and Scaled95 sampled qualification. {summary['counts']['qualified_with_at_least_10pct_gain']} also exceeded the registered 10% duration reduction from W. The paths remain fixed to the original W/P quintic geometry.</p>
<p class='warning'>W is a qualified strict-contract reference. P is a faster IPOPT reference that failed Scaled95 and is not a qualified strict-contract optimum. Dense/adaptive checks are sampled validation, not continuous-time torque or collision certification. IPOPT results are local solutions, not global time optima.</p>
<h2>Frozen cohort</h2><p>Three selected synthetic-torque tasks; all 15 registered method cells are present. No population-level confidence intervals are claimed.</p>
<img src='assets/duration_comparison.png' alt='Duration comparison by task'>
<h2>All task/method outcomes</h2><label>Search <input id='filter' placeholder='filter by task, arm, status'></label>
<div style='overflow:auto;max-height:650px'><table id='results'><thead><tr><th>Task</th><th>Method</th><th>W s</th><th>P s</th><th>R s</th><th>Reduction vs W %</th><th>IPOPT converged</th><th>Iterations</th><th>IPOPT solve s</th><th>Dilation screen s</th><th>Dilation total CPU s</th><th>Physical</th><th>Scaled95</th><th>Scaled95 limiting family</th><th>Joint index</th><th>World clearance m</th><th>Self clearance m</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<script>document.getElementById('filter').addEventListener('input',e=>{{const q=e.target.value.toLowerCase();for(const r of document.querySelectorAll('#results tbody tr'))r.hidden=!r.innerText.toLowerCase().includes(q)}})</script>
<h2>Duration and constraint profiles</h2><p>The dilation arm uses the exact frozen IPOPT squared-speed controls scaled as x/ρ² and scans the registered finite ρ grid. Tightened IPOPT arms keep the geometry fixed and tighten all dynamic envelopes around the registered centers (torque about zero). Plot selection uses saved R trajectories and independent Pinocchio torque audits.</p>
<img src='assets/registered_parameter_sweeps.png' alt='Dilation and beta parameter sweeps'>{''.join(sections)}
<h2>Native TimeOpt phase diagnosis</h2><p>The instrumented and uninstrumented runs are compared by outcome, not latency. {native['captured_lanes']} actual lanes from {native['captured_invocations']} finalizer snapshots were captured. Raw numerical arrays are checksum-linked below.</p>{native_table}
<p>{html.escape(native['interpretation'])}</p><p>Observed source semantics: seed bounds intersect a scalar squared-speed scale across sampled velocity, acceleration, jerk and torque constraints. An empty interval leaves the lane uninitialized. The finish logic emits infinite duration and NaN sentinel states for unsuccessful lanes. This narrows the observed behavior to the seed-feasibility/initialization gate; it does not establish why every path had an empty interval or prove that a different timing profile on that geometry is impossible.</p>
<h3>Native raw buffer receipts</h3><table><thead><tr><th>Trace</th><th>Trace SHA256</th><th>Raw buffers</th><th>NPZ SHA256</th><th>Bytes</th></tr></thead><tbody>{''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (x['trace'],x['trace_sha256'],x['buffers'],x['buffers_sha256'],x['bytes']))+'</tr>' for x in native_files)}</tbody></table>
<h2>Source provenance and limitations</h2><p>See <a href='source_provenance.json'>source_provenance.json</a>, <a href='native_phase_summary.json'>native_phase_summary.json</a>, <a href='summary.json'>machine-readable summary</a>, and <a href='artifact_manifest.json'>full SHA256 artifact manifest</a>. Large trajectory NPZs are retained at the exact manifest paths and are not embedded in this HTML.</p><p>Source scene identifiers were frozen in task records; this report also records checksums for the exact scene YAML files. The three cases are preselected, so results describe these cases only. Collision and dynamics checks remain sampled. Synthetic derating is not a physical payload experiment. No continuous-time certificate or global optimality claim is made.</p></html>"""
    (report / "index.html").write_text(page, encoding="utf-8")
    manifest = generate_manifest(
        root,
        report,
        source=source_root,
        extra_roots=(
            ("native_phase_instrumented", native_arm.parents[1]),
            ("native_phase_control", native_control.parents[1]),
        ),
    )
    summary["artifact_manifest"] = manifest
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT)
    parser.add_argument("--source", type=Path, default=PREVIOUS)
    parser.add_argument("--native-arm", type=Path, default=NATIVE_PROBE)
    parser.add_argument("--native-control", type=Path, default=NATIVE_CONTROL)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    result = publish(
        args.root, args.source, args.report, args.native_arm, args.native_control
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "counts": result["counts"],
                "report": str(args.report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
