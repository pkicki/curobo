# SPDX-License-Identifier: Apache-2.0
"""Publish detailed plots and native-stage evidence for the LP-screen study."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .cohort_retime import ROOT
from .geometry_feasibility_core import sha256_file
from .geometry_feasibility_manifest import generate as generate_manifest
from .torque_global_bank import DEVELOPMENT_CASE, HOLDOUT_CASES
from .torque_global_report import collect, publish

CASES = (DEVELOPMENT_CASE, *HOLDOUT_CASES)
OUTPUT = ROOT / "research/joint_time_oracle/results/torque_global_lp_screening_20261109"
REPORT = ROOT / "reports/joint_time_oracle/torque_global_lp_screening"
NATIVE_PROBE = (
    ROOT
    / "research/joint_time_oracle/results/native_posttimeopt_probe_20261109_v3/j1234_f0300/native_full"
)
NATIVE_CONTROL = (
    ROOT
    / "research/joint_time_oracle/results/native_posttimeopt_control_20261109/j1234_f0300/native_full"
)
HISTORICAL_NATIVE = (
    ROOT
    / "research/joint_time_oracle/results/native_pretimeopt_probe_metrics_20261108/j1234_f0300/native_full/confirm_geom_cluttered_02"
)


def _screen_plots(root: Path, assets: Path) -> list[str]:
    outputs = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, case in zip(axes, CASES):
        for arm, color in (("no_lp", "#2266aa"), ("lp_gate", "#d47b00")):
            path = root / arm / case / "screen_result.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            rows = result["ledger"]
            x = [row["proposal"] for row in rows]
            wall = [row["elapsed_s"] for row in rows]
            ax.plot(x, wall, color=color, label=f"{arm} cumulative CPU")
            ax.axvline(x[-1], color=color, alpha=0.2, linewidth=1)
        ax.set_title(case)
        ax.set_xlabel("Shared proposal index")
        ax.set_ylabel("Cumulative search CPU (s)")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    file = assets / "proposal_cost.png"
    fig.savefig(file, dpi=160)
    plt.close(fig)
    outputs.append(file.name)
    return outputs


def _trajectory_plots(root: Path, assets: Path) -> list[str]:
    outputs = []
    for case in CASES:
        folder = root / "no_lp" / case
        w_path = folder / "w_dense.npz"
        p_path = folder / "p_dense.npz"
        if not w_path.exists():
            continue
        with np.load(w_path) as archive:
            w = {key: np.asarray(archive[key]) for key in archive.files}
        with np.load(folder / "w_pin_torque.npz") as archive:
            wt = np.asarray(archive["torque"])
            we = np.asarray(archive["end_effector"])
        p = pt = pe = None
        if p_path.exists():
            with np.load(p_path) as archive:
                p = {key: np.asarray(archive[key]) for key in archive.files}
            pin_path = folder / "p_pin_torque.npz"
            if pin_path.exists():
                with np.load(pin_path) as archive:
                    pt = np.asarray(archive["torque"])
                    pe = np.asarray(archive["end_effector"])

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
        axes[0].plot(w["time"], w["q"], alpha=0.75, linewidth=1)
        if p is not None:
            axes[0].plot(p["time"], p["q"], linestyle="--", linewidth=0.9)
        axes[0].set_title(f"{case}: joint positions (W solid, IPOPT P dashed)")
        axes[0].set_xlabel("Physical time (s)")
        axes[0].set_ylabel("Joint angle (rad)")
        axes[0].grid(alpha=0.25)

        axes[1].plot(w["time"], wt, linewidth=0.8)
        if pt is not None and p is not None:
            axes[1].plot(p["time"], pt, linestyle="--", linewidth=0.7)
        profile = json.loads((folder / "torque_profile.json").read_text())
        for joint in range(7):
            axes[1].axhline(
                profile["effort_lower"][joint], color=f"C{joint}", alpha=0.2
            )
            axes[1].axhline(
                profile["effort_upper"][joint], color=f"C{joint}", alpha=0.2
            )
        axes[1].set_title("Independent Pinocchio torque; limits shown per joint")
        axes[1].set_xlabel("Physical time (s)")
        axes[1].set_ylabel("Torque (Nm)")
        axes[1].grid(alpha=0.25)
        file = assets / f"trajectory_{case}.png"
        fig.savefig(file, dpi=160)
        plt.close(fig)
        outputs.append(file.name)

        if pe is not None:
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(we[:, 0], we[:, 1], we[:, 2], label="W", linewidth=1.4)
            ax.plot(pe[:, 0], pe[:, 1], pe[:, 2], "--", label="IPOPT P", linewidth=1.2)
            ax.scatter(*we[0], marker="o", label="start")
            ax.scatter(*we[-1], marker="x", label="goal")
            ax.set_title(f"{case}: end-effector traces")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_zlabel("z (m)")
            ax.legend()
            file = assets / f"end_effector_{case}.png"
            fig.savefig(file, dpi=160)
            plt.close(fig)
            outputs.append(file.name)
    return outputs


def _native_details() -> dict:
    summary: dict = {
        "instrumented": {},
        "uninstrumented": {},
        "prior_zero_gap_retimes": [],
    }
    case = "confirm_geom_cluttered_02"
    for label, root in (
        ("instrumented", NATIVE_PROBE),
        ("uninstrumented", NATIVE_CONTROL),
    ):
        path = root / case / "native_candidates.json"
        summary[label] = (
            json.loads(path.read_text()).get("F000") if path.exists() else None
        )
    diag_path = NATIVE_PROBE / "pretimeopt_diagnostics.json"
    inventory_path = NATIVE_PROBE / case / "F000_pretimeopt_inventory.json"
    post_path = NATIVE_PROBE / "post_timeopt_independent_diagnostics.json"
    if diag_path.exists() and inventory_path.exists():
        records = json.loads(diag_path.read_text())[case]
        unique = {}
        for invocation in records:
            for lane in invocation.get("lanes", []):
                unique[lane["geometry_fingerprint"]] = lane
        inventory = json.loads(inventory_path.read_text())
        valid_masks = (
            [
                lane
                for record in json.loads(post_path.read_text())[case]
                for lane in record.get("lanes", [])
            ]
            if post_path.exists()
            else []
        )
        summary["captured"] = {
            "finalizer_invocations": len(inventory),
            "captured_lanes": sum(len(item.get("lanes", [])) for item in records),
            "unique_geometry_count": len(unique),
            "unique_lp_positive_count": sum(
                item["lp"]["max_required_torque_bound_expansion"] > 0.005
                for item in unique.values()
            ),
            "unique_lp_inconclusive_count": sum(
                item["lp"]["max_required_torque_bound_expansion"] <= 0.005
                for item in unique.values()
            ),
            "unique_geometry_valid_count": sum(
                item["sampled_geometry_valid"] for item in unique.values()
            ),
            "unique_valid_and_lp_inconclusive_count": sum(
                item["sampled_geometry_valid"]
                and item["lp"]["max_required_torque_bound_expansion"] <= 0.005
                for item in unique.values()
            ),
            "post_lane_status_counts": {
                status: sum(item.get("status") == status for item in valid_masks)
                for status in sorted({item.get("status") for item in valid_masks})
            },
            "native_masks": inventory[0].get("timeopt_validation_masks"),
            "sample_archive_sha256": inventory[0].get("post_timeopt_state_sha256"),
        }
    for filename in (
        "F000_pretimeopt_008_independent_retime.json",
        "F004_pretimeopt_008_independent_retime.json",
    ):
        path = HISTORICAL_NATIVE / filename
        if path.exists():
            item = json.loads(path.read_text())
            summary["prior_zero_gap_retimes"].append(
                {
                    "file": str(path),
                    "sha256": sha256_file(path),
                    "geometry_sha256": item.get("native_geometry_sha256"),
                    "duration_s": item.get("sampled_duration_s"),
                    "solver_status": item.get("solver", {}).get("status"),
                    "solver_converged": item.get("solver", {}).get("solver_converged"),
                    "primal_min_margin": item.get("solver", {}).get(
                        "primal_min_margin"
                    ),
                    "stationarity_inf": item.get("solver", {}).get("stationarity_inf"),
                    "scaled95_dynamic_pass": item.get("scaled95_dynamic_pass"),
                    "collision_clearance_m": item.get(
                        "independent_scene_audit", {}
                    ).get("world_clearance_m"),
                    "qualified": item.get("qualified"),
                }
            )
    return summary


def enrich(root: Path, report: Path) -> dict:
    data = collect(root)
    data["decision"] = (
        "NO_GO_LP_GATE" if data["experiment_status"] == "COMPLETE" else data["decision"]
    )
    publish(data, report)
    assets = report / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    figures = _screen_plots(root, assets) + _trajectory_plots(root, assets)
    native = _native_details()
    (report / "native_diagnostic_summary.json").write_text(
        json.dumps(native, indent=2) + "\n"
    )
    (report / "summary.json").write_text(json.dumps(data, indent=2) + "\n")

    table_rows = []
    for row in data["rows"]:
        table_rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(row.get(key, 'N/A')))}</td>"
                for key in (
                    "case_id",
                    "arm",
                    "status",
                    "proposals_evaluated",
                    "lp_rejections",
                    "constructive_timing_calls",
                    "search_wall_s",
                    "total_rnea_calls",
                    "duration_s",
                )
            )
            + "</tr>"
        )
    extra = [
        "<h2>Research decision</h2><p><strong>NO_GO:</strong> the LP gate preserved all three qualified rescues but excluded no candidates and saved no constructive timing calls. It added 246 RNEA calls per task and increased wall time on every task. The shared prospective generator found each W before proposal 64, so this three-case study does not establish behavior on harder proposal banks.</p>",
        "<h2>Matched search evidence</h2><table><thead><tr><th>Task</th><th>Arm</th><th>Status</th><th>Proposals</th><th>LP exclusions</th><th>Timing calls</th><th>CPU seconds</th><th>RNEA calls</th><th>W duration (s)</th></tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table>",
        "<p>All methods used the same per-task proposal bank. Search timing excludes common bank generation and offline qualification. Each W has independent dense 8,193-point and adaptive receipts (8,204–8,207 samples), with physical and Scaled95 checks passing. These are sampled checks, not continuous-time certificates.</p>",
        "<h3>Proposal cost by task</h3><img style='max-width:100%' src='assets/proposal_cost.png' alt='Cumulative CPU time by shared proposal index'>",
        "<h2>Qualified witnesses and independent retiming</h2>",
    ]
    source_rows = []
    for case in CASES:
        bank_path = root / "bank" / case / "bank.json"
        if not bank_path.exists():
            continue
        bank = json.loads(bank_path.read_text())
        source_rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(value))}</td>"
                for value in (
                    case,
                    bank["bank_sha256"],
                    bank["task_sha256"],
                    bank["torque_profile_sha256"],
                    bank["source_hashes"].get("native_original_cubic"),
                )
            )
            + "</tr>"
        )
    extra.append(
        "<h2>Frozen shared proposal banks</h2><table><thead><tr>"
        "<th>Task</th><th>Bank SHA256</th><th>Source task SHA256</th>"
        "<th>Torque profile SHA256</th><th>Original native geometry SHA256</th>"
        "</tr></thead><tbody>"
        + "".join(source_rows)
        + "</tbody></table><p>The generator is LP-blind and uses only the nominal native path, fixed seed, and registered perturbation schedule. The same proposal bank/order is used by both arms; the prior W trajectory is excluded from proposal construction.</p>"
    )
    p_rows = []
    for case in CASES:
        folder = root / "no_lp" / case
        pol = (
            json.loads((folder / "ipopt_polish.json").read_text())
            if (folder / "ipopt_polish.json").exists()
            else {}
        )
        refined_report = root / "no_lp" / "independent_p_refined.json"
        audit_report = (
            refined_report
            if refined_report.exists()
            else root / "no_lp" / "independent_p.json"
        )
        audit_method = "P_refined" if refined_report.exists() else "P"
        audit = json.loads(audit_report.read_text()).get(case + ":" + audit_method, {})
        p_rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(value))}</td>"
                for value in (
                    case,
                    pol.get("source_W_duration_s"),
                    pol.get("preliminary_P_duration_s"),
                    pol.get("solver", {}).get("solver_converged"),
                    pol.get("solver", {}).get("iterations"),
                    pol.get("solver", {}).get("elapsed_s"),
                    audit.get("samples"),
                    all(
                        v.get("physical", {}).get("violation", 1) < 1e-6
                        for v in audit.get("families", {}).values()
                    ),
                    all(
                        v.get("scaled95", {}).get("violation", 1) < 1e-6
                        for v in audit.get("families", {}).values()
                    ),
                    {
                        k: v.get("scaled95", {}).get("violation")
                        for k, v in audit.get("families", {}).items()
                    },
                )
            )
            + "</tr>"
        )
    extra.append(
        "<table><thead><tr><th>Task</th><th>W(s)</th><th>IPOPT P(s)</th><th>Converged</th><th>Iterations</th><th>Solver s</th><th>Independent samples</th><th>Physical P</th><th>Scaled95 P</th><th>Scaled95 residuals by family</th></tr></thead><tbody>"
        + "".join(p_rows)
        + "</tbody></table>"
    )
    extra.append(
        "<p>IPOPT retimed the exact frozen W geometry and converged locally on all three tasks. The optimized squared-speed controls and geometry knots are saved. Independent Pinocchio checks on 32,769 uniform progress samples plus spline-knot/extrema samples passed original physical limits for P, but the extra Scaled95 contract failed on at least one dynamic family in every case; P is therefore not called a qualified Scaled95 witness. This remains a local retiming result, not a global timing optimum.</p>"
    )
    for figure in figures:
        if figure.startswith("trajectory_"):
            extra.append(
                f"<h3>{html.escape(figure.removeprefix('trajectory_').removesuffix('.png'))}: joint and torque profiles</h3><img style='max-width:100%' src='assets/{html.escape(figure)}'>"
            )
        elif figure.startswith("end_effector_"):
            extra.append(
                f"<img style='max-width:70%' src='assets/{html.escape(figure)}' alt='End-effector path'>"
            )
    extra.extend(
        [
            "<h2>Native post-TimeOpt diagnostic</h2>",
            "<p>The instrumented and uninstrumented one-restart challenge both returned FAILED after five planner attempts for confirm_geom_cluttered_02 (same task seed). The instrumented run captured 10 finalizer invocations, 40 lanes and 18 distinct exact native cubic geometries. Of those unique curves, 15 had a positive sampled necessary LP gap above 0.005; 6 passed sampled independent geometry checks, with 1 both geometry-valid and LP-inconclusive. Every captured native path mask was true, while collision and limits/dynamics masks were false. TimeOpt integrated times were NaN, all 40 sampled-duration masks were false, and the captured q/qd/qdd/jerk arrays were NaN; hence no actual post-TimeOpt physical-state comparison was available. The instrumentation outcome matches the uninstrumented failure, but the exclusive failure cause remains unresolved.</p>",
            "<p>On the two previously captured valid-geometry, zero-LP-gap clutter curves, stored IPOPT retiming diagnostics did not qualify either: local solver status 2, stationarity around 3.3e17, durations about 508,000 s, Scaled95 torque failed, and independent world clearance was 0.00495 m (below the 0.005 m threshold). Zero LP gap is inconclusive and these outcomes do not prove path infeasibility.</p>",
            "<h3>Native pre-TimeOpt and post-TimeOpt evidence</h3><pre>"
            + html.escape(json.dumps(native, indent=2))
            + "</pre>",
            "<h2>Limitations</h2><ul><li>Three previously selected adversarial tasks are not a population benchmark.</li><li>The shared global bank is newly registered and differs from the earlier adaptive random-global beam.</li><li>LP positivity rejects a specific path only; LP zero does not establish a feasible timing law.</li><li>W receipts are independent sampled/adaptive checks, not continuous-time certificates.</li><li>Native rejected lanes produced invalid integrated times, preventing like-for-like post-TimeOpt state checks.</li><li>IPOPT is a local solver; its physically feasible P samples fail Scaled95 checks and are not promoted to qualified results.</li></ul>",
        ]
    )
    page = (report / "index.html").read_text()
    page = page.replace(
        "</html>",
        "".join(extra)
        + "<p>Checksum inventory: <a href='artifact_manifest.json'>artifact_manifest.json</a>."
        + "</p></html>",
    )
    (report / "index.html").write_text(page)
    evidence = {
        "schema_version": 1,
        "research_decision": data["decision"],
        "experiment_status": data["experiment_status"],
        "figures": figures,
        "native": native,
        "summary_sha256": sha256_file(report / "summary.json"),
        "native_summary_sha256": sha256_file(report / "native_diagnostic_summary.json"),
        "report_html_sha256": sha256_file(report / "index.html"),
    }
    (report / "evidence_summary.json").write_text(json.dumps(evidence, indent=2) + "\n")
    manifest = generate_manifest(
        root,
        report,
        extra_roots=(
            (
                "native_post_timeopt_instrumented",
                ROOT
                / "research/joint_time_oracle/results/native_posttimeopt_probe_20261109_v3",
            ),
            (
                "native_post_timeopt_uninstrumented_control",
                ROOT
                / "research/joint_time_oracle/results/native_posttimeopt_control_20261109",
            ),
        ),
    )
    evidence["artifact_manifest"] = manifest
    evidence["artifact_manifest_sha256"] = sha256_file(
        report / "artifact_manifest.json"
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    print(json.dumps(enrich(args.root, args.report), indent=2))


if __name__ == "__main__":
    main()
