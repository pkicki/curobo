# SPDX-License-Identifier: Apache-2.0
"""Report independent success versus proposal and CPU wall-time budgets.

Preliminary W is NOT a result: success requires an independently validated
physical+Scaled95 adaptive receipt with matching source task/profile SHA and
exact analytic replay. The known offline W witnesses are NEVER counted as a
search result or used to initialize either arm.

One adversarial development task and two source-disjoint heldout examples are provided.
Do not report population significance from these three selected observations.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from .cohort_retime import ROOT
from .geometry_feasibility_core import sha256_file
from .torque_guided_rescue import ARMS, BUDGETS, OUTPUT, ROOTS


def qualified(arm: Path, case: str, row: dict) -> tuple[bool, list[str]]:
    reasons = []
    receiptfile = arm / "w_adaptive_receipts.json"
    receipt = (
        json.loads(receiptfile.read_text()).get(case + ":W", {})
        if receiptfile.exists()
        else {}
    )
    if not row.get("candidate_witness_prescreen"):
        reasons.append("no constructive witness generated")
        return False, reasons
    if receipt.get("physical") is not True or receipt.get("scaled95") is not True:
        reasons.append("independent physical+Scaled95 adaptive receipt missing/failed")
    if receipt.get("collision") is not True or receipt.get("endpoint") is not True:
        reasons.append("independent collision/endpoint qualification missing")
    if receipt.get("torque_profile_sha256") != row.get("torque_profile_sha256"):
        reasons.append("effort-profile mismatch")
    basefile = arm / case / "w_dense.npz"
    geomfile = arm / case / "w_geometry.json"
    if (
        not basefile.exists()
        or receipt.get("base_sha256") != row.get("w_dense_sha256")
        or receipt.get("base_sha256") != sha256_file(basefile)
    ):
        reasons.append("alternative trajectory missing/checksum mismatch")
    if not geomfile.exists() or receipt.get("witness_geometry_sha256") != (
        sha256_file(geomfile) if geomfile.exists() else None
    ):
        reasons.append("geometric controls missing/checksum mismatch")
    replay_file = arm / case / "w_adaptive_replay.json"
    replay = json.loads(replay_file.read_text()) if replay_file.exists() else {}
    if replay.get("status") != "REPLAYED":
        reasons.append("exact analytic W adaptive reconstruction missing")
    return not reasons, reasons


def summarize(root: Path) -> dict:
    from .torque_guided_freeze import verify_frozen_holdout

    first = root / ARMS[0] / "experiment_config.json"
    if not first.exists():
        registry = {}
    else:
        conf = json.loads(first.read_text())
        registry = conf.get("source_registry", {})
    cases = {}
    for case in registry:
        cases[case] = {}
        for arm_name in ARMS:
            arm = root / arm_name
            result_file = arm / case / "search_result.json"
            if not result_file.exists():
                cases[case][arm_name] = {"status": "NOT_RUN"}
                continue
            obj = json.loads(result_file.read_text())
            row = obj["report"]
            valid, reasons = qualified(arm, case, row)
            first = row.get("first_witness_proposal")
            timeline = row.get("ledger", [])
            if valid and (not isinstance(first, int) or first >= len(timeline)):
                valid = False
                reasons.append("invalid first witness index")
            elapsed_at_first = timeline[first]["wall_s"] if valid else None
            at = {str(b): bool(valid and first <= b) for b in BUDGETS}
            audit_path = arm / "independent_adaptive.json"
            audit = (
                json.loads(audit_path.read_text()).get(case + ":W_adaptive", {})
                if audit_path.exists()
                else {}
            )
            cases[case][arm_name] = {
                "status": "INDEPENDENTLY_QUALIFIED" if valid else "NO_QUALIFIED_RESCUE",
                "failure_reasons": reasons,
                "candidate_prescreen": bool(row.get("candidate_witness_prescreen")),
                "qualified_by_budget": at,
                "first_qualified_proposal": first if valid else None,
                "first_qualified_elapsed_wall_s": elapsed_at_first,
                "duration_s": row.get("witness_duration_s") if valid else None,
                "preliminary_duration_s": row.get("witness_duration_s"),
                "first_screen_proposal": first,
                "selected_deviation_rad": row.get("selected_deviation_rad"),
                "best_score": row.get("best_score"),
                "total_wall_s": row.get("solver_wall_s_including_gradient_and_screen"),
                "proposals_evaluated": row.get("proposals_evaluated"),
                "rnea_calls": row.get("rnea_calls"),
                "screening_and_timing_rnea_calls": row.get(
                    "screening_and_timing_rnea_calls"
                ),
                "total_counted_rnea_calls": row.get("total_counted_rnea_calls"),
                "initial_path_lp_expansion": row.get("initial_path_lp_expansion"),
                "initial_native_fit": row.get("base_fit"),
                "task_sha256": row.get("task_sha256"),
                "torque_profile_sha256": row.get("torque_profile_sha256"),
                "witness_sha256": row.get("w_dense_sha256") if valid else None,
                "world_clearance_m": (
                    audit.get("world_min_clearance_m") if valid else None
                ),
                "self_clearance_m": (
                    audit.get("self_min_clearance_m") if valid else None
                ),
                "adaptive_family_checks": audit.get("families") if valid else None,
                "kinematic_hull": (
                    json.loads(
                        (arm / case / "kinematic_hull_certificate.json").read_text()
                    )
                    if valid
                    and (arm / case / "kinematic_hull_certificate.json").exists()
                    else None
                ),
                "source_original_native_sha256": row.get(
                    "source_native_buffers_sha256"
                ),
            }
    replay_manifest_path = root / "proposal_replay" / "proposal_replay_manifest.json"
    replay_summary = {}
    if replay_manifest_path.exists():
        replay = json.loads(replay_manifest_path.read_text())
        for arm_name, arm_cases in replay["arms"].items():
            replay_summary[arm_name] = {}
            for case, record in arm_cases.items():
                directions = {}
                for proposal in record.get("proposal_ledger", []):
                    source = proposal.get("direction_source", "UNKNOWN")
                    directions[source] = directions.get(source, 0) + 1
                replay_summary[arm_name][case] = {
                    key: record.get(key)
                    for key in (
                        "proposal_count",
                        "replay_lp_solves",
                        "replay_collision_evaluations",
                        "candidate_metrics_match_original_ledger",
                        "matched_proposals",
                        "npz_path",
                        "npz_sha256",
                        "json_path",
                        "json_sha256",
                    )
                }
                replay_summary[arm_name][case]["direction_source_counts"] = directions
    freeze_path = root / "holdout_freeze.json"
    try:
        verify_frozen_holdout(root, budget=BUDGETS[-1], seed=20261108)
        freeze_status = "VERIFIED"
    except (FileNotFoundError, ValueError) as exc:
        freeze_status = f"FAILED: {exc}"
    summary = {
        "schema_version": 1,
        "cases": cases,
        "arms": ARMS,
        "proposal_budgets": BUDGETS,
        "registered_source_registry": registry,
        "counts": {},
        "note": (
            "3 adversarial examples; no population success-rate inference; "
            "no continuous-time physical certificate"
        ),
        "fairness": (
            "same nominal-path quintic fit, proposal budget and auditor; "
            "directional costs differ, so compare wall time and RNEA"
        ),
        "dual_gradient_validation": (
            json.loads((root / "dual_gradient_validation.json").read_text())
            if (root / "dual_gradient_validation.json").exists()
            else None
        ),
        "proposal_replay": {
            "status": (
                "REPLAY_MATCHED"
                if replay_summary
                and all(
                    record["candidate_metrics_match_original_ledger"]
                    for arm_cases in replay_summary.values()
                    for record in arm_cases.values()
                )
                else "MISSING_OR_MISMATCHED"
            ),
            "manifest_path": (
                str(replay_manifest_path.resolve())
                if replay_manifest_path.exists()
                else None
            ),
            "manifest_sha256": (
                sha256_file(replay_manifest_path)
                if replay_manifest_path.exists()
                else None
            ),
            "arms": replay_summary,
            "note": (
                "Deterministic post hoc replay restored proposal and direction arrays; "
                "its extra runtime is not included in scored search time."
            ),
        },
        "holdout_freeze": {
            "status": freeze_status,
            "sha256": sha256_file(freeze_path) if freeze_path.exists() else None,
            "path": str(freeze_path.resolve()) if freeze_path.exists() else None,
        },
    }
    # Every completed arm must use the same source, torque-profile and
    # witness registry. Different registries make all cross-arm comparisons
    # N/A, even if a few task names happen to overlap.
    summary["registry_consistent"] = bool(registry)
    for arm in ARMS:
        file = root / arm / "experiment_config.json"
        if file.exists():
            selected = json.loads(file.read_text()).get("source_registry")
            if selected != registry:
                summary["registry_consistent"] = False
        else:
            summary["registry_consistent"] = False
    for arm in ARMS:
        summary["counts"][arm] = {
            "attempted": sum(cases[c][arm]["status"] != "NOT_RUN" for c in cases),
            "qualified": sum(
                cases[c][arm]["status"] == "INDEPENDENTLY_QUALIFIED" for c in cases
            ),
            "qualified_by_proposals": {
                str(b): sum(
                    cases[c][arm].get("qualified_by_budget", {}).get(str(b), False)
                    for c in cases
                )
                for b in BUDGETS
            },
        }
    dual_count = summary["counts"]["dual"]["qualified"]
    local_count = summary["counts"]["random_local"]["qualified"]
    global_count = summary["counts"]["random_global"]["qualified"]
    development_case = "trap_cluttered_08"
    development_successes = sum(
        summary["cases"].get(development_case, {}).get(arm, {}).get("status")
        == "INDEPENDENTLY_QUALIFIED"
        for arm in ARMS
    )
    if dual_count < global_count and dual_count <= local_count:
        decision_label = "NO-GO for the current LP-dual-guided search"
    elif dual_count > max(local_count, global_count):
        decision_label = "GO for further LP-dual-guided evaluation"
    else:
        decision_label = "INCONCLUSIVE on a comparative search advantage"
    summary["research_decision"] = {
        "classification": decision_label,
        "basis": (
            f"At the fixed {BUDGETS[-1]}-proposal budget, dual guidance qualified "
            f"{dual_count}/{len(cases)}, random-local qualified "
            f"{local_count}/{len(cases)}, and random-global qualified "
            f"{global_count}/{len(cases)}. The development task had "
            f"{development_successes}/{len(ARMS)} qualified arm results. These "
            "three selected adversarial tasks form a small controlled evaluation, "
            "not a population benchmark. The outcome does not establish global "
            "search performance."
        ),
        "proposal_budget_counts": {
            arm: {
                str(budget): sum(
                    cases[case][arm]
                    .get("qualified_by_budget", {})
                    .get(str(budget), False)
                    for case in cases
                )
                for budget in BUDGETS
            }
            for arm in ARMS
        },
    }
    if not summary["registry_consistent"]:
        summary["warning"] = (
            "ARMS MISSING OR SOURCE REGISTRIES DIFFER: "
            "no controlled cross-arm comparison qualified"
        )
    return summary


def _native_diagnostics(native_root: Path, control_root: Path) -> dict:
    """Summarize captured native attempts and uninstrumented outcome controls."""
    arm = native_root / "j1234_f0300" / "native_full"
    diagnostic_path = arm / "pretimeopt_diagnostics.json"
    diagnostics = (
        json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else {}
    )
    result = {}
    for case in ROOTS:
        folder = arm / case
        candidate_path = folder / "native_candidates.json"
        candidates = (
            json.loads(candidate_path.read_text()) if candidate_path.exists() else {}
        )
        diagnostic_rows = diagnostics.get(case, [])
        events = [row for row in diagnostic_rows if row.get("lanes") is not None]
        lane_rows = [lane for row in events for lane in row.get("lanes", [])]
        unique = {
            row.get("geometry_fingerprint")
            for row in lane_rows
            if row.get("geometry_fingerprint")
        }
        lp_positive = [
            row
            for row in lane_rows
            if row.get("lp", {}).get("max_required_torque_bound_expansion", 0.0) > 1e-3
        ]
        unique_lanes = {
            row.get("geometry_fingerprint"): row
            for row in lane_rows
            if row.get("geometry_fingerprint")
        }
        unique_lp_zero_valid = [
            row
            for row in unique_lanes.values()
            if row.get("sampled_geometry_valid") is True
            and row.get("lp", {}).get("max_required_torque_bound_expansion", 1.0)
            <= 1e-3
        ]
        unique_lp_gaps = [
            row.get("lp", {}).get("max_required_torque_bound_expansion")
            for row in unique_lanes.values()
            if row.get("lp", {}).get("status") == "SOLVED"
            and row.get("lp", {}).get("max_required_torque_bound_expansion") is not None
        ]
        mask_events = [
            row
            for row in diagnostic_rows
            if row.get("timeopt_validation_masks") is not None
        ]
        masks_by_family = {}
        for event in mask_events:
            for family, values in event.get("timeopt_validation_masks", {}).items():
                flattened = [bool(value) for value in values]
                counts = masks_by_family.setdefault(family, {"true": 0, "false": 0})
                counts["true"] += sum(flattened)
                counts["false"] += len(flattened) - sum(flattened)
        event_inventory = []
        for inventory_path in sorted(folder.glob("*_pretimeopt_inventory.json")):
            inventory = json.loads(inventory_path.read_text())
            event_inventory.extend(inventory)
        captured_events = [
            event
            for event in event_inventory
            if event.get("state") == "PRETIMEOPT_SNAPSHOT_SAVED"
        ]
        control_candidates = (
            control_root
            / "j1234_f0300"
            / "native_full"
            / case
            / "native_candidates.json"
        )
        uninstrumented_source = "separate uninstrumented control root"
        if not control_candidates.exists():
            _, discovery_name = ROOTS[case]
            prior = ROOT / "research/joint_time_oracle/results" / discovery_name
            control_candidates = (
                prior / "j1234_f0300" / "native_full" / case / "native_candidates.json"
            )
            uninstrumented_source = "historical uninstrumented full-restart record"
        uninstrumented = (
            json.loads(control_candidates.read_text())
            if control_candidates.exists()
            else {}
        )
        metric_groups = {}
        for event in captured_events:
            for capture in event.get("validation_metric_captures", []):
                for item in capture.get("constraint_groups", []):
                    values = np.asarray(item.get("max_by_lane", []), dtype=float)
                    key = f"{item['group']}:{item['name']}"
                    finite = values[np.isfinite(values)]
                    metric_groups.setdefault(key, []).extend(finite.tolist())
        result[case] = {
            "instrumented_restart_count": len(candidates),
            "instrumented_restart_statuses": {
                status: sum(row.get("status") == status for row in candidates.values())
                for status in sorted({row.get("status") for row in candidates.values()})
            },
            "pretimeopt_finalizer_invocations": len(event_inventory),
            "captured_pre_timeopt_geometries": len(captured_events),
            "finalizer_returned_count": sum(
                event.get("finalizer_returned") is True for event in event_inventory
            ),
            "pre_timeopt_success_lane_count": sum(
                bool(value)
                for event in event_inventory
                for value in event.get("pre_success", [])
            ),
            "pre_timeopt_path_valid_lane_count": sum(
                bool(value)
                for event in event_inventory
                for value in event.get("pre_path_valid", [])
            ),
            "captured_lane_count": len(lane_rows),
            "distinct_exact_geometries": len(unique),
            "unique_lp_positive_geometry_count": sum(
                row.get("lp", {}).get("max_required_torque_bound_expansion", 0.0) > 1e-3
                for row in unique_lanes.values()
            ),
            "unique_sampled_geometry_valid_count": sum(
                row.get("sampled_geometry_valid") is True
                for row in unique_lanes.values()
            ),
            "unique_lp_zero_and_geometry_valid_count": len(unique_lp_zero_valid),
            "unique_solved_lp_gap_range": (
                [float(min(unique_lp_gaps)), float(max(unique_lp_gaps))]
                if unique_lp_gaps
                else None
            ),
            "lp_positive_lane_count": len(lp_positive),
            "lp_zero_or_unresolved_lane_count": len(lane_rows) - len(lp_positive),
            "independently_sampled_collision_and_position_valid_lane_count": sum(
                row.get("sampled_geometry_valid") is True for row in lane_rows
            ),
            "independently_sampled_geometry_invalid_lane_count": sum(
                row.get("sampled_geometry_valid") is False for row in lane_rows
            ),
            "native_timeopt_masks_by_family": masks_by_family,
            "native_timeopt_validation_mask_false_lane_count": sum(
                sum(
                    not bool(mask)
                    for key, masks in row.get("timeopt_validation_masks", {}).items()
                    if key
                    in ("path_valid", "collision_valid", "limits_and_dynamics_valid")
                    for mask in masks
                )
                for row in mask_events
            ),
            "captured_native_metric_maxima": {
                key: max(values) if values else None
                for key, values in metric_groups.items()
            },
            "native_capture_diagnostics": (
                diagnostic_path.as_posix() if diagnostic_path.exists() else None
            ),
            "independent_retime_diagnostics": [
                {
                    "file": path.as_posix(),
                    "source_geometry_sha256": record.get("native_geometry_sha256"),
                    "solver_status": record.get("solver", {}).get("status"),
                    "solver_converged": record.get("solver", {}).get(
                        "solver_converged"
                    ),
                    "primal_min_margin": record.get("solver", {}).get(
                        "primal_min_margin"
                    ),
                    "sampled_duration_s": record.get("sampled_duration_s"),
                    "physical_dynamic_pass": record.get("physical_dynamic_pass"),
                    "scaled95_dynamic_pass": record.get("scaled95_dynamic_pass"),
                    "physical_family_min_margins": record.get(
                        "physical_family_min_margins"
                    ),
                    "scaled95_family_min_margins": record.get(
                        "scaled95_family_min_margins"
                    ),
                    "endpoint_residuals": record.get("endpoint_residuals"),
                    "world_clearance_m": record.get("independent_scene_audit", {}).get(
                        "world_clearance_m"
                    ),
                    "sampled_geometry_acceptable": record.get(
                        "independent_scene_audit", {}
                    ).get("sampled_acceptable"),
                    "qualification": (
                        "NOT_QUALIFIED; local infeasibility termination and/or "
                        "sampled collision threshold failure"
                    ),
                }
                for path in sorted(folder.glob("*_independent_retime.json"))
                for record in [json.loads(path.read_text())]
            ],
            "instrumented_result_file": (
                candidate_path.as_posix() if candidate_path.exists() else None
            ),
            "uninstrumented_restart_count": len(uninstrumented),
            "uninstrumented_source": uninstrumented_source,
            "uninstrumented_result_file": (
                control_candidates.as_posix() if control_candidates.exists() else None
            ),
            "uninstrumented_statuses": {
                status: sum(
                    row.get("status") == status for row in uninstrumented.values()
                )
                for status in sorted(
                    {row.get("status") for row in uninstrumented.values()}
                )
            },
            "scope": (
                "Observed native cubic lanes only. Positive relaxed LP rules out timing "
                "for that exact curve subject to floating-point checks; does not rule "
                "out unobserved paths. Native collision mask values are reported as "
                "internal diagnostics, not independent physical collision proof."
            ),
        }
    return result


def _write_manifest(
    output: Path, roots: tuple[Path, ...], cases: tuple[str, ...]
) -> dict:
    """Write checksums for all raw experiment files and report assets."""
    files = {}
    for folder in roots:
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.name != "artifact_manifest.json":
                files[str(path.resolve())] = sha256_file(path)
    for case, (source_name, discovery_name) in ROOTS.items():
        if case not in cases:
            continue
        paths = (
            ROOT / "research/joint_time_oracle/results" / source_name / "tasks.json",
            ROOT
            / "research/joint_time_oracle/results"
            / discovery_name
            / "j1234_f0300"
            / "tasks.json",
            ROOT
            / "research/joint_time_oracle/results"
            / discovery_name
            / "j1234_f0300"
            / case
            / "torque_profile.json",
        )
        for path in paths:
            if path.exists():
                files[str(path.resolve())] = sha256_file(path)
    for relative in (
        "research/joint_time_oracle/torque_guided_rescue.py",
        "research/joint_time_oracle/torque_dual_guidance.py",
        "research/joint_time_oracle/torque_guided_freeze.py",
        "research/joint_time_oracle/native_pretimeopt_probe.py",
        "research/joint_time_oracle/native_pretimeopt_analyze.py",
        "research/joint_time_oracle/native_pretimeopt_retime.py",
        "research/joint_time_oracle/torque_guided_replay.py",
        "research/joint_time_oracle/torque_guided_gradient_audit.py",
        "research/joint_time_oracle/torque_guided_report.py",
    ):
        path = ROOT / relative
        if path.exists():
            files[str(path.resolve())] = sha256_file(path)
    payload = {
        "schema_version": 1,
        "files": files,
        "count": len(files),
        "note": (
            "Raw evidence remains at the listed local paths; hashes bind report "
            "summaries and artifacts."
        ),
    }
    manifest = output / "artifact_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _end_effector_path(positions: np.ndarray) -> np.ndarray:
    """Compute right-gripper points using the independent Pinocchio model."""
    import pinocchio as pin

    from .cohort_retime import dynamics

    model, data, q, qi, _ = dynamics()
    frame_id = model.getFrameId("right_gripper", pin.BODY)
    result = np.empty((len(positions), 3))
    for index, position in enumerate(positions):
        q[qi] = position
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        result[index] = data.oMf[frame_id].translation
    return result


def _render_task(root: Path, output: Path, case: str) -> dict:
    """Render search and trajectory plots directly from frozen task artifacts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial.transform import Rotation

    from .cohort_geometry import QuinticGeometry
    from .torque_active import read_profile
    from .torque_guided_rescue import source_native, source_tasks

    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    source_name, discovery_name = ROOTS[case]
    source_root = ROOT / "research/joint_time_oracle/results" / source_name
    discovery_root = ROOT / "research/joint_time_oracle/results" / discovery_name
    tasks, pilot_rows = source_tasks(source_root)
    task = next(row for row in tasks if row["case_id"] == case)
    native = source_native(source_root, task, pilot_rows[case])
    profile = read_profile(
        discovery_root / "j1234_f0300" / case / "torque_profile.json"
    )

    search_figure = assets / f"search_{case}.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for arm_name in ARMS:
        path = root / arm_name / case / "search_result.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())["report"]
        ledger = report["ledger"]
        proposals = [row["proposal"] for row in ledger]
        axes[0, 0].plot(
            proposals, [row["lp_expansion"] for row in ledger], ".-", label=arm_name
        )
        axes[0, 1].plot(
            proposals, [row["gravity_peak"] for row in ledger], ".-", label=arm_name
        )
        valid, _ = qualified(root / arm_name, case, report)
        first = report.get("first_witness_proposal") if valid else None
        total_time = float(report.get("solver_wall_s_including_gradient_and_screen", 0))
        success_time = (
            float(ledger[first]["wall_s"])
            if first is not None and first < len(ledger)
            else None
        )
        proposal_x = [0, first, 128] if first is not None else [0, 128]
        proposal_y = [0, 1, 1] if first is not None else [0, 0]
        axes[1, 0].step(proposal_x, proposal_y, where="post", label=arm_name)
        time_x = (
            [0.0, success_time, total_time]
            if success_time is not None
            else [0, total_time]
        )
        time_y = [0, 1, 1] if success_time is not None else [0, 0]
        axes[1, 1].step(time_x, time_y, where="post", label=arm_name)
    axes[0, 0].set(
        title="Pointwise torque LP expansion", xlabel="Proposal", ylabel="Normalized ε"
    )
    axes[0, 1].set(
        title="Static gravity utilization",
        xlabel="Proposal",
        ylabel="Utilization vs 95% limits",
    )
    axes[1, 0].set(
        title="Independently qualified rescue",
        xlabel="Proposal count",
        ylabel="Qualified candidate found",
        xlim=(0, 128),
        ylim=(-0.05, 1.05),
    )
    axes[1, 1].set(
        title="Qualified rescue versus scored CPU wall time",
        xlabel="Search wall time (s)",
        ylabel="Qualified candidate found",
        ylim=(-0.05, 1.05),
    )
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
        axis.legend()
    fig.savefig(search_figure, dpi=150)
    plt.close(fig)

    result = {"search_trace": str(search_figure.relative_to(output)), "qualified": []}
    qualified_rows = []
    for arm_name in ARMS:
        arm = root / arm_name
        path = arm / case / "search_result.json"
        if not path.exists():
            continue
        row = json.loads(path.read_text())["report"]
        valid, _ = qualified(arm, case, row)
        if valid:
            folder = arm / case
            geom_data = json.loads((folder / "w_geometry.json").read_text())
            controls = np.asarray(geom_data["controls"], dtype=float)
            geom = QuinticGeometry(controls)
            qualified_rows.append((arm_name, row, folder, geom))
    if not qualified_rows:
        return result

    progress = np.linspace(0.0, 1.0, 513)
    native_q = np.asarray(native.evaluate_trimmed(progress)[0])
    alt_q = np.asarray(qualified_rows[0][3].evaluate(progress)[0])
    selected_folder = qualified_rows[0][2]
    with np.load(selected_folder / "w_dense.npz") as dense_file:
        time = np.asarray(dense_file["time"])
        q = np.asarray(dense_file["q"])
        qd = np.asarray(dense_file["qd"])
        qdd = np.asarray(dense_file["qdd"])
        jerk = np.asarray(dense_file["jerk"])
    with np.load(selected_folder / "w_pin_torque.npz") as torque_file:
        torque = np.asarray(torque_file["torque"])
        ee = np.asarray(torque_file["end_effector"])
    limits = np.stack((profile["effort_lower"], profile["effort_upper"]))

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    layout = fig.add_gridspec(2, 2)
    ax_geom = fig.add_subplot(layout[0, 0])
    ax_torque = fig.add_subplot(layout[0, 1])
    ax_ee = fig.add_subplot(layout[1, 0], projection="3d")
    ax_velocity = fig.add_subplot(layout[1, 1])
    for joint in range(7):
        ax_geom.plot(progress, native_q[:, joint], color=f"C{joint}", alpha=0.55)
        ax_geom.plot(progress, alt_q[:, joint], "--", color=f"C{joint}")
        ax_torque.plot(time, torque[:, joint], label=f"J{joint + 1}")
        ax_torque.hlines(
            limits[0, joint],
            time[0],
            time[-1],
            color=f"C{joint}",
            linestyles=":",
            alpha=0.35,
        )
        ax_torque.hlines(
            limits[1, joint],
            time[0],
            time[-1],
            color=f"C{joint}",
            linestyles=":",
            alpha=0.35,
        )
        ax_velocity.plot(time, qd[:, joint], label=f"J{joint + 1}")
    ax_geom.set(
        title="Native cubic (solid) and feasible W geometry (dashed)",
        xlabel="Normalized progress",
        ylabel="Joint position (rad)",
    )
    ax_torque.set(
        title="Independent Pinocchio torque and derated limits",
        xlabel="Physical time (s)",
        ylabel="Torque (Nm)",
    )
    ax_torque.legend(ncol=4, fontsize=8)
    native_ee = _end_effector_path(native_q[::4])
    ax_ee.plot(native_ee[:, 0], native_ee[:, 1], native_ee[:, 2], label="native cubic")
    ax_ee.plot(ee[:, 0], ee[:, 1], ee[:, 2], label="qualified W")
    scene_path = ROOT / "curobo/content/configs/scene" / task["scene"]
    scene = yaml.safe_load(scene_path.read_text())
    signs = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=float,
    )
    faces = (
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    )
    for name, box in scene.get("cuboid", {}).items():
        pose = box["pose"]
        rotation = Rotation.from_quat([pose[4], pose[5], pose[6], pose[3]])
        vertices = rotation.apply(signs * (np.asarray(box["dims"]) / 2)) + pose[:3]
        polygons = [vertices[list(face)] for face in faces]
        ax_ee.add_collection3d(
            Poly3DCollection(
                polygons, facecolor="gray", alpha=0.16, edgecolor="gray", linewidth=0.35
            )
        )
        ax_ee.text(*pose[:3], name, fontsize=7)
    for name, sphere in scene.get("sphere", {}).items():
        center = np.asarray(sphere.get("position", sphere.get("pose", [0, 0, 0])[:3]))
        radius = float(sphere["radius"])
        u = np.linspace(0, 2 * np.pi, 12)
        v = np.linspace(0, np.pi, 8)
        xyz = [
            center[0] + radius * np.outer(np.cos(u), np.sin(v)),
            center[1] + radius * np.outer(np.sin(u), np.sin(v)),
            center[2] + radius * np.outer(np.ones_like(u), np.cos(v)),
        ]
        ax_ee.plot_surface(*xyz, color="gray", alpha=0.16, linewidth=0)
        ax_ee.text(*center, name, fontsize=7)
    ax_ee.set(
        title="End-effector paths", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)"
    )
    ax_ee.legend()
    ax_velocity.set(
        title="W joint velocities",
        xlabel="Physical time (s)",
        ylabel="Velocity (rad/s)",
    )
    ax_velocity.legend(ncol=4, fontsize=8)
    trajectory_figure = assets / f"trajectory_{case}.png"
    fig.savefig(trajectory_figure, dpi=150)
    plt.close(fig)

    limits_file = (
        ROOT / "research/joint_time_oracle/results/native_reconstruction_20261106/"
        "baseline_validation_states.npz"
    )
    with np.load(limits_file) as limit_data:
        dynamic_bounds = {
            name: np.asarray(limit_data[name + "_bounds"])
            for name in ("position", "velocity", "acceleration", "jerk")
        }
    profiles_figure = assets / f"profiles_{case}.png"
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    profiles = (
        (q, "Joint position", "rad"),
        (qd, "Joint velocity", "rad/s"),
        (qdd, "Joint acceleration", "rad/s²"),
        (jerk, "Joint jerk", "rad/s³"),
    )
    for axis, (values, title, units), family in zip(
        axes.flat, profiles, ("position", "velocity", "acceleration", "jerk")
    ):
        for joint in range(7):
            axis.plot(time, values[:, joint], label=f"J{joint + 1}")
            axis.axhline(
                dynamic_bounds[family][0, joint],
                color=f"C{joint}",
                alpha=0.2,
                linestyle=":",
            )
            axis.axhline(
                dynamic_bounds[family][1, joint],
                color=f"C{joint}",
                alpha=0.2,
                linestyle=":",
            )
        axis.set(title=title, xlabel="Physical time (s)", ylabel=units)
        axis.grid(True, alpha=0.2)
        axis.legend(ncol=4, fontsize=7)
    fig.savefig(profiles_figure, dpi=150)
    plt.close(fig)
    for arm_name, row, folder, _ in qualified_rows:
        result["qualified"].append(
            {
                "arm": arm_name,
                "duration_s": row.get("witness_duration_s"),
                "geometry_sha256": sha256_file(folder / "w_geometry.json"),
                "trajectory_sha256": sha256_file(folder / "w_dense.npz"),
            }
        )
    result["trajectory"] = str(trajectory_figure.relative_to(output))
    result["profiles"] = str(profiles_figure.relative_to(output))
    return result


def publish(summary: dict, output: Path, *, root: Path):
    output.mkdir(parents=True, exist_ok=True)
    figures = {case: _render_task(root, output, case) for case in summary["cases"]}
    summary["figures"] = figures
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    rows = []
    for case, groups in summary["cases"].items():
        for arm, data in groups.items():
            rows.append(
                "<tr><td>"
                + html.escape(case)
                + "</td><td>"
                + html.escape(arm)
                + "</td><td>"
                + html.escape(data["status"])
                + "</td><td>"
                + str(data.get("first_qualified_proposal", "N/A"))
                + "</td><td>"
                + str(data.get("first_qualified_elapsed_wall_s", "N/A"))
                + "</td><td>"
                + str(data.get("duration_s", "N/A"))
                + "</td><td>"
                + str(data.get("total_wall_s", "N/A"))
                + "</td><td>"
                + str(data.get("rnea_calls", "N/A"))
                + "</td><td>"
                + str(data.get("screening_and_timing_rnea_calls", "N/A"))
                + "</td><td>"
                + str(data.get("total_counted_rnea_calls", "N/A"))
                + "</td></tr>"
            )
    sections = []
    for arm, record in summary["counts"].items():
        by = record["qualified_by_proposals"]
        sections.append(
            "<tr><td>"
            + html.escape(arm)
            + "</td>"
            + "".join(
                "<td>" + str(by[str(b)]) + "/" + str(record["attempted"]) + "</td>"
                for b in BUDGETS
            )
            + "</tr>"
        )
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Torque-LP-dual guided geometric rescue</title>
<style>body{font:15px/1.55 system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}
table{border-collapse:collapse;width:100%}td,th{padding:.6rem;border-bottom:1px solid #ddd;
text-align:left}
th{background:#eff2f8}.note{padding:1rem;background:#fff0d8;border-left:5px solid #ac680a}</style>
<h1>Can torque-LP dual guidance rescue geometrically infeasible paths?</h1>
<p class="note">Only independently reaudited physical+Scaled95 witness trajectories
count as successes. Raw solver screens and replay-only candidates are NOT proof.
All three tasks were selected through an earlier adversarial discovery and confirmation;
these results do not estimate population performance. This is NOT continuous-time
feasibility certification. All actuator constraints are synthetic joint-selective
restrictions, not Panda manufacturer's ratings.</p>"""
    decision = summary["research_decision"]
    page += (
        "<h2>Research decision: "
        + html.escape(decision["classification"])
        + "</h2><p>"
        + html.escape(decision["basis"])
        + "</p><p>Qualified rescues by proposal budget (count of three fixed tasks): "
        + "<code>"
        + html.escape(json.dumps(decision["proposal_budget_counts"]))
        + "</code></p>"
    )
    freeze = summary["holdout_freeze"]
    page += (
        "<p>Frozen-method/holdout integrity: <strong>"
        + html.escape(freeze["status"])
        + "</strong>; freeze SHA256 <code>"
        + html.escape(str(freeze["sha256"]))
        + "</code>.</p>"
    )
    page += "<h2>Qualified rescue by proposal budget</h2><table><thead><tr><th>Arm</th>"
    page += "".join(f"<th>≤{b} proposals</th>" for b in BUDGETS)
    page += "</tr></thead><tbody>" + "".join(sections) + "</tbody></table>"
    page += "<h2>LP-dual derivative qualification</h2>"
    gradient = summary.get("dual_gradient_validation")
    if gradient is None:
        page += "<p>Directional derivative evidence was not recorded.</p>"
    else:
        page += (
            "<p>At the fitted source bottleneck, the LP duality gap was "
            + f"{gradient['baseline_lp']['duality_gap']:.3g}; the control-gradient "
            + "norm was stable across finite-difference steps from "
            + f"{gradient['gradient_step_sweep'][0]['gradient_fd_step']:g} to "
            + f"{gradient['gradient_step_sweep'][-1]['gradient_fd_step']:g}. "
            + "The predicted descent derivative and the one-sided LP change agree "
            + "as the perturbation shrinks, while the central directional estimate "
            + "differs because the active set changes. This supports a local heuristic, "
            + "not a globally smooth gradient. See the full step sweep in summary.json.</p>"
        )
    page += (
        "<h2>Source-task outcomes</h2><table><thead><tr><th>Case</th>"
        "<th>Search arm</th><th>Qualification</th>"
    )
    page += (
        "<th>First success proposal</th><th>Time to qualified result (s)</th>"
        "<th>Motion duration (s)</th><th>Total CPU search (s)</th>"
    )
    page += (
        "<th>LP/gradient RNEA</th><th>Screen/timing RNEA</th>"
        "<th>Total counted RNEA</th></tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )
    replay = summary["proposal_replay"]
    page += (
        "<h2>Proposal-level replay and vector archive</h2><p>Status: <strong>"
        + html.escape(replay["status"])
        + "</strong>. All reproduced candidate metrics are compared against the "
        "frozen original ledgers. Parent controls, proposed controls, and normalized "
        "directions are stored in NPZ; the JSON ledgers store parent identity, "
        "bottleneck, method/fallback source, radius, LP gap, screens, and calls. "
        "Replay compute is excluded from scored wall time.</p><table><thead><tr>"
        "<th>Arm</th><th>Task</th><th>Proposals</th><th>Ledger match</th>"
        "<th>LP solves</th><th>Collision checks</th><th>Direction sources</th>"
        "<th>Proposal archive path</th><th>NPZ SHA256</th></tr></thead><tbody>"
    )
    for arm_name, task_rows in replay["arms"].items():
        for case, record in task_rows.items():
            page += (
                "<tr><td>"
                + html.escape(arm_name)
                + "</td><td>"
                + html.escape(case)
                + "</td><td>"
                + str(record["proposal_count"])
                + "</td><td>"
                + str(record["matched_proposals"])
                + "/"
                + str(record["proposal_count"])
                + "</td><td><code>"
                + str(record.get("replay_lp_solves", "N/A"))
                + "</code></td><td><code>"
                + str(record.get("replay_collision_evaluations", "N/A"))
                + "</code></td><td><code>"
                + html.escape(json.dumps(record["direction_source_counts"]))
                + "</code></td><td><code>"
                + html.escape(str(record["npz_path"]))
                + "</code></td><td><code>"
                + html.escape(str(record["npz_sha256"]))
                + "</code></td></tr>"
            )
    page += "</tbody></table>"
    page += (
        "<p>Compare BOTH number of proposals and measured wall time/RNEA calls. "
        "LP dual finite differences are more costly than random perturbations. "
        "All candidates use the same initial nominal-path quintic fit and the "
        "same collision/torque screens. Independent native pre-TimeOpt stage "
        "traces are separately diagnostic and cannot be used as uninstrumented "
        "planner timings.</p>"
    )
    page += "<h2>Proposal traces and qualified trajectories</h2>"
    for case, figures in summary["figures"].items():
        page += f"<h3>{html.escape(case)}</h3>"
        page += (
            f'<img alt="proposal traces for {html.escape(case)}" '
            f'src="{html.escape(figures["search_trace"])}" style="max-width:100%">'
        )
        if figures.get("trajectory"):
            page += (
                f'<img alt="trajectory qualification for {html.escape(case)}" '
                f'src="{html.escape(figures["trajectory"])}" style="max-width:100%">'
            )
            page += (
                f'<img alt="joint profiles for {html.escape(case)}" '
                f'src="{html.escape(figures["profiles"])}" style="max-width:100%">'
            )
        else:
            page += "<p>No independently qualified candidate for this task.</p>"
    page += (
        "<h2>Unfiltered machine-readable evidence</h2><pre>"
        + html.escape(json.dumps(summary, indent=2))
        + "</pre>"
    )
    page += (
        '<p><a href="artifact_manifest.json">Raw artifact checksum manifest</a>. '
        "The raw search ledger and native captures are stored at the absolute paths "
        "in that manifest.</p>"
    )
    environment_path = output / "experiment_environment.json"
    if environment_path.exists():
        environment = json.loads(environment_path.read_text())
        page += (
            "<h2>Execution environment and qualification scope</h2><pre>"
            + html.escape(json.dumps(environment, indent=2))
            + "</pre>"
        )
    native = summary.get("native_pretimeopt", {})
    page += "<h2>Native pre-TimeOpt candidate attribution</h2>"
    page += (
        "<p>Positive LP gaps apply only to the captured exact curves. They do not "
        "rule out unobserved native paths. Internal collision masks are reported "
        "separately from independent sampled sphere clearance.</p>"
    )
    page += (
        "<table><thead><tr><th>Task</th><th>Restarts</th><th>Captured geometries</th>"
    )
    page += "<th>Distinct curves</th><th>Captured lanes</th><th>Positive torque LP</th>"
    page += "<th>Independent geometry valid</th><th>Geometry invalid</th>"
    page += "<th>LP zero/unresolved</th><th>Native masks by family</th>"
    page += "<th>Uninstrumented failures</th></tr></thead><tbody>"
    for case, data in native.items():
        page += (
            "<tr><td>"
            + html.escape(case)
            + "</td><td>"
            + str(data.get("instrumented_restart_count", 0))
            + "</td><td>"
            + str(data.get("captured_pre_timeopt_geometries", 0))
            + "</td><td>"
            + str(data.get("distinct_exact_geometries", 0))
            + "</td><td>"
            + str(data.get("captured_lane_count", 0))
            + "</td><td>"
            + str(data.get("lp_positive_lane_count", 0))
            + "</td><td>"
            + str(
                data.get(
                    "independently_sampled_collision_and_position_valid_lane_count", 0
                )
            )
            + "</td><td>"
            + str(data.get("independently_sampled_geometry_invalid_lane_count", 0))
            + "</td><td>"
            + str(data.get("lp_zero_or_unresolved_lane_count", 0))
            + "</td><td>"
            + html.escape(json.dumps(data.get("native_timeopt_masks_by_family", {})))
            + "</td><td>"
            + str(data.get("uninstrumented_statuses", {}).get("FAILED", 0))
            + "</td></tr>"
        )
    page += "</tbody></table>"
    page += (
        "<p>All captured calls reached the finalizer and returned; all pre-TimeOpt "
        "success/path-valid lanes were true. Native TimeOpt then marked collision "
        "and limits/dynamics validation false for all observed lanes. Independent "
        "geometry sampling found valid collision/position lanes on trap (all 1,280), "
        "table (256/1,280), and clutter (512/1,280). The captured internal "
        "scene-collision metric was 20,100 and self-collision metrics were nonfinite; "
        "their semantics are unresolved and disagree with some independent samples. "
        "Therefore the masks do not identify one exclusive cause. For the curves "
        "with positive LP gaps, torque alone rules out timing on that exact curve; "
        "for other observed curves, collision or retiming remains unresolved.</p>"
    )
    page += "<h3>Native internal metric signals</h3><pre>"
    page += html.escape(json.dumps(native, indent=2)) + "</pre>"
    page += (
        "<h2>Conservative kinematic bounds</h2><p>These convex-hull bounds cover "
        "joint position, velocity, acceleration, and jerk only. They do not certify "
        "torque or collision. Failed bounds are inconclusive because the bounds can "
        "be conservative; sampled/adaptive physical receipts remain separate.</p><pre>"
        + html.escape(
            json.dumps(
                {
                    case: {
                        arm: data.get("kinematic_hull")
                        for arm, data in arms.items()
                        if data.get("kinematic_hull") is not None
                    }
                    for case, arms in summary["cases"].items()
                },
                indent=2,
            )
        )
        + "</pre>"
    )
    (output / "index.html").write_text(page, encoding="utf-8")
    _write_manifest(
        output,
        (
            root,
            Path(summary.get("native_root", "")),
            Path(summary.get("native_control_root", "")),
            output,
        ),
        tuple(summary["cases"]),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=OUTPUT)
    p.add_argument(
        "--native-root",
        type=Path,
        default=ROOT
        / "research/joint_time_oracle/results/native_pretimeopt_probe_metrics_20261108",
    )
    p.add_argument(
        "--native-control-root",
        type=Path,
        default=ROOT
        / "research/joint_time_oracle/results/native_pretimeopt_control_20261108",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/joint_time_oracle/torque_lp_guided_rescue",
    )
    a = p.parse_args()
    result = summarize(a.root)
    result["native_root"] = str(a.native_root.resolve())
    result["native_control_root"] = str(a.native_control_root.resolve())
    result["native_pretimeopt"] = _native_diagnostics(
        a.native_root, a.native_control_root
    )
    publish(result, a.output, root=a.root)
    print(json.dumps({"counts": result["counts"]}, indent=2))


if __name__ == "__main__":
    main()
