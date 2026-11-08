# SPDX-License-Identifier: Apache-2.0
"""Fail-closed taxonomy for geometry-dependent torque feasibility experiments.

Three distinct statements:
A. Original fixed geometry: a numerically reliable relaxed torque LP gap.
B. Alternative geometry: independently sampled/adaptive Scaled95 feasible.
C. Native full planner: observed success or a completed finite-budget failure.

A+B demonstrates the motivating *geometric* feasibility distinction even
when native planning C succeeds. A+B+C failure is a stricter algorithmic
counterexample. We never call a runtime exception a native planning failure.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from .cohort_retime import path_coefficients
from .geometry_feasibility_core import scenarios, sha256_file
from .geometry_feasibility_discover import OUT, SOURCE, source_native, source_tasks
from .torque_active import read_profile

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/joint_time_oracle/geometry_feasibility"


def classify(
    witness: dict | None,
    lp: dict | None,
    native: dict,
    native_receipts: dict,
    *,
    budget: int = 32,
) -> dict:
    if budget < 1:
        raise ValueError("budget must be >=1")
    witness_good = bool(
        isinstance(witness, dict)
        and witness.get("physical") is True
        and witness.get("scaled95") is True
        and witness.get("collision") is True
        and witness.get("endpoint") is True
    )
    lp_good = bool(
        isinstance(lp, dict)
        and lp.get("status") == "NUMERICALLY_STRONG_ORIGINAL_PATH_INFEASIBILITY"
        and lp.get("numerical_dual_certificate", {}).get(
            "numerically_strong_infeasibility"
        )
        is True
    )
    parents = {f"F{k:03d}": native.get(f"F{k:03d}") for k in range(budget)}
    missing = sum(row is None for row in parents.values())
    failed = sum(
        isinstance(row, dict) and row.get("status") == "FAILED"
        for row in parents.values()
    )
    errors = sum(
        isinstance(row, dict) and row.get("status") == "ERROR"
        for row in parents.values()
    )
    successes = [
        (name, row)
        for name, row in native.items()
        if row.get("status") == "SUCCESS"
        and isinstance(row.get("restart"), int)
        and 0 <= row["restart"] < budget
    ]
    native_ok = [
        name
        for name, row in successes
        if native_receipts.get(name, {}).get("physical") is True
        and native_receipts[name].get("torque_profile_sha256")
        == row.get("torque_profile_sha256")
    ]
    if not witness_good:
        category = "NO_QUALIFIED_ALTERNATIVE"
    elif not lp_good:
        category = "NO_STRONG_ORIGINAL_GEOMETRY_INFEASIBILITY_PROOF"
    elif native_ok:
        category = "GEOMETRY_OBSTRUCTION_BUT_NATIVE_REPLANNED_SUCCESSFULLY"
    elif missing:
        category = "GEOMETRY_OBSTRUCTION_NATIVE_BUDGET_INCOMPLETE"
    elif errors:
        category = "GEOMETRY_OBSTRUCTION_NATIVE_EXECUTION_BLOCKED"
    elif successes:
        category = "GEOMETRY_OBSTRUCTION_NATIVE_RETURNED_UNQUALIFIED_TRAJECTORY"
    elif failed == budget:
        category = "GEOMETRY_OBSTRUCTION_NATIVE_BUDGET_FAILURE"
    else:
        category = "GEOMETRY_OBSTRUCTION_NATIVE_STATUS_INDETERMINATE"
    return {
        "category": category,
        "original_fixed_geometry_numerically_infeasible": lp_good,
        "alternative_independent_scaled95_feasible": witness_good,
        "native_verified_physical": len(native_ok),
        "native_verified_candidate_ids": native_ok,
        "native_success_candidate_count": len(successes),
        "native_failed_parent_restarts": failed,
        "native_execution_errors": errors,
        "native_missing_parent_restarts": missing,
        "native_restart_status_counts": {
            status: sum(
                isinstance(row, dict) and row.get("status") == status
                for row in parents.values()
            )
            for status in ("SUCCESS", "FAILED", "ERROR")
        },
        "native_budget_restarts": budget,
        "scope": (
            "sampled witness, numerical LP, finite planning budget; "
            "not global or continuous-time certificate"
        ),
    }


def verified_nominal_native_control(source: Path, case: str) -> bool:
    rowsfile = source / "pilot.jsonl"
    auditfile = source / "independent_sampled.json"
    if not rowsfile.exists() or not auditfile.exists():
        return False
    rows = {
        row["case_id"]: row
        for row in (
            json.loads(line)
            for line in rowsfile.read_text().splitlines()
            if line.strip()
        )
    }
    native = rows.get(case, {}).get("methods", {}).get("A10", {})
    if native.get("duration_s") is None or native.get("native_success") is False:
        return False
    check = json.loads(auditfile.read_text()).get(case + ":A10")
    if not isinstance(check, dict) or check.get("samples", 0) < 8193:
        return False
    if (
        check.get("self_min_clearance_m") is None
        or check["self_min_clearance_m"] < -1e-7
    ):
        return False
    world = check.get("world_min_clearance_m")
    if world is not None and world < -1e-7:
        return False
    for family, diag in check.get("families", {}).items():
        physical = diag.get("physical", {})
        violation = physical.get("violation")
        if violation is None or violation > 1e-4:
            return False
    return all(
        k in check.get("families", {})
        for k in ("position", "velocity", "acceleration", "jerk", "torque")
    )


def attempt_category(item: dict) -> str:
    """Keep endpoint, LP, and candidate-search attrition distinct."""
    if item.get("status") == "NOMINAL_SOURCE_UNAVAILABLE":
        return "NOMINAL_SOURCE_UNAVAILABLE"
    if item.get("endpoint_admitted") is not True:
        return "ENDPOINT_REJECTED"
    status = item.get("status")
    if status == "ORIGINAL_LP_NOT_INFEASIBLE":
        return "ORIGINAL_LP_NOT_INFEASIBLE"
    if status == "POSITIVE_ORIGINAL_LP_NEEDS_ALTERNATIVE":
        return "NO_CONSTRUCTIVE_ALTERNATIVE_DISCOVERED"
    if status == "CONSTRUCTIVE_WITNESS_PRESCREEN":
        return "WITNESS_REQUIRES_QUALIFICATION"
    return "EXPERIMENT_INCOMPLETE"


def collect(root: Path, source: Path, *, budget: int = 32) -> dict:
    registration = json.loads((root / "discovery_registration.json").read_text())
    allowed = {x["scenario_id"] for x in scenarios()}
    chosen = registration.get("scenarios", [])
    if (
        not chosen
        or not all(
            isinstance(x, dict) and x.get("scenario_id") in allowed for x in chosen
        )
        or len({x["scenario_id"] for x in chosen}) != len(chosen)
    ):
        raise ValueError("registered scenario grid mismatch")
    summary = json.loads((root / "discovery_summary.json").read_text())
    source_task_list, source_rows = source_tasks(source)
    source_task_by_case = {task["case_id"]: task for task in source_task_list}
    geometry_cache = {}
    sampled = {}
    source_audit = source / "independent_sampled.json"
    if source_audit.exists():
        sampled = json.loads(source_audit.read_text())
    outcomes = []
    for scenario in registration["scenarios"]:
        name = scenario["scenario_id"]
        arm = root / name
        ledger = arm / "attempt_inventory.json"
        if not ledger.exists():
            outcomes.append({"scenario": name, "status": "NOT_SCANNED"})
            continue
        attempts = json.loads(ledger.read_text())
        wfile = arm / "w_adaptive_receipts.json"
        witnesses = json.loads(wfile.read_text()) if wfile.exists() else {}
        nfile = arm / "native_full" / "adaptive_receipts.json"
        nreceipts = json.loads(nfile.read_text()) if nfile.exists() else {}
        for item in attempts:
            case = item["case_id"]
            row = {
                "scenario": name,
                "case_id": case,
                "discovery_status": item["status"],
                "endpoint_admitted": item["endpoint_admitted"],
                "original_prescreen_lp": item.get("original_relaxed_lp", {}).get(
                    "max_required_torque_bound_expansion"
                ),
                "candidate_proposals": len(item.get("candidate_attempts", [])),
                "constructive_candidate_duration_s": item.get("chosen_duration_s"),
            }
            folder = arm / case
            refined_file = folder / "refined_original_path_lp.json"
            lp = json.loads(refined_file.read_text()) if refined_file.exists() else {}
            row["original_lp_dual_lower_bound"] = lp.get(
                "numerical_dual_certificate", {}
            ).get("dual_lower_bound")
            if lp.get("worst_progress") is not None:
                if case not in geometry_cache:
                    geometry_cache[case] = source_native(
                        source, source_task_by_case[case], source_rows[case]
                    )
                tangent = geometry_cache[case].evaluate_trimmed(
                    np.array([lp["worst_progress"]])
                )[1]
                row["lp_witness_progress"] = lp["worst_progress"]
                row["lp_witness_path_derivative_norm"] = float(np.linalg.norm(tangent))
                path_sample = np.asarray(
                    geometry_cache[case].evaluate_trimmed(
                        np.array([lp["worst_progress"]])
                    )
                )
                coefficients = path_coefficients(path_sample)[:, 0, :]
                accel_coeff, speed_coeff, gravity_coeff = coefficients
                profile = read_profile(folder / "torque_profile.json")
                effort = np.stack((profile["effort_lower"], profile["effort_upper"]))
                halfwidth = (effort[1] - effort[0]) / 2.0
                lp_matrix = np.vstack(
                    (
                        np.column_stack((speed_coeff, accel_coeff, -halfwidth)),
                        np.column_stack((-speed_coeff, -accel_coeff, -halfwidth)),
                    )
                )
                row_norm = np.linalg.norm(lp_matrix, axis=1, keepdims=True)
                row_scaled = lp_matrix / np.maximum(row_norm, 1e-30)
                col_norm = np.linalg.norm(row_scaled, axis=0, keepdims=True)
                scaled_matrix = row_scaled / np.maximum(col_norm, 1e-30)
                certificate = lp.get("numerical_dual_certificate", {})
                row["lp_matrix_condition_number_raw"] = float(np.linalg.cond(lp_matrix))
                row["lp_matrix_condition_number_row_column_scaled"] = float(
                    np.linalg.cond(scaled_matrix)
                )
                row["lp_coefficients_by_joint"] = {
                    "a_M_qs": accel_coeff.tolist(),
                    "m_M_qss_plus_C": speed_coeff.tolist(),
                    "gravity": gravity_coeff.tolist(),
                    "normalization_halfwidth_Nm": halfwidth.tolist(),
                }
                row["lp_active_joint_bound_sides"] = [
                    {
                        "joint_1based": index % 7 + 1,
                        "side": "upper" if index < 7 else "lower",
                    }
                    for index in certificate.get("active_torque_inequality_ids", [])
                ]
            if item["status"] == "CONSTRUCTIVE_WITNESS_PRESCREEN":
                candidate_file = arm / "native_full" / case / "native_candidates.json"
                native = (
                    json.loads(candidate_file.read_text())
                    if candidate_file.exists()
                    else {}
                )
                matched = {
                    k.removeprefix(case + ":"): v
                    for k, v in nreceipts.items()
                    if k.startswith(case + ":")
                }
                res = classify(
                    witnesses.get(case + ":W"), lp, native, matched, budget=budget
                )
                nominal_ok = verified_nominal_native_control(source, case)
                row["original_nominal_control_verified"] = nominal_ok
                if (
                    res["category"] == "GEOMETRY_OBSTRUCTION_NATIVE_BUDGET_FAILURE"
                    and not nominal_ok
                ):
                    res["category"] = (
                        "GEOMETRY_OBSTRUCTION_NATIVE_FAILURE_CONTROL_UNVERIFIED"
                    )
                row.update(res)
                witness_diag = (
                    json.loads((arm / "independent_sampled.json").read_text()).get(
                        case + ":W", {}
                    )
                    if (arm / "independent_sampled.json").exists()
                    else {}
                )
                row["alternative_world_clearance_m"] = witness_diag.get(
                    "world_min_clearance_m"
                )
                row["alternative_self_clearance_m"] = witness_diag.get(
                    "self_min_clearance_m"
                )
                nominal_diag = sampled.get(case + ":A10", {})
                row["original_world_clearance_m"] = nominal_diag.get(
                    "world_min_clearance_m"
                )
                row["original_self_clearance_m"] = nominal_diag.get(
                    "self_min_clearance_m"
                )
                attempts_for_case = item.get("candidate_attempts", [])
                if attempts_for_case:
                    row["geometry_deviation_rad"] = attempts_for_case[-1].get(
                        "arc_length_aligned_geometry_deviation_rad"
                    )
                row["refined_lp_file"] = (
                    str(refined_file) if refined_file.exists() else None
                )
                row["witness_receipt_file"] = str(wfile) if wfile.exists() else None
                row["native_candidate_file"] = (
                    str(candidate_file) if candidate_file.exists() else None
                )
            else:
                row["category"] = attempt_category(item)
            outcomes.append(row)
    counts = {}
    for row in outcomes:
        key = row.get("category", row.get("status", "UNKNOWN"))
        counts[key] = counts.get(key, 0) + 1
    qualified_rows = [
        row
        for row in outcomes
        if row.get("alternative_independent_scaled95_feasible") is True
        and row.get("original_fixed_geometry_numerically_infeasible") is True
    ]
    taskgen_path = source / "task_generator_manifest.json"
    planning_path = source / "planning_manifest.json"
    source_inventory = {
        "task_count": len(source_task_list),
        "population_task_counts": {
            population: sum(
                task["population"] == population for task in source_task_list
            )
            for population in sorted({task["population"] for task in source_task_list})
        },
        "task_inventory_sha256": sha256_file(source / "tasks.json"),
        "task_generator": (
            json.loads(taskgen_path.read_text()) if taskgen_path.exists() else None
        ),
        "nominal_planning": (
            json.loads(planning_path.read_text()) if planning_path.exists() else None
        ),
        "independent_nominal_a10_qualified_count": sum(
            verified_nominal_native_control(source, task["case_id"])
            for task in source_task_list
        ),
    }
    return {
        "schema_version": 1,
        "registration": registration,
        "discovery_summary": summary,
        "native_budget": budget,
        "counts": counts,
        "cases": outcomes,
        "source_inventory": source_inventory,
        "qualified_cell_count": len(qualified_rows),
        "qualified_source_task_count": len({row["case_id"] for row in qualified_rows}),
        "note": (
            "Source-case/scenario cells are correlated; adversarial discovery "
            "≠ confirmatory prevalence."
        ),
        "independence": (
            "new source-disjoint tasks and fixed scenario selection required "
            "for generalization"
        ),
    }


def collect_study(
    root: Path,
    source: Path,
    *,
    budget: int,
    confirmation_root: Path | None = None,
    confirmation_source: Path | None = None,
) -> dict:
    """Collect discovery evidence and optional prospectively frozen confirmation."""
    data = collect(root, source, budget=budget)
    if confirmation_root is not None:
        if confirmation_source is None:
            raise ValueError("confirmation source inventory is required")
        data["confirmation"] = collect(
            confirmation_root, confirmation_source, budget=budget
        )
    historical_path = ROOT / "reports/joint_time_oracle/extreme_torque/summary.json"
    if historical_path.exists():
        historical = json.loads(historical_path.read_text())
        data["historical_uniform_derating"] = {
            "summary_path": str(historical_path.relative_to(ROOT)),
            "status": historical.get("status"),
            "decision_counts": historical.get("decision_counts", {}),
            "strong_geometric_failure_examples": historical.get(
                "strong_geometric_failure_examples"
            ),
            "factor_summary": historical.get("factor_summary", {}),
            "raw_root": historical.get("raw_root"),
        }
    selection_path = root / "confirmation_registration.json"
    if selection_path.exists():
        data["confirmation_scenario_freeze"] = json.loads(selection_path.read_text())
    return data


def publish(data: dict, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(data, indent=2) + "\n")

    def table_rows(cases: list[dict]) -> str:
        rows = []
        for case in cases:
            active = (
                ", ".join(
                    f"J{item['joint_1based']} {item['side']}"
                    for item in case.get("lp_active_joint_bound_sides", [])
                )
                or "N/A"
            )
            condition = case.get("lp_matrix_condition_number_row_column_scaled")
            lp_detail = active + "; κscaled=" + str(condition or "N/A")
            rows.append(
                "<tr><td>" + html.escape(str(case.get("case_id", "—"))) + "</td>"
                "<td>" + html.escape(case.get("scenario", "—")) + "</td>"
                "<td>" + str(case.get("original_prescreen_lp", "N/A")) + "</td>"
                "<td>" + str(case.get("original_lp_dual_lower_bound", "N/A")) + "</td>"
                "<td>" + html.escape(lp_detail) + "</td>"
                "<td>"
                + str(case.get("lp_witness_path_derivative_norm", "N/A"))
                + "</td>"
                "<td>"
                + str(case.get("original_world_clearance_m", "N/A"))
                + "/"
                + str(case.get("original_self_clearance_m", "N/A"))
                + "; "
                + str(case.get("alternative_world_clearance_m", "N/A"))
                + "/"
                + str(case.get("alternative_self_clearance_m", "N/A"))
                + "</td>"
                "<td>" + str(case.get("geometry_deviation_rad", "N/A")) + "</td>"
                "<td>"
                + str(case.get("constructive_candidate_duration_s", "N/A"))
                + "</td>"
                "<td>"
                + str(case.get("native_failed_parent_restarts", "N/A"))
                + "/"
                + str(case.get("native_missing_parent_restarts", "N/A"))
                + "/"
                + str(case.get("native_execution_errors", "N/A"))
                + "</td>"
                "<td>"
                + html.escape(case.get("category", case.get("status", "unknown")))
                + "</td></tr>"
            )
        return "\n".join(rows)

    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Geometry-dependent torque infeasibility</title>
<style>body{font:15px/1.5 system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}
table{border-collapse:collapse;width:100%}
th,td{border-bottom:1px solid #ddd;padding:.45rem;text-align:left}
th{background:#edf2f7}
.note{background:#fff0d8;border-left:4px solid #b36b00;padding:1rem}</style>
<h1>Geometry-dependent torque infeasibility</h1>
<p class="note">A strong geometry witness requires a positive numerically
dual-validated original fixed-path torque LP AND a different independently
sampled/adaptive Scaled95-feasible trajectory. Native full-planner outcomes
are separate: a finite-budget native failure is NOT proof of global
impossibility. The model uses synthetic joint-selective actuator limits,
not the manufacturer's Panda specifications. Dense/adaptive sampled
validation is not continuous-time certification.</p>"""
    counts = data["counts"]
    geom = sum(v for k, v in counts.items() if k.startswith("GEOMETRY_OBSTRUCTION"))
    failed = counts.get("GEOMETRY_OBSTRUCTION_NATIVE_BUDGET_FAILURE", 0)
    page += (
        f"<p>Qualified discovery cells: {data['qualified_cell_count']} "
        f"across {data['qualified_source_task_count']} distinct source task IDs.</p>"
    )
    page += (
        f"<p><strong>{geom}</strong> qualified geometric distinctions; "
        f"<strong>{failed}</strong> complete native-budget failures with such a distinction.</p>"
    )
    page += "<h2>Discovery cohort: every registered task/scenario attempt</h2><table><thead><tr>"
    page += (
        "<th>Task</th><th>Actuator scenario</th><th>LP prescreen ε</th>"
        "<th>LP dual lower bound</th><th>LP active bounds / scaled condition</th>"
    )
    page += "<th>|qₛ| at LP point</th><th>Original world/self; alternative world/self (m)</th>"
    page += "<th>Geometry deviation (rad)</th>"
    page += "<th>Witness duration (s)</th><th>Native failed/missing/errors</th><th>Outcome</th>"
    page += "</tr></thead><tbody>" + table_rows(data["cases"]) + "</tbody></table>"
    page += (
        "<h2>Discovery outcome counts</h2><pre>"
        + html.escape(json.dumps(counts, indent=2))
        + "</pre>"
    )
    page += "<h2>Frozen discovery source inventory</h2><pre>"
    page += html.escape(json.dumps(data["source_inventory"], indent=2)) + "</pre>"
    confirmation = data.get("confirmation")
    if confirmation:
        page += "<h2>Source-disjoint confirmation cohort</h2>"
        page += (
            "<p>Scenario IDs were frozen from development evidence before "
            "confirmation tasks were planned. "
        )
        page += "This cohort is reported separately from the adversarial discovery scan.</p>"
        page += "<table><thead><tr><th>Task</th><th>Actuator scenario</th><th>LP prescreen ε</th>"
        page += (
            "<th>LP dual lower bound</th><th>LP active bounds / scaled condition</th>"
        )
        page += "<th>|qₛ| at LP point</th>"
        page += "<th>Original world/self; alternative world/self (m)</th>"
        page += "<th>Geometry deviation (rad)</th><th>Witness duration (s)</th>"
        page += (
            "<th>Native failed/missing/errors</th><th>Outcome</th></tr></thead><tbody>"
        )
        page += table_rows(confirmation["cases"]) + "</tbody></table>"
        page += "<h3>Confirmation outcome counts</h3><pre>"
        page += html.escape(json.dumps(confirmation["counts"], indent=2)) + "</pre>"
        page += "<h3>Frozen confirmation source inventory</h3><pre>"
        page += (
            html.escape(json.dumps(confirmation["source_inventory"], indent=2))
            + "</pre>"
        )
    if data.get("confirmation_scenario_freeze"):
        page += "<h3>Confirmation scenario freeze record</h3><pre>"
        page += (
            html.escape(json.dumps(data["confirmation_scenario_freeze"], indent=2))
            + "</pre>"
        )
    historical = data.get("historical_uniform_derating")
    if historical:
        page += "<h2>Historical uniform-derating negative control</h2>"
        page += (
            "<p>This earlier experiment tested uniform torque reduction on "
            "the historical 20-task cohort. "
        )
        page += (
            "It is preserved separately: endpoint gravity rejected many "
            "cells, and no original-path LP obstruction was found. "
        )
        page += (
            "The new joint-selective cases are an adversarial source/task "
            "arm, not a replacement of that result.</p>"
        )
        page += (
            "<pre>"
            + html.escape(
                json.dumps(
                    {
                        "status": historical["status"],
                        "decision_counts": historical["decision_counts"],
                        "strong_geometric_failure_examples": historical[
                            "strong_geometric_failure_examples"
                        ],
                        "summary_path": historical["summary_path"],
                    },
                    indent=2,
                )
            )
            + "</pre>"
        )
        page += "<p>Earlier path/timing studies remain available in the "
        page += "<a href='../torque_mechanisms/index.html'>torque-mechanisms report</a> and "
        page += "<a href='../extreme_torque/index.html'>uniform extreme-torque report</a>.</p>"
    challenge_rows = [
        row
        for section in (data, confirmation)
        if section
        for row in section["cases"]
        if row.get("category") == "GEOMETRY_OBSTRUCTION_NATIVE_BUDGET_FAILURE"
    ]
    completed_failures = sum(
        row.get("native_failed_parent_restarts", 0) for row in challenge_rows
    )
    runtime_errors = sum(
        row.get("native_execution_errors", 0) for row in challenge_rows
    )
    missing_restarts = sum(
        row.get("native_missing_parent_restarts", 0) for row in challenge_rows
    )
    page += (
        "<h2>Native failure interpretation</h2><p>For the "
        f"{len(challenge_rows)} qualified native challenges, the planner "
        f"returned no successful trajectory in {completed_failures} completed "
        f"parent restarts, with {runtime_errors} runtime errors and "
        f"{missing_restarts} missing restarts. The current planner interface "
        "did not expose pre-TimeOpt candidate geometries for these unsuccessful "
        "results, so the internal failure stage cannot be attributed to "
        "geometric search versus native retiming. The independent LP obstruction "
        "applies specifically to the frozen original curve; it does not prove "
        "that all native planner routes are infeasible.</p>"
    )
    sections = [data] + ([confirmation] if confirmation else [])
    all_qualified = [
        row
        for section in sections
        for row in section["cases"]
        if row.get("alternative_independent_scaled95_feasible") is True
        and row.get("original_fixed_geometry_numerically_infeasible") is True
    ]
    strict_failures = [
        row
        for row in challenge_rows
        if row.get("native_execution_errors", 0) == 0
        and row.get("native_missing_parent_restarts", 0) == 0
    ]
    task_ids = sorted({row["case_id"] for row in all_qualified})
    page += (
        "<h2>Research conclusion</h2><p>The frozen data contain "
        f"{len(all_qualified)} qualifying task/scenario cells across "
        f"{len(task_ids)} distinct source task IDs. Each has a positive, "
        "dual-checked relaxed torque LP obstruction on the exact original "
        "curve and a geometrically distinct alternative with independent "
        "sampled physical and Scaled95 receipts. The source-disjoint "
        f"confirmation contributes {confirmation['qualified_cell_count'] if confirmation else 0} "
        "cells. This demonstrates geometry-dependent feasibility for the "
        "tested curves and frozen synthetic limits. The alternative timings "
        "are constructive feasibility witnesses, not minimum-time solutions. "
        f"The full native planner exhausted its registered budget in "
        f"{len(strict_failures)} qualified cases, but the failed calls did not "
        "expose pre-TimeOpt paths, so this does not identify the native "
        "pipeline failure stage or prove global planner infeasibility.</p>"
    )
    visualizations = [
        ("trap_cluttered_08", "assets/trap_cluttered_08.png"),
        (
            "confirm_geom_cluttered_02",
            "confirmation/assets/confirm_geom_cluttered_02.png",
        ),
        ("confirm_geom_table_07", "confirmation/assets/confirm_geom_table_07.png"),
    ]
    present = [
        (name, path) for name, path in visualizations if (output / path).exists()
    ]
    if present:
        page += "<h2>Qualified trajectory examples</h2>"
        for name, path in present:
            page += (
                "<figure><img style='max-width:100%;height:auto' src='"
                + html.escape(path)
                + "' alt='"
                + html.escape(name)
                + "'>"
            )
            page += (
                "<figcaption>"
                + html.escape(name)
                + " — generated from frozen trajectory and Pinocchio torque "
                "artifacts.</figcaption></figure>"
            )
    page += "<h2>Discovery registration and qualification scope</h2><pre>"
    page += (
        html.escape(
            json.dumps(
                {
                    "registration": data["registration"],
                    "confirmation_registration": (
                        confirmation["registration"] if confirmation else None
                    ),
                    "limits": "synthetic joint-selective derating",
                    "validation": (
                        "independent sampled Pinocchio and configured collision "
                        "spheres; no continuous-time proof"
                    ),
                },
                indent=2,
            )
        )
        + "</pre></html>"
    )
    (output / "index.html").write_text(page, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=OUT)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--report", type=Path, default=REPORT)
    p.add_argument("--restarts", type=int, default=32)
    p.add_argument("--confirmation-root", type=Path)
    p.add_argument("--confirmation-source", type=Path)
    a = p.parse_args()
    data = collect_study(
        a.root,
        a.source,
        budget=a.restarts,
        confirmation_root=a.confirmation_root,
        confirmation_source=a.confirmation_source,
    )
    publish(data, a.report)
    print(
        json.dumps(
            {"cases": len(data["cases"]), "categories": data["counts"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
