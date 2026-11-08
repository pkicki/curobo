# SPDX-License-Identifier: Apache-2.0
"""Fail-closed classification of extreme-torque feasibility witnesses.

A strong qualitative case requires:
  * frozen endpoint-admitted physical constraints;
  * independent exact-quintic dense+adaptive witness Scaled95 pass;
  * ordinary severe-stress full native planner control had a verified success;
  * exactly the registered native-budget attempts completed as FAILED
    (not ERROR), without any registered native candidate SUCCESS.

It establishes algorithmic failure to FIND a solution within that budget,
not mathematical infeasibility of any native geometry or global optimality.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from .extreme_torque_core import check_manifest

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "research/joint_time_oracle/results/extreme_torque_witness_20261108"
CONTROL = (
    ROOT / "research/joint_time_oracle/results/torque_stress_final_20261107/severe"
)
REPORT = ROOT / "reports/joint_time_oracle/extreme_torque"


def original_native_control(control: Path, case: str) -> bool:
    """Return whether the same task had a verified source severe native plan."""
    folder = control / "native_full" / case
    saved = folder / "native_candidates.json"
    receipt = control / "native_full" / "adaptive_receipts.json"
    if not saved.exists() or not receipt.exists():
        return False
    records = json.loads(saved.read_text())
    verified = json.loads(receipt.read_text())
    return any(
        entry.get("status") == "SUCCESS"
        and entry.get("torque_profile_sha256")
        == verified.get(case + ":" + name, {}).get("torque_profile_sha256")
        and verified.get(case + ":" + name, {}).get("physical") is True
        for name, entry in records.items()
    )


def classify_case(
    witness: dict | None,
    control_pass: bool,
    candidates: dict,
    native_receipts: dict,
    *,
    expected_restarts: int = 32,
) -> dict:
    """Never call runtime errors, unverified native success, or missing runs 'failure'."""
    if expected_restarts < 1:
        raise ValueError("expected_restarts must be positive")
    witness_qualified = (
        isinstance(witness, dict)
        and witness.get("physical") is True
        and witness.get("scaled95") is True
    )
    native_parent = {
        f"F{i:03d}": candidates.get(f"F{i:03d}") for i in range(expected_restarts)
    }
    missing = sum(v is None for v in native_parent.values())
    failures = sum(
        isinstance(v, dict) and v.get("status") == "FAILED"
        for v in native_parent.values()
    )
    errors = sum(
        isinstance(v, dict) and v.get("status") == "ERROR"
        for v in native_parent.values()
    )
    successes = sum(
        isinstance(v, dict) and v.get("status") == "SUCCESS"
        for v in native_parent.values()
    )
    candidate_successes = [
        (name, entry)
        for name, entry in candidates.items()
        if entry.get("status") == "SUCCESS"
        and isinstance(entry.get("restart"), int)
        and 0 <= entry["restart"] < expected_restarts
    ]
    verified_native = [
        name
        for name, entry in candidate_successes
        if native_receipts.get(name, {}).get("physical") is True
        and native_receipts[name].get("torque_profile_sha256")
        == entry.get("torque_profile_sha256")
    ]
    # A source-case qualified W is a necessary witness. A prescreen alone never counts.
    if not witness_qualified:
        decision = "NO_INDEPENDENT_FEASIBLE_WITNESS"
    elif missing:
        decision = "INCOMPLETE_NATIVE_BUDGET"
    elif errors:
        decision = "BLOCKED_NATIVE_EXECUTION_ERRORS"
    elif verified_native:
        decision = "NATIVE_FOUND_FEASIBLE_TRAJECTORY"
    elif successes or candidate_successes:
        decision = "NATIVE_RETURNED_UNQUALIFIED_CANDIDATES"
    elif failures == expected_restarts and control_pass:
        decision = "NATIVE_BUDGET_FAILURE_WITH_FEASIBLE_ALTERNATIVE"
    elif failures == expected_restarts:
        decision = "NATIVE_BUDGET_FAILURE_BUT_CONTROL_NOT_VERIFIED"
    else:
        decision = "INDETERMINATE_NATIVE_STATUS"
    return {
        "decision": decision,
        "witness_independent_scaled95": bool(witness_qualified),
        "original_native_severe_control_pass": bool(control_pass),
        "expected_restarts": expected_restarts,
        "native_completed_failed_restarts": failures,
        "native_execution_errors": errors,
        "native_successful_parent_restarts": successes,
        "native_successful_candidate_lanes": len(candidate_successes),
        "native_verified_physical_candidates": len(verified_native),
        "missing_native_restarts": missing,
        "native_verified_names": verified_native,
        "interpretation": (
            "Finite-budget observed native algorithm failure; not geometric "
            "impossibility or global optimality."
        ),
    }


def build(root: Path, control: Path, *, expected_restarts: int = 32) -> dict:
    """Classify every registered factor/task trial from frozen receipts."""
    manifest = json.loads((root / "manifest.json").read_text())
    if not check_manifest(manifest):
        raise ValueError("extreme torque preregistration checksum mismatch")
    outcomes = []
    for regime in sorted(manifest["arms"]):
        arm = root / regime
        attempts = json.loads((arm / "attempt_inventory.json").read_text())
        witness_file = arm / "w_adaptive_receipts.json"
        witnesses = (
            json.loads(witness_file.read_text()) if witness_file.exists() else {}
        )
        native_file = arm / "native_full" / "adaptive_receipts.json"
        native_receipts = (
            json.loads(native_file.read_text()) if native_file.exists() else {}
        )
        for entry in attempts:
            case = entry["case_id"]
            row = {
                "case_id": case,
                "regime": regime,
                "fraction_of_severe_effort_limit": entry["factor"],
                "endpoint_admitted": entry["admitted"],
                "witness_prescreen_pass": entry["witness_screen_pass"],
                "witness_slowdown": entry.get("witness_slowdown"),
                "witness_duration_s": entry.get("witness_duration_s"),
            }
            if not entry["admitted"]:
                row["decision"] = "ENDPOINT_STATIC_GRAVITY_REJECTED"
                row["reason"] = entry.get("reason")
                outcomes.append(row)
                continue
            folder = arm / "native_full" / case
            candidate_file = folder / "native_candidates.json"
            candidates = (
                json.loads(candidate_file.read_text())
                if candidate_file.exists()
                else {}
            )
            native_by_case = {
                k.removeprefix(case + ":"): v
                for k, v in native_receipts.items()
                if k.startswith(case + ":")
            }
            diagnosis = classify_case(
                witnesses.get(case + ":W"),
                original_native_control(control, case),
                candidates,
                native_by_case,
                expected_restarts=expected_restarts,
            )
            row.update(diagnosis)
            lp_file = arm / case / "path_torque_lp.json"
            lp = json.loads(lp_file.read_text()) if lp_file.exists() else {}
            original_lp = lp.get("original_fixed_curobo", {})
            alternative_lp = lp.get("alternative_quintic", {})
            row["original_path_relaxed_torque_lp_infeasible"] = bool(
                original_lp.get("strong_fixed_path_infeasibility_witness") is True
            )
            row["original_path_minimum_relaxed_bound_expansion"] = original_lp.get(
                "max_required_torque_bound_expansion"
            )
            row["original_path_infeasible_progress"] = original_lp.get("worst_progress")
            row["alternative_path_lp_contradicts_witness"] = bool(
                alternative_lp.get("strong_fixed_path_infeasibility_witness") is True
                and diagnosis["witness_independent_scaled95"]
            )
            # Separate mechanism evidence from the primary finite-budget claim.
            # A positive original-path LP gap is evidence THIS curve is
            # infeasible for any retiming, not that all native paths are.
            row["strong_geometric_failure_example"] = bool(
                diagnosis["decision"]
                == "NATIVE_BUDGET_FAILURE_WITH_FEASIBLE_ALTERNATIVE"
                and row["original_path_relaxed_torque_lp_infeasible"]
                and not row["alternative_path_lp_contradicts_witness"]
            )
            if row["alternative_path_lp_contradicts_witness"]:
                row["decision"] = "BLOCKED_CONTRADICTORY_ALTERNATIVE_LP"
                row["strong_geometric_failure_example"] = False
            row["witness_receipt_file"] = (
                str(witness_file) if witness_file.exists() else None
            )
            row["native_candidate_file"] = (
                str(candidate_file) if candidate_file.exists() else None
            )
            outcomes.append(row)
    summary = {
        "schema_version": 1,
        "registered_source_tasks": manifest["arms"],
        "expected_native_restarts": expected_restarts,
        "status": "QUALITATIVE ADAPTIVE-SAMPLED EXPERIMENT — NOT continuous-time safety proof",
        "cases": outcomes,
        "decision_counts": {},
        "note": (
            "Adversarial levels are exploratory and chosen as a frozen grid BEFORE "
            "native outcomes; selected-case examples are not unbiased population statistics."
        ),
    }
    summary["strong_geometric_failure_examples"] = sum(
        x.get("strong_geometric_failure_example") is True for x in outcomes
    )
    for row in outcomes:
        decision = row["decision"]
        summary["decision_counts"][decision] = (
            summary["decision_counts"].get(decision, 0) + 1
        )
    return summary


def publish(summary: dict, out: Path):
    """Write a machine-readable summary and offline HTML report."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    counts = summary["decision_counts"]
    success = counts.get("NATIVE_BUDGET_FAILURE_WITH_FEASIBLE_ALTERNATIVE", 0)
    rows = []
    for row in summary["cases"]:
        label = row["decision"]
        rows.append(
            "<tr><td>" + html.escape(row["case_id"]) + "</td>"
            "<td>" + html.escape(row["regime"]) + "</td>"
            "<td>" + ("yes" if row["endpoint_admitted"] else "no") + "</td>"
            "<td>"
            + ("yes" if row.get("witness_independent_scaled95") else "no")
            + "</td>"
            "<td>"
            + (
                "N/A"
                if row.get("witness_slowdown") is None
                else str(row["witness_slowdown"])
            )
            + "</td>"
            "<td>" + str(row.get("native_completed_failed_restarts", "N/A")) + "</td>"
            "<td>" + html.escape(label) + "</td></tr>"
        )
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Extreme torque: feasibility witness vs native replanning</title>
<style>body{font:15px/1.5 system-ui;max-width:1300px;margin:2rem auto;padding:0 1rem}
table{border-collapse:collapse;width:100%}th,td{padding:.5rem;border-bottom:1px solid #ddd;
text-align:left}
th{background:#eef1f6}.note{padding:1rem;background:#fff0d8;border-left:5px solid #b46c00}</style>
<h1>Can native cuRobo fail while another geometric path remains feasible?</h1>
<p class="note"><strong>Qualification:</strong> only independently validated
exact-quintic adaptive Scaled95 witnesses count. Native failure means no
solution found in the preregistered finite restart budget; it does NOT
prove all native geometries infeasible or any global optimality. Endpoint-
infeasible profiles, missing restarts, planner runtime errors and native
unverified successes are all separate categories. Analytic sphere collision
and sampled Pinocchio checks are not continuous-time certificates.</p>"""
    page += (
        f"<p><strong>{success}</strong> finite-budget native-failure-with-feasible-"
        "alternative cases; "
        f"<strong>{summary['strong_geometric_failure_examples']}</strong> also have "
        "a strictly positive relaxed torque infeasibility LP on the original path.</p>"
    )
    page += "<table><thead><tr><th>Case</th><th>Torque regime</th><th>Endpoint admitted</th>"
    page += "<th>Independent witness</th><th>Witness slowdown</th><th>Native failed restarts</th>"
    page += (
        "<th>Decision</th></tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>"
    )
    page += (
        "<h2>Decision counts</h2><pre>"
        + html.escape(json.dumps(counts, indent=2))
        + "</pre>"
    )
    page += (
        "<p>All factor-case trials, including excluded or unrun cases, are preserved "
        "in the manifest."
    )
    page += (
        " Native negative controls use the previously verified severe baseline on "
        "the same task.</p></html>"
    )
    (out / "index.html").write_text(page, encoding="utf-8")


def main():
    """Build the report from frozen extreme-torque results."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=OUTPUT)
    p.add_argument("--control", type=Path, default=CONTROL)
    p.add_argument("--report", type=Path, default=REPORT)
    p.add_argument("--restarts", type=int, default=32)
    a = p.parse_args()
    summary = build(a.root, a.control, expected_restarts=a.restarts)
    publish(summary, a.report)
    print(
        json.dumps(
            {
                "cases": len(summary["cases"]),
                "decision_counts": summary["decision_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
