# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Classify cells and compare arms from native, diagnosis, search and validation files.

Pure NumPy/SciPy; run in either environment AFTER ``independent_validate``.

Cell classes (per task x payload):
  ENDPOINT_INADMISSIBLE      static gravity/collision at endpoints (not a planner failure)
  NATIVE_SUCCESS             native success at R1 (validated separately)
  CONFIRMED_HARD_FAILURE     native failed at its top budget AND some arm produced a
                             trajectory that passes independent Scaled95 validation
  UNRESOLVED_FAILURE         native failed; no validated alternative (NOT a proof)
  NATIVE_SUBOPTIMAL          native validated success, and a validated alternative is
                             >= 5% shorter
Arm comparison is PAIRED by source task (cells of one task are averaged first);
budget curves use the search wall time of the first feasible candidate together
with that candidate's validation outcome.
"""

from __future__ import annotations

import argparse
import html
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import read_json, write_json

BUDGETS_S = (0.1, 0.3, 1.0, 3.0, float("inf"))
SUBOPT = 0.05


def _val(path: Path) -> dict | None:
    f = path.with_name(path.name + ".validation.json") if not str(path).endswith(".json") else path
    return read_json(f) if f.exists() else None


def load_campaign(camp: Path) -> list[dict]:
    rows = []
    for summ in sorted(camp.glob("*/*/summary.json")):
        s = read_json(summ)
        d = summ.parent
        native_val = _val(d / "native")
        row = {"task_id": s["task_id"], "payload": s["payload"], "role": s["role"],
               "native_success": s["native"]["success"], "native_duration_s": s["native"]["duration_s"],
               "native_budget": s["native"]["restarts_budget"], "native_validation": native_val,
               "native_wall_s": s["native"]["wall_s"], "arms": {},
               "native_geometry_k107_validation": _val(d / "native_geometry_k107")}
        for key, info in s["arms"].items():
            arm, seed = key.split("_s")
            first_val = _val(d / f"{key}_first")
            best_val = first_val if info.get("best_is_first") else _val(d / f"{key}_best")
            row["arms"].setdefault(arm, []).append({
                "seed": int(seed), "first_feasible_s": info["first_feasible_s"],
                "first_feasible_proposals": info["first_feasible_proposals"],
                "first_valid": bool(first_val and first_val.get("scaled95_pass")),
                "best_valid": bool(best_val and best_val.get("scaled95_pass")),
                "best_duration_s": best_val.get("duration_s") if best_val else None,
                "wall_s": info["wall_s"], "oracle_calls": info["oracle_calls"],
            })
        rows.append(row)
    return rows


def classify(row: dict) -> str:
    valid_alt = [r["best_duration_s"] for runs in row["arms"].values() for r in runs
                 if r["best_valid"] and r["best_duration_s"] is not None]
    if not row["native_success"]:
        return "CONFIRMED_HARD_FAILURE" if valid_alt else "UNRESOLVED_FAILURE"
    nv = row["native_validation"]
    if nv and nv.get("scaled95_pass") and valid_alt and min(valid_alt) <= (1 - SUBOPT) * nv["duration_s"]:
        return "NATIVE_SUBOPTIMAL"
    if nv and not nv.get("scaled95_pass"):
        return "NATIVE_SUCCESS_FAILS_INDEPENDENT_VALIDATION"
    return "NATIVE_SUCCESS"


def arm_metrics(rows: list[dict]) -> dict:
    """Per arm: rescue@budget on native failures, duration ratio on native successes."""
    out = {}
    arms = sorted({a for r in rows for a in r["arms"]})
    fails = [r for r in rows if not r["native_success"]]
    succ = [r for r in rows if r["native_success"] and r["native_validation"]
            and r["native_validation"].get("scaled95_pass")]
    for arm in arms:
        m = {}
        for b in BUDGETS_S:
            per_task = defaultdict(list)
            for r in fails:
                for run in r["arms"].get(arm, []):
                    ok = run["first_valid"] and run["first_feasible_s"] is not None and run["first_feasible_s"] <= b
                    per_task[r["task_id"]].append(float(ok))
            vals = [np.mean(v) for v in per_task.values()]
            m[f"rescue_rate_at_{b:g}s"] = None if not vals else float(np.mean(vals))
        ratios = defaultdict(list)
        for r in succ:
            for run in r["arms"].get(arm, []):
                if run["best_valid"] and run["best_duration_s"]:
                    ratios[r["task_id"]].append(run["best_duration_s"] / r["native_validation"]["duration_s"])
        rv = [np.mean(v) for v in ratios.values()]
        m["native_success_tasks"] = len(rv)
        m["median_duration_ratio_vs_native"] = None if not rv else float(np.median(rv))
        m["fraction_tasks_5pct_shorter"] = None if not rv else float(np.mean(np.asarray(rv) <= 1 - SUBOPT))
        ffp = [run["first_feasible_proposals"] for r in fails for run in r["arms"].get(arm, [])
               if run["first_feasible_proposals"] is not None]
        m["median_first_feasible_proposals"] = None if not ffp else float(np.median(ffp))
        calls = [run["oracle_calls"] for r in rows for run in r["arms"].get(arm, [])]
        walls = [run["wall_s"] for r in rows for run in r["arms"].get(arm, [])]
        m["median_oracle_calls"] = None if not calls else float(np.median(calls))
        m["median_search_wall_s"] = None if not walls else float(np.median(walls))
        out[arm] = m
    return out


def retimer_only(rows: list[dict]) -> dict:
    """K107 on the exact native geometry vs native TimeOpt (same path, both validated)."""
    ratios = [r["native_geometry_k107_validation"]["duration_s"] / r["native_validation"]["duration_s"]
              for r in rows if r["native_success"] and r["native_validation"]
              and r["native_validation"].get("scaled95_pass") and r.get("native_geometry_k107_validation")
              and r["native_geometry_k107_validation"].get("scaled95_pass")]
    return {"cells": len(ratios), "median_ratio": float(np.median(ratios)) if ratios else None,
            "fraction_5pct_shorter": float(np.mean(np.asarray(ratios) <= 0.95)) if ratios else None}


def paired_test(rows: list[dict], a: str, b: str) -> dict:
    """Paired (by task) comparison of best validated duration on cells where both arms succeed."""
    from scipy.stats import wilcoxon

    per = defaultdict(lambda: {a: [], b: []})
    for r in rows:
        for arm in (a, b):
            for run in r["arms"].get(arm, []):
                if run["best_valid"] and run["best_duration_s"]:
                    per[r["task_id"]][arm].append(run["best_duration_s"])
    pairs = [(np.mean(v[a]), np.mean(v[b])) for v in per.values() if v[a] and v[b]]
    if len(pairs) < 6:
        return {"tasks": len(pairs), "note": "too few paired tasks"}
    x, y = np.asarray(pairs).T
    stat = wilcoxon(x, y)
    return {"tasks": len(pairs), "median_ratio": float(np.median(x / y)), "wilcoxon_p": float(stat.pvalue)}


def decide(metrics: dict, tests: dict) -> str:
    """GO: D beats B on rescue at matched full budget or on paired validated duration.
    PARTIAL: D needs fewer proposals to the first feasible candidate but no wall-time
    or outcome gain. NO_GO: same-budget blind search matches or beats feedback."""
    if "D" not in metrics or "B" not in metrics:
        return "INCOMPLETE"
    d, b = metrics["D"], metrics["B"]
    rd, rb = d.get("rescue_rate_at_infs"), b.get("rescue_rate_at_infs")
    better_rescue = rd is not None and rb is not None and rd > rb
    t = tests.get("D_vs_B", {})
    better_time = t.get("median_ratio") is not None and t["median_ratio"] < 1 and t.get("wilcoxon_p", 1) < 0.05
    if better_rescue or better_time:
        return "GO_RESEARCH (development split only; confirm on held-out split)"
    pd_, pb = d.get("median_first_feasible_proposals"), b.get("median_first_feasible_proposals")
    if pd_ is not None and pb is not None and pd_ < pb:
        return "PARTIAL"
    return "NO_GO"


def native_tables(run: Path) -> dict:
    rows = [read_json(f) for f in run.glob("cells/*/*/native_R*.json") if f.stem.count("_") == 1]
    tab = defaultdict(Counter)
    for r in rows:
        key = f"R{r['restarts_budget']}|{r['payload']['label']}|{r['stratum']}"
        tab[key]["cells"] += 1
        tab[key]["success"] += int(r["success"])
        for st in r["failure_stages"]:
            if st:
                tab[key][st] += 1
    return {k: dict(v) for k, v in sorted(tab.items())}


def diagnosis_table(run: Path) -> dict:
    c, agree = Counter(), []
    for f in run.glob("cells/*/*/seed_diagnosis_R*.json"):
        d = read_json(f)
        c.update(d.get("class_counts") or {})
        if d.get("seed_replay_agreement") is not None:
            agree.append(d["seed_replay_agreement"])
    return {"lane_classes": dict(c), "mean_seed_replay_agreement": float(np.mean(agree)) if agree else None}


def write_html(path: Path, summary: dict) -> None:
    def table(d: dict) -> str:
        if not d:
            return "<p>none</p>"
        cols = sorted({k for v in d.values() for k in (v if isinstance(v, dict) else {"value": v})})
        head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
        body = ""
        for k, v in d.items():
            v = v if isinstance(v, dict) else {"value": v}
            body += f"<tr><td>{html.escape(str(k))}</td>" + "".join(
                f"<td>{'' if v.get(c) is None else (f'{v[c]:.3f}' if isinstance(v.get(c), float) else v.get(c))}</td>"
                for c in cols) + "</tr>"
        return f"<table><tr><th></th>{head}</tr>{body}</table>"

    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>TimeOpt-guided planning</title>
<style>body{{font-family:sans-serif;margin:16px;background:#fff;color:#111}}table{{border-collapse:collapse;margin:8px 0}}
td,th{{border:1px solid #ccc;padding:3px 6px;font-size:13px}}</style></head><body>
<h1>TimeOpt-guided geometric planning: {html.escape(summary.get('label', ''))}</h1>
<p><b>Decision:</b> {html.escape(summary['decision'])}</p>
<h2>Cell classes</h2>{table(summary['cell_classes'])}
<h2>Arms</h2>{table(summary['arm_metrics'])}
<h2>Paired tests</h2>{table(summary['paired_tests'])}
<h2>Retimer-only (K107 on exact native geometry)</h2>{table(summary['retimer_only_on_native_geometry'])}
<h2>Native seed diagnosis (lanes)</h2>{table(summary['seed_diagnosis']['lane_classes'])}
<p>Seed replay agreement with native CUDA: {summary['seed_diagnosis']['mean_seed_replay_agreement']}</p>
<h2>Native outcomes</h2>{table(summary['native'])}
<p>All durations are independently validated (Pinocchio, Scaled95, sampled). Sampled checks are not
continuous-time certificates. Development split only unless stated.</p></body></html>"""
    path.write_text(page)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    rows = load_campaign(a.campaign)
    for r in rows:
        r["class"] = classify(r)
    metrics = arm_metrics(rows)
    tests = {f"{x}_vs_{y}": paired_test(rows, x, y) for x, y in (("D", "B"), ("C", "B"), ("D", "C"))}
    summary = {
        "label": a.label, "cells": len(rows),
        "cell_classes": dict(Counter(r["class"] for r in rows)),
        "arm_metrics": metrics, "paired_tests": tests, "decision": decide(metrics, tests),
        "retimer_only_on_native_geometry": retimer_only(rows),
        "native": native_tables(a.run), "seed_diagnosis": diagnosis_table(a.run),
        "rows": rows,
    }
    write_json(a.campaign / "report_summary.json", summary)
    write_html(a.campaign / "report.html", summary)
    print({k: summary[k] for k in ("cells", "cell_classes", "decision")})
    for arm, m in metrics.items():
        print(arm, m)


if __name__ == "__main__":
    main()
