# SPDX-License-Identifier: Apache-2.0
"""Aggregate frozen multi-problem joint-time-oracle outcomes.

Input JSONL: one record per attempted, unique task, e.g.
{"case_id":"p0001","cohort":"pilot","population":"table",
 "methods":{"A10":{"duration_s":0.5,"verification":{"physical":true,
 "scaled95":true,"endpoint":true,"dense":true,
 "collision_independent":true,"dynamics_independent":true}}}}

Absent verification fields are NOT passes. This module evaluates statistics;
it does not execute robot optimization or independently validate trajectories.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

METHODS = ("A10", "A40", "B", "C", "D")
LABELS = {
    "A10": "cuRoboV2-Fast TimeOpt default (10 iterations)",
    "A40": "cuRoboV2-Fast TimeOpt 40 iterations",
    "B": "Independent timing on original cuRobo path",
    "C": "Joint path and time",
    "D": "Independent retiming of joint optimizer geometry",
}
PAIRS = (("A10", "A40"), ("A40", "B"), ("B", "C"),
         ("B", "D"), ("A10", "C"), ("A40", "C"))
REQUIRED = ("physical", "endpoint", "dense", "collision_independent",
            "dynamics_independent")


def read_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    if not isinstance(rows, list):
        raise ValueError("input must be a JSON array or JSONL")
    ids = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("every record must be an object")
        for key in ("case_id", "cohort", "population", "methods"):
            if key not in row:
                raise ValueError(f"missing required field {key}")
        if not isinstance(row["case_id"], str) or not row["case_id"]:
            raise ValueError("case_id must be a nonempty string")
        if row["case_id"] in ids:
            raise ValueError(f"duplicate task: {row['case_id']}")
        ids.add(row["case_id"])
        if row["cohort"] not in ("pilot", "confirm"):
            raise ValueError(f"invalid cohort: {row['cohort']}")
        if not isinstance(row["methods"], dict):
            raise ValueError("methods must be an object")
        unknown = set(row["methods"]) - set(METHODS)
        if unknown:
            raise ValueError(f"unknown methods: {sorted(unknown)}")
        for name, result in row["methods"].items():
            if not isinstance(result, dict):
                raise ValueError(f"{name} must contain result metadata")
            duration = result.get("duration_s")
            if duration is not None and (
                not isinstance(duration, (float, int))
                or isinstance(duration, bool)
                or not math.isfinite(duration)
                or duration <= 0
            ):
                raise ValueError(f"invalid {name} duration in {row['case_id']}")
    return rows


def qualified(result: dict | None, mode: str) -> bool:
    """Fail closed: a candidate cannot pass without explicit evidence flags."""
    if not isinstance(result, dict) or result.get("duration_s") is None:
        return False
    checks = result.get("verification")
    if not isinstance(checks, dict):
        return False
    needs = REQUIRED + (("scaled95",) if mode == "scaled95" else ())
    return all(checks.get(flag) is True for flag in needs)


def bootstrap_median(values: list[float], *, seed: int = 3107) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    arr = sorted(statistics.median(rng.choices(values, k=len(values)))
                 for _ in range(2000))
    return [arr[49], arr[1949]]


def analyze(rows: list[dict]) -> dict:
    out = {"schema_version": 1, "total_tasks": len(rows), "groups": {},
           "global_warning": "Local candidates are not global optima. "
               "Sampled validation is not a continuous-time certificate. "
               "Unknown/missing validation flags fail closed."}
    groups = {"all": rows}
    for cohort in sorted({r["cohort"] for r in rows}):
        groups[cohort] = [r for r in rows if r["cohort"] == cohort]
    for pop in sorted({r["population"] for r in rows}):
        groups["population:" + pop] = [r for r in rows if r["population"] == pop]
    for cohort in ("pilot", "confirm"):
        for pop in sorted({r["population"] for r in rows}):
            subset = [r for r in rows
                      if r["cohort"] == cohort and r["population"] == pop]
            if subset:
                groups[cohort + "/" + pop] = subset
    for group, subset in groups.items():
        totals = {"tasks": len(subset), "methods": {}, "comparisons": {}}
        for name in METHODS:
            found = [r["methods"][name] for r in subset if name in r["methods"]]
            totals["methods"][name] = {
                "recorded": len(found),
                "with_duration": sum(f.get("duration_s") is not None for f in found),
                "physical_qualified": sum(qualified(f, "physical") for f in found),
                "scaled95_qualified": sum(qualified(f, "scaled95") for f in found),
                "solver_converged": sum(f.get("solver_converged") is True for f in found),
                "control_bound_active": sum(f.get("control_bound_active") is True
                                            for f in found),
            }
        for mode in ("physical", "scaled95"):
            for a, b in PAIRS:
                reductions = []
                abs_saved = []
                for r in subset:
                    first, second = r["methods"].get(a), r["methods"].get(b)
                    if not qualified(first, mode) or not qualified(second, mode):
                        continue
                    t1, t2 = first["duration_s"], second["duration_s"]
                    reductions.append(100 * (t1 - t2) / t1)
                    abs_saved.append(t1 - t2)
                n = len(reductions)
                totals["comparisons"][mode + ":" + a + "->" + b] = {
                    "eligible_pairs": n,
                    "unpaired_or_unqualified": len(subset) - n,
                    "median_improvement_pct": statistics.median(reductions) if n else None,
                    "mean_improvement_pct": statistics.mean(reductions) if n else None,
                    "median_absolute_saving_ms": 1000 * statistics.median(abs_saved) if n else None,
                    "median_bootstrap95_pct": bootstrap_median(reductions) if n else None,
                    "improved_count": sum(x > 0 for x in reductions),
                    "slower_count": sum(x < 0 for x in reductions),
                    "improvement_gt_1pct": sum(x > 1 for x in reductions),
                    "improvement_gt_5pct": sum(x > 5 for x in reductions),
                    "improvement_gt_10pct": sum(x > 10 for x in reductions),
                }
        out["groups"][group] = totals
    return out


def fmt(v, digits=3):
    return "—" if v is None else f"{v:.{digits}f}"


def html_report(rows: list[dict], summary: dict) -> str:
    esc = html.escape
    head = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Joint time-oracle: cohort comparison</title>
<style>
body{font:15px/1.5 system-ui,sans-serif;background:#f7f8fb;color:#1a2431;margin:0}
main{max-width:1300px;margin:0 auto;padding:2rem}
h1,h2{letter-spacing:-.03em} .warn{background:#fff1db;border-left:5px solid #b66900;padding:1rem}
table{border-collapse:collapse;width:100%;background:white;margin:1rem 0}
th,td{border-bottom:1px solid #e0e4e9;padding:.55rem;text-align:left}
thead{background:#e7edf5} .scroll{overflow-x:auto}
.pill{display:inline-block;padding:.12rem .5rem;background:#e7ebf0;border-radius:10px}
small{color:#526073} .bad{color:#9b2525}.good{color:#12623b}
</style><main>
<h1>Joint path-and-time optimization: frozen cohort</h1>
<p class="warn"><strong>Interpretation gate:</strong> candidates are local, not globally optimal.
Comparisons require explicit dense and independent validation flags for both trajectories.
Physical and 95%-scaled limits are analyzed separately. Missing, failed, or
unqualified trajectories never count as zero improvement.</p>"""
    parts = [head, f"<p><strong>{len(rows)}</strong> frozen task records. "
             "This page summarizes provided results; it does not run or certify experiments.</p>"]
    for group, stats in summary["groups"].items():
        parts.append("<section><h2>" + esc(group) +
                     f" <small>({stats['tasks']} attempted)</small></h2>")
        parts.append("<div class='scroll'><table><thead><tr>"
                     "<th>Method</th><th>Recorded</th><th>With duration</th>"
                     "<th>Physical-qualified</th><th>Scaled95-qualified</th>"
                     "<th>Converged</th><th>At control bound</th></tr></thead><tbody>")
        for name, data in stats["methods"].items():
            parts.append("<tr><td>" + esc(name) + " — " + esc(LABELS[name]) +
                         "</td>" + "".join(f"<td>{data[k]}</td>" for k in
                         ("recorded","with_duration","physical_qualified",
                          "scaled95_qualified","solver_converged","control_bound_active"))+
                         "</tr>")
        parts.append("</tbody></table></div>")
        parts.append("<div class='scroll'><table><thead><tr><th>Feasibility</th>"
                     "<th>Contrast</th><th>Paired / attempted</th><th>Median (%)</th>"
                     "<th>Bootstrap 95% CI (%)</th><th>Mean (%)</th><th>Faster / slower</th>"
                     "<th>&gt;5% faster</th></tr></thead><tbody>")
        for key, data in stats["comparisons"].items():
            mode, contrast = key.split(":",1)
            ci = data["median_bootstrap95_pct"]
            ci_label = "—" if ci is None else f"[{ci[0]:.2f}, {ci[1]:.2f}]"
            parts.append("<tr><td>" + esc(mode) + "</td><td>" + esc(contrast) +
                         f"</td><td>{data['eligible_pairs']} / {stats['tasks']}</td>"
                         f"<td>{fmt(data['median_improvement_pct'])}</td>"
                         "<td>" + ci_label + "</td>"
                         f"<td>{fmt(data['mean_improvement_pct'])}</td>"
                         f"<td>{data['improved_count']} / {data['slower_count']}</td>"
                         f"<td>{data['improvement_gt_5pct']}</td></tr>")
        parts.append("</tbody></table></div></section>")
    parts.append("<h2>Per-case evidence</h2><div class='scroll'><table>"
                 "<thead><tr><th>ID</th><th>Cohort</th><th>Population</th>" +
                 "".join("<th>" + name + " (s)</th>" for name in METHODS) +
                 "<th>Physical-qualified B→C (%)</th><th>Scaled95-qualified B→C (%)</th>"
                 "</tr></thead><tbody>")
    for row in rows:
        out = "<tr>" + "".join("<td>"+esc(str(row[k]))+"</td>" for k in
                                  ("case_id","cohort","population"))
        for name in METHODS:
            r = row["methods"].get(name)
            out += "<td>" + fmt(r.get("duration_s") if r else None,6)
            if r and r.get("duration_s") is not None:
                out += " <small>(" + ("P" if qualified(r,"physical") else "unqualified") + ")</small>"
            out += "</td>"
        for mode in ("physical","scaled95"):
            a,b=row["methods"].get("B"),row["methods"].get("C")
            value=(100*(a["duration_s"]-b["duration_s"])/a["duration_s"]
                   if qualified(a,mode) and qualified(b,mode) else None)
            out += "<td>"+fmt(value)+"</td>"
        parts.append(out+"</tr>")
    parts.append("</tbody></table></div><p><small>Positive percentages mean the second "
                 "method is faster. P = passed the explicit physical checks. "
                 "A10/A40 compare iteration budgets on the original geometry; "
                 "B is independent timing on original geometry; C optimizes geometry and "
                 "time; D independently retimes C geometry, controlling for timing-law quality. "
                 "B→C alone does not isolate a causal geometric gain.</small></p>")
    parts.append("</main></html>")
    return "\n".join(parts)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,type=Path,
                        help="frozen JSONL or JSON array of attempted tasks")
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(argv)
    rows=read_records(args.input)
    args.output.mkdir(parents=True,exist_ok=True)
    summary=analyze(rows)
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (args.output/"index.html").write_text(html_report(rows,summary),encoding="utf-8")
    print(f"{len(rows)} tasks; wrote {args.output/'summary.json'} and index.html")


if __name__=="__main__":
    main()
