# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish source-derived torque mechanism and native replanning evidence."""

# ruff: noqa: E501  # Embedded HTML/SVG strings are intentionally kept intact.

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "research/joint_time_oracle/results/torque_stress_final_20261107"
REPORT = ROOT / "reports/joint_time_oracle/torque_mechanisms"


def _fmt(value: float | None, unit: str = "", digits: int = 3) -> str:
    """Format one optional numeric result for the report."""
    return "N/A" if value is None else f"{value:.{digits}f}{unit}"


def _read(path: Path, default: object) -> object:
    """Load JSON evidence if present."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _svg_line_chart(
    x: np.ndarray,
    series: list[tuple[str, np.ndarray, str]],
    title: str,
    x_label: str,
    y_label: str,
    *,
    width: int = 920,
    height: int = 360,
) -> str:
    """Build a self-contained SVG line chart from stored numeric arrays."""
    pad_left, pad_right, pad_top, pad_bottom = 70, 25, 42, 55
    all_y = np.concatenate([np.asarray(y, dtype=float).ravel() for _, y, _ in series])
    finite = all_y[np.isfinite(all_y)]
    if finite.size == 0:
        return "<p>Chart unavailable: no finite samples.</p>"
    y_min, y_max = float(finite.min()), float(finite.max())
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    spread = y_max - y_min
    y_min -= spread * 0.05
    y_max += spread * 0.05
    x = np.asarray(x, dtype=float)
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0

    def px(value: float) -> float:
        return pad_left + (value - x_min) / (x_max - x_min) * (
            width - pad_left - pad_right
        )

    def py(value: float) -> float:
        return (
            height
            - pad_bottom
            - (value - y_min) / (y_max - y_min) * (height - pad_top - pad_bottom)
        )

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="24" text-anchor="middle" font-size="16">{html.escape(title)}</text>',
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = y_min + fraction * (y_max - y_min)
        y_pos = py(value)
        parts.append(
            f'<path d="M {pad_left} {y_pos:.1f} H {width-pad_right}" stroke="#e4e8ed"/>'
        )
        parts.append(
            f'<text x="{pad_left-8}" y="{y_pos+4:.1f}" text-anchor="end" font-size="11">{value:.3g}</text>'
        )
    parts.append(
        f'<path d="M {pad_left} {pad_top} V {height-pad_bottom} H {width-pad_right}" fill="none" stroke="#253447"/>'
    )
    for label, values, color in series:
        values = np.asarray(values, dtype=float)
        usable = min(len(x), len(values))
        if usable < 2:
            continue
        step = max(1, usable // 700)
        indices = list(range(0, usable, step))
        if (usable - 1) % step:
            indices.append(usable - 1)
        segments: list[list[str]] = []
        segment: list[str] = []
        for index in indices:
            if np.isfinite(x[index]) and np.isfinite(values[index]):
                segment.append(
                    f"{px(float(x[index])):.1f},{py(float(values[index])):.1f}"
                )
            elif segment:
                segments.append(segment)
                segment = []
        if segment:
            segments.append(segment)
        for coordinates in segments:
            parts.append(
                f'<polyline points="{" ".join(coordinates)}" fill="none" '
                f'stroke="{color}" stroke-width="1.7"/>'
            )
    legend_y = height - 32
    legend_x = pad_left
    for label, _, color in series:
        item_width = min(240, 42 + len(label) * 6)
        if legend_x + item_width > width - pad_right:
            legend_y = min(height - 16, legend_y + 15)
            legend_x = pad_left
        parts.append(
            f'<path d="M {legend_x} {legend_y} h 20" stroke="{color}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{legend_x+25}" y="{legend_y+4}" font-size="11">{html.escape(label)}</text>'
        )
        legend_x += item_width
    parts.append(
        f'<text x="{width/2}" y="{height-2}" text-anchor="middle" font-size="12">{html.escape(x_label)}</text>'
    )
    parts.append(
        f'<text transform="translate(14 {height/2}) rotate(-90)" text-anchor="middle" font-size="12">{html.escape(y_label)}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a compact accessible HTML table."""
    head = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _case_charts(arm: Path, case: str, chart_dir: Path) -> list[str]:
    """Generate torque-component and path-coordinate SVGs for one source case."""
    folder = arm / "mechanisms" / case
    profile = folder / "torque_components.npz"
    if not profile.exists():
        return []
    with np.load(profile) as trace:
        analysis = _read(folder / "analysis.json", {})
        b_joint = int(analysis["methods"]["B"]["metrics"]["max_torque_joint"])
        d_joint = int(analysis["methods"]["D"]["metrics"]["max_torque_joint"])
        chart_dir.mkdir(parents=True, exist_ok=True)
        name = f"{case}_torque.svg"
        series = []
        colors = {
            "gravity": "#4c78a8",
            "inertia": "#f58518",
            "velocity_bias": "#54a24b",
            "total": "#e45756",
        }
        for method, joint in (("b", b_joint), ("d", d_joint)):
            for component in ("gravity", "inertia", "velocity_bias", "total"):
                series.append(
                    (
                        f"{method.upper()} j{joint+1} {component}",
                        trace[f"{method}_{component}"][:, joint],
                        colors[component],
                    )
                )
        svg = _svg_line_chart(
            trace["b_normalized_time"],
            series,
            f"{case}: RNEA torque components at each path's peak joint",
            "normalized physical time",
            "torque (N·m)",
        )
        (chart_dir / name).write_text(svg, encoding="utf-8")
    result = [
        f'<figure><img src="assets/{name}" alt="Torque-component profiles for {html.escape(case)}"><figcaption>RNEA decomposition; the B and D curves use each path\'s peak-utilization joint.</figcaption></figure>'
    ]

    intervals = folder / "path_torque_intervals.npz"
    if intervals.exists():
        with np.load(intervals) as trace:
            series = []
            for method, color in (("b", "#4c78a8"), ("d", "#e45756")):
                series.extend(
                    [
                        (
                            f"{method.upper()} lower x′",
                            trace[f"{method}_global_xprime_lower"],
                            color,
                        ),
                        (
                            f"{method.upper()} upper x′",
                            trace[f"{method}_global_xprime_upper"],
                            "#54a24b" if method == "b" else "#f58518",
                        ),
                    ]
                )
            name = f"{case}_xprime.svg"
            svg = _svg_line_chart(
                trace["b_s"],
                series,
                f"{case}: torque-allowable squared-speed acceleration intervals",
                "normalized path coordinate s",
                "allowable x′ (s⁻¹)",
            )
            (chart_dir / name).write_text(svg, encoding="utf-8")
        result.append(
            f'<figure><img src="assets/{name}" alt="Torque allowable x prime intervals for {html.escape(case)}"><figcaption>Intersection across joints of allowable x′ intervals from asymmetric actuator limits. The shown feasible region is a path-coordinate diagnostic, not a continuous certificate.</figcaption></figure>'
        )
        with np.load(intervals) as trace:
            common_arc = np.linspace(0.0, 1.0, 401)
            colors = (
                "#4c78a8",
                "#f58518",
                "#54a24b",
                "#e45756",
                "#72b7b2",
                "#b279a2",
                "#ff9da6",
            )
            colors_d = (
                "#9ab6d1",
                "#fac28b",
                "#8ccca7",
                "#f1a3a2",
                "#9ed5d1",
                "#d0a9c4",
                "#ffc5ca",
            )
            series = []
            for method in ("b", "d"):
                q = np.asarray(trace[f"{method}_q"])
                ds = np.linalg.norm(np.diff(q, axis=0), axis=1)
                arc = np.r_[0.0, np.cumsum(ds)]
                if arc[-1] > 0:
                    arc /= arc[-1]
                for joint in range(q.shape[1]):
                    values = np.interp(common_arc, arc, q[:, joint])
                    color = colors[joint] if method == "b" else colors_d[joint]
                    series.append(
                        (f"{method.upper()} joint {joint + 1}", values, color)
                    )
            name = f"{case}_configuration.svg"
            svg = _svg_line_chart(
                common_arc,
                series,
                f"{case}: B/D joint configurations on normalized arc length",
                "normalized joint-space arc length",
                "joint position (rad)",
            )
            (chart_dir / name).write_text(svg, encoding="utf-8")
        result.append(
            f'<figure><img src="assets/{name}" alt="Arc-length-aligned joint configuration comparison for {html.escape(case)}"><figcaption>Each path is independently resampled at uniform normalized joint-space arc length; this view shows the configuration changes associated with the different geometry.</figcaption></figure>'
        )
    return result


def _full_torque_evidence(
    arm: Path, comparison: dict, chart_dir: Path, budget: int = 32
) -> tuple[list[str], list[list[str]]]:
    """Plot independently audited Full/Full-R torque and summarize utilization."""
    cases = (
        comparison.get("results", {})
        .get("adaptive_qualified:physical", {})
        .get("cases", {})
    )
    selected: list[tuple[str, dict, dict, int]] = []
    for case, entry in cases.items():
        rows = entry.get("budget_comparisons", {})
        candidate = rows.get(str(budget), {})
        native, retimed = candidate.get("best_native"), candidate.get("best_matched")
        if native and retimed:
            selected.append((case, native, retimed, budget))
    if not selected:
        return [], []
    pair = max(
        selected,
        key=lambda row: row[1]["duration_s"] - row[2]["duration_s"],
    )
    case, native, retimed, _ = pair
    folder = arm / "native_full" / case
    profile = json.loads((arm / case / "torque_profile.json").read_text())
    lower = np.asarray(profile["effort_lower"], dtype=float)
    upper = np.asarray(profile["effort_upper"], dtype=float)
    common_time_fraction = np.linspace(0.0, 1.0, 2001)
    series = []
    table = []
    palette = (
        "#4c78a8",
        "#f58518",
        "#54a24b",
        "#e45756",
        "#72b7b2",
        "#b279a2",
        "#ff9da6",
    )
    for name, method, shades in (
        (native["method"], "Full", palette),
        (
            retimed["method"],
            "Full-R",
            (
                "#9ab6d1",
                "#fac28b",
                "#8ccca7",
                "#f1a3a2",
                "#9ed5d1",
                "#d0a9c4",
                "#ffc5ca",
            ),
        ),
    ):
        torque_path = folder / f"{name.lower()}_adaptive_pin_torque.npz"
        with np.load(torque_path) as archive:
            time = np.asarray(archive["time"], dtype=float)
            torque = np.asarray(archive["torque"], dtype=float)
        normalized = np.where(torque >= 0.0, torque / upper, torque / (-lower))
        utilization = float(np.max(np.abs(normalized)))
        for joint in range(torque.shape[1]):
            aligned = np.interp(
                common_time_fraction,
                time / time[-1],
                normalized[:, joint],
            )
            series.append((f"{method} j{joint+1}", aligned, shades[joint]))
        table.append(
            [
                html.escape(case),
                method,
                html.escape(name),
                _fmt(
                    native["duration_s"] if method == "Full" else retimed["duration_s"],
                    " s",
                    6,
                ),
                _fmt(utilization, "×", 5),
                str(len(time)),
            ]
        )
    series.extend(
        [
            ("physical +1 limit", np.ones_like(common_time_fraction), "#333333"),
            ("physical −1 limit", -np.ones_like(common_time_fraction), "#666666"),
        ]
    )
    chart_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{case}_full_torque_utilization.svg"
    svg = _svg_line_chart(
        common_time_fraction,
        series,
        f"{case}: independently audited normalized actuator torque",
        "fraction of execution duration",
        "signed torque / applicable effort limit",
    )
    (chart_dir / filename).write_text(svg, encoding="utf-8")
    chart = [
        f'<figure><img src="assets/{filename}" alt="Full and Full-R joint torques normalized by derated effort limits"><figcaption>Best adaptive-verified physical-limit pair at restart budget {budget}; torque arrays come from model-aligned Pinocchio RNEA and are normalized against each joint\'s asymmetric signed effort limits. Curves are aligned by execution-duration fraction; durations are listed separately. Samples are not a continuous-time certificate.</figcaption></figure>'
    ]
    return chart, table


def publish(source: Path, output: Path) -> dict:
    """Build an offline report directly from frozen raw JSON and NPZ evidence."""
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "EXPLORATORY_OFFLINE_ANALYSIS",
        "source_root": str(source),
        "regimes": {},
    }
    sections: list[str] = []
    derivative_path = (
        source.parent / "native_reconstruction_20261106" / "cuda_derivative_sweep.json"
    )
    derivative_evidence = _read(derivative_path, None)
    if isinstance(derivative_evidence, dict):
        entries = derivative_evidence.get("entries", [])
        derivative_rows = []
        for family in ("rnea", "world", "self"):
            family_entries = [row for row in entries if row.get("family") == family]
            for step in derivative_evidence.get("steps", []):
                step_entries = [
                    row for row in family_entries if row.get("step") == step
                ]
                smooth = [
                    row
                    for row in step_entries
                    if not row.get("possible_transition", False)
                ]
                transition_count = len(step_entries) - len(smooth)
                derivative_rows.append(
                    [
                        family,
                        _fmt(step, "", 3),
                        str(len(step_entries)),
                        str(transition_count),
                        _fmt(
                            max(
                                (row.get("abs_error", 0.0) for row in smooth),
                                default=None,
                            ),
                            "",
                            5,
                        ),
                        _fmt(
                            (
                                float(np.median([row["abs_error"] for row in smooth]))
                                if smooth
                                else None
                            ),
                            "",
                            6,
                        ),
                    ]
                )
        sections.append(
            "<section><h2>CUDA derivative qualification</h2><p>Stored step sweep checks "
            "RNEA's full output Jacobian at baseline/random states and world/self collision "
            "max-cost VJPs for free, near-contact, and penetrating states. Entries flagged "
            "as active-set/contact transitions are reported separately because central "
            "differences are not expected to match a single branch derivative there. The "
            "fixed-path B/D solvers use independent Pinocchio and analytic Jacobians; these "
            "CUDA derivatives affect native optimization and are not used as a certificate "
            "for the independent verifier.</p>"
        )
        sections.append(
            _table(
                [
                    "Family",
                    "FD step",
                    "Entries",
                    "Transition flags",
                    "Max abs error away from transitions",
                    "Median abs error",
                ],
                derivative_rows,
            )
        )
        sections.append("</section>")
    for regime in ("moderate", "severe"):
        arm = source / regime
        mechanism_root = arm / "mechanisms"
        mechanism_summary = _read(mechanism_root / "summary.json", {"cases": []})
        items = mechanism_summary.get("cases", [])
        complete = [item for item in items if item.get("status") == "COMPLETE"]
        comparison_path = arm / "native_full" / "full_planner_comparison.json"
        comparison = _read(comparison_path, None)
        summary["regimes"][regime] = {
            "decomposition_attempted": len(items),
            "decomposition_complete": len(complete),
            "native_full_comparison": comparison is not None,
        }
        sections.append(
            f"<section><h2>{html.escape(regime.title())} torque stress</h2>"
        )
        sections.append(
            f"<p>{len(complete)}/{len(items)} B/D cases have independent Pinocchio decomposition. Historical matched-retimer median: <strong>{_fmt(float(np.median([x['B_to_D_improvement_pct'] for x in complete])) if complete else None, '%')}</strong>. These are synthetic uniform actuator deratings, not a payload experiment.</p>"
        )
        if complete:
            path_delta = np.asarray(
                [item["path_length_increase_pct"] for item in complete]
            )
            gravity_delta = np.asarray(
                [item["gravity_rms_delta_Nm"] for item in complete]
            )
            inertia_delta = np.asarray(
                [item["inertia_rms_delta_Nm"] for item in complete]
            )
            positive = sum(item["B_to_D_improvement_pct"] > 0 for item in complete)
            sections.append(
                "<p>Across these frozen tasks, geometry changes improved duration on "
                f"{positive}/{len(complete)} tasks; median joint-path length change was "
                f"{_fmt(float(np.median(path_delta)), '%')}; the median gravity-RMS change "
                f"was {_fmt(float(np.median(gravity_delta)), ' Nm')}, while median "
                f"inertia-RMS change was {_fmt(float(np.median(inertia_delta)), ' Nm')} "
                f"({int(np.sum(inertia_delta > 0))}/{len(inertia_delta)} increased). "
                "This supports a torque-demand redistribution interpretation, but does "
                "not isolate causality.</p>"
            )
        mech_rows = []
        for item in sorted(
            complete, key=lambda row: row["B_to_D_improvement_pct"], reverse=True
        ):
            details = _read(mechanism_root / item["case_id"] / "analysis.json", {})
            b = details.get("methods", {}).get("B", {}).get("metrics", {})
            d = details.get("methods", {}).get("D", {}).get("metrics", {})
            parity = max(
                b.get("previous_pin_audit_max_difference_Nm", 0.0),
                d.get("previous_pin_audit_max_difference_Nm", 0.0),
            )
            mech_rows.append(
                [
                    html.escape(item["case_id"]),
                    _fmt(item.get("B_to_D_improvement_pct"), "%"),
                    _fmt(item.get("path_length_increase_pct"), "%"),
                    _fmt(item.get("gravity_rms_delta_Nm"), " Nm"),
                    _fmt(item.get("inertia_rms_delta_Nm"), " Nm"),
                    _fmt(item.get("velocity_bias_rms_delta_Nm"), " Nm"),
                    _fmt(parity, " Nm", 6),
                ]
            )
        sections.append(
            "<h3>All frozen severe source tasks: B versus D decomposition</h3>"
        )
        sections.append(
            _table(
                [
                    "Task",
                    "B→D (%)",
                    "Path length Δ (%)",
                    "Gravity RMS Δ",
                    "Inertia RMS Δ",
                    "Velocity-bias RMS Δ",
                    "Max parity error",
                ],
                mech_rows,
            )
        )
        if regime == "severe" and complete:
            gains = sorted(
                complete, key=lambda row: row["B_to_D_improvement_pct"], reverse=True
            )
            representative = gains[:10] + [gains[len(gains) // 2], gains[-1]]
            seen = set()
            sections.append("<h3>Torque component and path-coordinate profiles</h3>")
            sections.append(
                "<p>Profiles are shown for the ten largest B→D duration improvements, a median-ranked case, and the smallest (including negative if present) case. RMS and component shifts are descriptive; the path-dependent x′ intervals provide local limiting-constraint evidence, not a unique causal decomposition.</p>"
            )
            for item in representative:
                if item["case_id"] not in seen:
                    sections.extend(_case_charts(arm, item["case_id"], assets))
                    seen.add(item["case_id"])
            path_rows = []
            for item in complete:
                path_file = (
                    mechanism_root / item["case_id"] / "path_torque_intervals.npz"
                )
                if not path_file.exists():
                    continue
                with np.load(path_file) as data:
                    row = [html.escape(item["case_id"])]
                    for method in ("b", "d"):
                        for side in ("lower", "upper"):
                            limiting = np.asarray(
                                data[f"{method}_limiting_{side}_joint"]
                            )
                            limiting = limiting[(limiting >= 0) & (limiting < 7)]
                            if limiting.size:
                                counts = np.bincount(limiting, minlength=7)
                                dominant = int(np.argmax(counts)) + 1
                                fraction = float(counts.max() / counts.sum())
                                row.append(f"j{dominant} ({fraction:.0%})")
                            else:
                                row.append("N/A")
                    path_rows.append(row)
            if path_rows:
                sections.append(
                    "<h3>Most frequent torque-limiting joint by path acceleration side</h3>"
                    "<p>At each sampled path coordinate, the intersection of asymmetric torque "
                    "bounds yields a lower and upper allowable x′. The table reports the most "
                    "frequent joint attaining each bound; ties and intervals without a finite "
                    "bound are excluded.</p>"
                )
                sections.append(
                    _table(
                        ["Task", "B lower", "B upper", "D lower", "D upper"],
                        path_rows,
                    )
                )

        native_root = arm / "native_full"
        diversity_rows = []
        for path in sorted(native_root.glob("*/geometry_diversity.json")):
            info = _read(path, {})
            diversity_rows.append(
                [
                    html.escape(info.get("case_id", path.parent.name)),
                    str(info.get("independent_restart_count", "N/A")),
                    str(info.get("native_candidate_count", "N/A")),
                    str(info.get("exact_geometry_count", "N/A")),
                    str(info.get("arc_aligned_unique_geometry_count", "N/A")),
                    _fmt(info.get("geometric_diversity_tolerance_rad"), " rad", 5),
                ]
            )
        if diversity_rows:
            sections.append(
                "<h3>Native planner geometric diversity</h3><p>Restarts use different deterministic seeds; candidate lanes are clustered by joint-space curve after normalized arc-length alignment.</p>"
            )
            sections.append(
                _table(
                    [
                        "Task",
                        "Restarts",
                        "Valid lanes",
                        "Exact curves",
                        "Arc-aligned curves",
                        "Cluster tolerance",
                    ],
                    diversity_rows,
                )
            )

        runtime_rows = []
        case_runtime: dict[str, list[dict[str, float | int]]] = {
            group: [] for group in ("all", "free", "table", "cluttered")
        }
        for case_dir in sorted(native_root.iterdir() if native_root.exists() else []):
            candidate_file = case_dir / "native_candidates.json"
            retime_file = case_dir / "matched_retime_candidates.json"
            if not candidate_file.exists():
                continue
            native_records = _read(candidate_file, {})
            retime_records = _read(retime_file, {})
            population = next(
                (
                    group
                    for group in ("free", "table", "cluttered")
                    if case_dir.name.startswith("pilot_" + group)
                ),
                "all",
            )
            for budget in (1, 4, 16, 32):
                restarts = [
                    record
                    for record in native_records.values()
                    if int(record.get("restart", budget)) < budget
                    and record.get("name", "").startswith("F")
                ]
                timing_records = [
                    record
                    for record in retime_records.values()
                    if int(record.get("restart", budget)) < budget
                    and record.get("status") == "COMPLETE"
                ]
                retime_seconds = 0.0
                for record in timing_records:
                    passes = record.get("passes", [])
                    if passes:
                        retime_seconds += sum(
                            float(
                                item.get(
                                    "complete_solve_wall_s",
                                    item.get("elapsed_s", 0.0),
                                )
                            )
                            for item in passes
                        )
                runtime = {
                    "attempts": len(restarts),
                    "planner_setup_warm_s": sum(
                        float(record.get("setup_and_warmup_wall_s", 0.0))
                        for record in restarts
                    ),
                    "planner_solve_s": sum(
                        float(record.get("plan_wall_s", 0.0)) for record in restarts
                    ),
                    "retime_unique_s": retime_seconds,
                }
                for group in {"all", population}:
                    case_runtime[group].append(runtime)
        for group, tasks_runtime in case_runtime.items():
            if not tasks_runtime:
                continue
            for budget in (1, 4, 16, 32):
                # Runtime records are re-derived per frozen prefix, so restart
                # and retiming costs are summarized from stored candidate rows.
                planner_setup = []
                planner_solve = []
                retime_time = []
                tasks_at_cap = 0
                tasks_with_attempt = 0
                for case_dir in sorted(native_root.iterdir()):
                    candidate_file = case_dir / "native_candidates.json"
                    if not candidate_file.exists():
                        continue
                    population = next(
                        (
                            value
                            for value in ("free", "table", "cluttered")
                            if case_dir.name.startswith("pilot_" + value)
                        ),
                        "all",
                    )
                    if group not in ("all", population):
                        continue
                    native_records = _read(candidate_file, {})
                    attempted_restart_ids = {
                        int(record["restart"])
                        for record in native_records.values()
                        if isinstance(record, dict)
                        and isinstance(record.get("restart"), int)
                    }
                    history_path = case_dir / "restart_attempt_history.jsonl"
                    if history_path.exists():
                        for line in history_path.read_text().splitlines():
                            if line.strip():
                                history_row = json.loads(line)
                                if isinstance(history_row.get("restart"), int):
                                    attempted_restart_ids.add(history_row["restart"])
                    if attempted_restart_ids:
                        tasks_with_attempt += 1
                        if max(attempted_restart_ids) + 1 >= budget:
                            tasks_at_cap += 1
                    restart_records = [
                        record
                        for record in native_records.values()
                        if record.get("name", "").startswith("F")
                        and int(record.get("restart", budget)) < budget
                    ]
                    planner_setup.append(
                        sum(
                            float(row.get("setup_and_warmup_wall_s", 0.0))
                            for row in restart_records
                        )
                    )
                    planner_solve.append(
                        sum(
                            float(row.get("plan_wall_s", 0.0))
                            for row in restart_records
                        )
                    )
                    retime_records = _read(
                        case_dir / "matched_retime_candidates.json", {}
                    )
                    retime_time.append(
                        sum(
                            sum(
                                float(item.get("complete_solve_wall_s", 0.0))
                                for item in row.get("passes", [])
                            )
                            for row in retime_records.values()
                            if row.get("status") == "COMPLETE"
                            and int(row.get("restart", budget)) < budget
                        )
                    )
                runtime_rows.append(
                    [
                        group,
                        str(budget),
                        f"{tasks_at_cap}/{tasks_with_attempt}",
                        _fmt(sum(planner_setup), " s", 1),
                        _fmt(sum(planner_solve), " s", 2),
                        _fmt(sum(retime_time), " s", 1),
                    ]
                )
        if runtime_rows:
            sections.append(
                "<h3>Offline search and matched-retiming effort</h3><p>Planner setup/warmup "
                "is reported separately from warmed plan time. Full-R time sums unique "
                "successful IPOPT candidate solves; byte/geometry aliases are not counted as "
                "new solves. The task column is tasks reaching the stated restart cap over "
                "tasks with any planner attempt: only the nine-task preflight reached caps "
                "above one restart, while all other source tasks received one. These offline "
                "search times are not production latency.</p>"
            )
            sections.append(
                _table(
                    [
                        "Scene class",
                        "Restart budget",
                        "Tasks at cap / represented",
                        "Setup/warmup total",
                        "Native plan total",
                        "Full-R solver total",
                    ],
                    runtime_rows,
                )
            )

        multiplier_root = source.parent / "torque_multiplier_rerun_20261107" / regime
        multiplier_rows = []
        torque_dominance_ratios = []
        for case_dir in sorted(
            multiplier_root.iterdir() if multiplier_root.exists() else []
        ):
            if not case_dir.is_dir():
                continue
            for method in ("b", "d"):
                rerun = _read(case_dir / f"{method}_grid513.json", None)
                historical = _read(arm / case_dir.name / f"{method}_grid513.json", None)
                if not isinstance(rerun, dict):
                    continue
                duals = rerun.get("constraint_multipliers", {})
                other_sums = [
                    float(duals.get(name, {}).get("sum_abs", 0.0))
                    for name in ("velocity", "acceleration", "jerk")
                ]
                torque_sum = float(duals.get("torque", {}).get("sum_abs", 0.0))
                torque_dominance_ratios.append(
                    torque_sum / max(other_sums) if max(other_sums) > 0.0 else math.inf
                )
                passes = rerun.get("refinement_passes", [])
                mesh_points = (
                    "N/A"
                    if not passes
                    else f"{passes[0].get('collocation_points')}→{passes[-1].get('collocation_points')}"
                )
                mesh_delta_ms = (
                    None
                    if len(passes) < 2
                    else 1000.0 * (passes[-1]["duration_s"] - passes[0]["duration_s"])
                )
                parity = (
                    None
                    if not isinstance(historical, dict)
                    else rerun.get("duration_s", 0.0)
                    - historical.get("duration_s", 0.0)
                )
                multiplier_rows.append(
                    [
                        html.escape(case_dir.name),
                        method.upper(),
                        _fmt(rerun.get("duration_s"), " s", 6),
                        _fmt(parity, " s", 6),
                        str(rerun.get("status")),
                        str(rerun.get("iterations")),
                        _fmt(rerun.get("primal_min_margin"), "", 7),
                        _fmt(rerun.get("stationarity_inf"), "", 3),
                        mesh_points,
                        _fmt(mesh_delta_ms, " ms", 5),
                        _fmt(duals.get("torque", {}).get("max_abs"), "", 4),
                        _fmt(duals.get("torque", {}).get("sum_abs"), "", 4),
                        _fmt(duals.get("velocity", {}).get("max_abs"), "", 4),
                        _fmt(duals.get("acceleration", {}).get("max_abs"), "", 4),
                        _fmt(duals.get("jerk", {}).get("max_abs"), "", 4),
                    ]
                )
        if multiplier_rows:
            sections.append(
                "<h3>Fixed-path IPOPT multiplier reruns</h3><p>These are separate reruns; "
                "the frozen historical B/D results are untouched. The duration delta checks "
                "parity with historical grid-513 outputs. Multipliers are raw IPOPT values "
                "for dimensionless pointwise margins scaled by the 95% constraint half-range; "
                "they indicate local sensitivity only and are not directly comparable across "
                "families without accounting for scaling.</p>"
            )
            dominant_count = sum(value > 1.0 for value in torque_dominance_ratios)
            sections.append(
                f"<p>Torque's summed absolute multipliers exceed the largest other-family "
                f"sum in {dominant_count}/{len(torque_dominance_ratios)} qualified B/D reruns. "
                "The table case where the original B path has near-zero torque multipliers "
                "is retained as a negative/mechanism contrast.</p>"
            )
            sections.append(
                _table(
                    [
                        "Task",
                        "Method",
                        "Duration",
                        "Δ historical",
                        "IPOPT status",
                        "Iterations",
                        "Primal min margin",
                        "Stationarity ∞",
                        "Collocation points",
                        "Refinement Δ duration",
                        "Torque λ max abs",
                        "Torque λ sum abs",
                        "Velocity λ max abs",
                        "Acceleration λ max abs",
                        "Jerk λ max abs",
                    ],
                    multiplier_rows,
                )
            )

        if comparison:
            sections.append(
                "<h3>Fresh cuRobo planner under derated effort limits</h3><p>The Full arm constructs the planner with the derated asymmetric actuator limits before geometric planning. Full-R preserves each native spline and retimes it with the same IPOPT solver as B/D. Restart budget is independent planner restarts, each with four TrajOpt lanes.</p>"
            )
            torque_charts, torque_rows = _full_torque_evidence(
                arm, comparison, assets, budget=32
            )
            sections.extend(torque_charts)
            if torque_rows:
                sections.append(
                    "<h4>Selected full-planner torque audit</h4>"
                    "<p>Each row is an adaptive-verified physical-feasible candidate. The chart "
                    "shows a matched Full and Full-R pair from the task with the largest observed "
                    "Full-R-to-D gap at this budget.</p>"
                )
                sections.append(
                    _table(
                        [
                            "Task",
                            "Method",
                            "Candidate",
                            "Duration",
                            "Max torque utilization",
                            "Samples",
                        ],
                        torque_rows,
                    )
                )
            for qualification in ("adaptive_qualified:physical", "sampled:physical"):
                groups = (
                    comparison.get("results", {})
                    .get(qualification, {})
                    .get("groups", {})
                )
                if not groups:
                    continue
                rows = []
                for group in ("all", "free", "table", "cluttered"):
                    for budget in (1, 4, 16, 32):
                        result = groups.get(f"{group}:{budget}", {})
                        native = result.get("native_to_D", {})
                        fixed = result.get("matched_to_D", {})
                        timing = result.get("native_to_matched", {})
                        bfullr = result.get("B_to_matched", {})
                        rows.append(
                            [
                                group,
                                str(budget),
                                f"{native.get('paired_n', 0)}/{native.get('attempted_n', 0)}",
                                f"{fixed.get('paired_n', 0)}/{fixed.get('attempted_n', 0)}",
                                _fmt(native.get("median_pct"), "%"),
                                _fmt(bfullr.get("median_pct"), "%"),
                                _fmt(timing.get("median_pct"), "%"),
                                _fmt(fixed.get("median_pct"), "%"),
                                (
                                    "N/A"
                                    if not fixed.get("paired_bootstrap95_median_pct")
                                    else f"[{_fmt(fixed['paired_bootstrap95_median_pct'][0], '%')}, {_fmt(fixed['paired_bootstrap95_median_pct'][1], '%')}]"
                                ),
                            ]
                        )
                sections.append(f"<h4>{html.escape(qualification)}</h4>")
                sections.append(
                    _table(
                        [
                            "Scene class",
                            "Restarts",
                            "Full/D pairs",
                            "Full-R/D pairs",
                            "Full→D mixed (%)",
                            "B→Full-R (%)",
                            "Full→Full-R timing (%)",
                            "Full-R→D geometry (%)",
                            "paired bootstrap CI",
                        ],
                        rows,
                    )
                )

        sections.append("</section>")

    # Additional source-disjoint and negative-control cohorts stay separate
    # from the original severe source tasks and their paired moderate versions.
    additional = source.parent / "torque_mechanisms_additional_20261108"
    additional_rows = []
    task_rows = []
    for cohort, label in (
        ("severe", "Wide severe, 10 tasks"),
        ("moderate", "Original moderate, 6 tasks"),
    ):
        comparison = _read(
            additional / cohort / "native_full" / "full_planner_comparison.json", None
        )
        if not isinstance(comparison, dict):
            continue
        results = comparison.get("results", {})
        qualified = results.get("adaptive_qualified:physical", {})
        group = qualified.get("groups", {}).get("all:1", {})
        contrast = group.get("matched_to_D", {})
        ci = contrast.get("paired_bootstrap95_median_pct")
        summary.setdefault("additional_cohorts", {})[cohort] = {
            "source_task_count": comparison.get("source_task_count"),
            "physical_adaptive_pairs": contrast.get("paired_n", 0),
            "median_fullr_to_d_pct": contrast.get("median_pct"),
            "paired_bootstrap95_median_pct": ci,
            "d_faster_count": contrast.get("improved_count", 0),
            "d_slower_count": contrast.get("slower_count", 0),
            "scaled95_adaptive_pairs": results.get("adaptive_qualified:scaled95", {})
            .get("groups", {})
            .get("all:1", {})
            .get("matched_to_D", {})
            .get("paired_n", 0),
        }
        runtime_samples = {"setup_s": [], "plan_s": [], "retime_s": []}
        for case, record in qualified.get("cases", {}).items():
            pair = record.get("budget_comparisons", {}).get("1", {})
            full = pair.get("best_native") or {}
            full_r = pair.get("best_matched") or {}
            case_root = additional / cohort / "native_full" / case
            native_record = _read(case_root / "native_candidates.json", {}).get(
                full.get("method", ""), {}
            )
            retime_record = _read(case_root / "matched_retime_candidates.json", {}).get(
                full_r.get("method", ""), {}
            )
            if isinstance(native_record.get("setup_and_warmup_wall_s"), (int, float)):
                runtime_samples["setup_s"].append(
                    float(native_record["setup_and_warmup_wall_s"])
                )
            if isinstance(native_record.get("plan_wall_s"), (int, float)):
                runtime_samples["plan_s"].append(float(native_record["plan_wall_s"]))
            solve_passes = retime_record.get("passes", [])
            if solve_passes:
                runtime_samples["retime_s"].append(
                    sum(
                        float(item.get("complete_solve_wall_s", 0.0))
                        for item in solve_passes
                    )
                )
        summary.setdefault("additional_runtime", {})[cohort] = {
            key + "_median": float(np.median(values)) if values else None
            for key, values in runtime_samples.items()
        }
        additional_rows.append(
            [
                label,
                f"{contrast.get('paired_n', 0)}/{contrast.get('attempted_n', 0)}",
                _fmt(contrast.get("median_pct"), "%"),
                "N/A" if not ci else f"[{ci[0]:.2f}, {ci[1]:.2f}]%",
                str(contrast.get("improved_count", 0)),
                str(contrast.get("slower_count", 0)),
                _fmt(group.get("native_to_matched", {}).get("median_pct"), "%"),
                _fmt(group.get("B_to_matched", {}).get("median_pct"), "%"),
                _fmt(
                    summary["additional_runtime"][cohort]["setup_s_median"],
                    " s",
                    2,
                ),
                _fmt(
                    summary["additional_runtime"][cohort]["plan_s_median"],
                    " s",
                    3,
                ),
                _fmt(
                    summary["additional_runtime"][cohort]["retime_s_median"],
                    " s",
                    2,
                ),
            ]
        )
        for case, record in qualified.get("cases", {}).items():
            pair = record.get("budget_comparisons", {}).get("1", {})
            full = pair.get("best_native") or {}
            full_r = pair.get("best_matched") or {}
            task_rows.append(
                [
                    html.escape(label),
                    html.escape(case),
                    html.escape(record.get("population", "N/A")),
                    _fmt(record.get("reference", {}).get("B"), " s", 6),
                    _fmt(full.get("duration_s"), " s", 6),
                    _fmt(full_r.get("duration_s"), " s", 6),
                    _fmt(record.get("reference", {}).get("D"), " s", 6),
                    _fmt(pair.get("matched_to_D_improvement_pct"), "%", 3),
                ]
            )
    if additional_rows:
        sections.append(
            "<section><h2>Additional full-planner cohorts</h2><p>These results are "
            "separate from the original 20-task severe cohort. Each frozen task received "
            "one fresh native planner restart with four internal seeds. All accepted native "
            "geometries were independently IPOPT-retimed, adaptively audited, and compared "
            "with D. The wide severe subset has 10 source-disjoint tasks (4 free, 3 table, "
            "3 cluttered); the moderate negative control has 6 original tasks (2 per "
            "population). Positive Full-R→D favors the joint-optimized geometry. These "
            "are synthetic derating experiments; finite adaptive samples do not certify "
            "continuous-time feasibility. File-level SHA256 provenance is in "
            '<a href="artifact_manifest.json">artifact_manifest.json</a>.</p>'
        )
        sections.append(
            _table(
                [
                    "Frozen cohort",
                    "Adaptive physical pairs / attempted",
                    "Median Full-R→D",
                    "Paired bootstrap 95% CI",
                    "D faster",
                    "D slower",
                    "Median Full→Full-R timing gain",
                    "Median B→Full-R",
                    "Median planner setup/warmup",
                    "Median native plan time",
                    "Median IPOPT Full-R solve time",
                ],
                additional_rows,
            )
        )
        sections.append(
            "<h3>Per-task durations</h3>"
            + _table(
                [
                    "Cohort",
                    "Task",
                    "Population",
                    "B (s)",
                    "Full (s)",
                    "Full-R (s)",
                    "D (s)",
                    "Full-R→D",
                ],
                task_rows,
            )
        )
        sections.append("</section>")

    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Torque mechanisms and full cuRobo planner</title>
<style>body{font:15px/1.5 system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;color:#1d2938}h1,h2,h3{line-height:1.2}section{margin:3rem 0;border-top:2px solid #d8dee7;padding-top:1rem}table{border-collapse:collapse;width:100%;font-size:13px;margin:1rem 0 2rem}td,th{border-bottom:1px solid #ddd;padding:.45rem;text-align:left}th{background:#f3f6f9}figure{margin:1.5rem 0;padding:1rem;border:1px solid #d8dee7}figure img{display:block;width:100%;height:auto}figcaption{color:#58687a;font-size:13px}.note{background:#fff0d8;border-left:5px solid #b86c00;padding:1rem}</style>
<h1>Torque-aware geometric motion planning</h1><p class="note"><strong>Qualification.</strong> Results use synthetic actuator derating with the original Panda inertias. Independent RNEA decomposition and analytic sphere collision samples do not certify continuous-time safety. IPOPT and multistart planner results are local/budgeted solutions, not global optima. Native Full and Full-R comparisons are distinct: only Full-R matches the B/D timing solver.</p>
<p>Raw artifacts, task definitions, and checksums are stored under the report's configured results root. Historical reports and frozen B/C/D results remain unchanged.</p>"""
    page += "\n".join(sections)
    page += "<section><h2>Interpretation and limits</h2><p>The severe cohort's B→D paths are generally longer while execution is faster. Across the 20-task decomposition, the median geometric path increase is positive, median gravity RMS change is near zero, and inertial RMS rises in every case. This is consistent with redistributing actuator demands across joints and along the motion, rather than globally reducing torque magnitude. The x′ interval plots expose which joints constrain local path acceleration, but do not establish a unique causal explanation.</p><p>A positive adaptive-verified Full-R→D effect means the offline joint optimizer beat the best independently retimed native geometry found within the tested restart budget. A nonpositive effect only shows that this search found a match; it does not imply global optimality. Source tasks are paired across regimes and must not be counted as independent cases.</p><p>Sampled and adaptive checks are finite samples; there is no continuous-time certificate. Collision validation uses the configured analytic sphere model. Results are exploratory and limited to frozen Franka scenes and synthetic derating.</p></section></html>"
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(page, encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    """Publish the report from a results root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    print(json.dumps(publish(args.source, args.report), indent=2))


if __name__ == "__main__":
    main()
