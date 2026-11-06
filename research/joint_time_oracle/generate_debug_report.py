# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the saved debugging evidence; this is not the requested benchmark report."""
from __future__ import annotations

import html
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data = root / "research/joint_time_oracle/results/debug_20261106"
    manifest = json.loads((data / "manifest.json").read_text())
    rows = json.loads((data / "per_task.json").read_text())
    summary = json.loads((data / "summary.json").read_text())
    longer = root / "research/joint_time_oracle/results/debug_long_20261106"
    longer_rows = json.loads((longer / "per_task.json").read_text())
    longer_manifest = json.loads((longer / "manifest.json").read_text())
    def dump(value: object) -> str:
        return html.escape(json.dumps(value, indent=2))
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Joint path and time: debugging evidence</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:3em auto;padding:0 1em;line-height:1.5}}pre{{overflow:auto;background:#f3f4f5;padding:1em}}.warn{{border-left:5px solid #b43;padding:1em;background:#fff2ee}}</style>
<h1>Joint path and time: debugging evidence</h1>
<p class="warn"><strong>Not qualified.</strong> This is a single debugging attempt with a
three-iteration solver budget and a second longer run on the same task, not the requested benchmark. No independently validated
pair exists, so no conclusion about potential time savings can be drawn.</p>
<h2>Research Implications: Is Joint Path-and-Time Optimization Worth Pursuing?</h2>
<p>Unknown. First qualify native CUDA derivatives and independent dynamics and collision
validation, then run a frozen, representative multistart benchmark. A local solver does
not provide a global optimality certificate.</p>
<h2>Summary</h2><pre>{dump(summary)}</pre>
<h2>Task and solver results</h2><p>The longer solve returned solver success at 43 iterations, but dense sampled validation rejected it. The largest normalized constraint violation was approximately 0.043. These runs concern the same generated task and do not form a representative cohort.</p><h3>Short run</h3><pre>{dump(rows)}</pre><h3>Longer run</h3><pre>{dump(longer_rows)}</pre>
<h2>Configuration and hardware</h2><pre>{dump(manifest)}</pre><h3>Longer run configuration</h3><pre>{dump(longer_manifest)}</pre>
<h2>Limitations</h2><p>The baseline check is sampled only; the optimizer uses cuRobo
RNEA and collision callbacks and has not been cross-checked against an independent
engine. This debugging run used an external PyTorch device-property compatibility
shim and is not a performance measurement. The generated cohort, multistart,
adaptive verification, swept validation, control retiming experiment, detailed plots,
and pilot and main studies remain incomplete.</p></html>"""
    destination = root / "reports/joint_time_oracle/index.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page)


if __name__ == "__main__":
    main()
