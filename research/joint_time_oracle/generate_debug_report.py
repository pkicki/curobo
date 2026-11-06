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
    recovery_dir = root / "research/joint_time_oracle/results/diagnosis_20261106"
    diagnosis = json.loads((recovery_dir / "diagnosis.json").read_text())
    physical = json.loads((recovery_dir / "physical_summary.json").read_text())
    native = json.loads((recovery_dir / "native_validation.json").read_text())
    pin_recovery = json.loads((recovery_dir / "recovery_pinocchio_validation.json").read_text())
    pin_baseline = json.loads((recovery_dir / "baseline_pinocchio_validation.json").read_text())
    derivatives = json.loads((recovery_dir / "derivatives.json").read_text())
    derivative_summary = {}
    for family in sorted({entry["family"] for entry in derivatives["entries"]}):
        family_entries = [entry for entry in derivatives["entries"] if entry["family"] == family]
        derivative_summary[family] = {
            "entries_including_step_sweep": len(family_entries),
            "max_absolute_error": max(entry["abs_error"] for entry in family_entries),
            "max_relative_error": max(entry["relative_error"] for entry in family_entries),
            "fail_mixed_tolerance": sum(entry["abs_error"] > 0.01 + 0.02 * abs(entry["analytic"])
                                        for entry in family_entries),
        }
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
<p>Unknown. Complete native CUDA derivative and collision qualification and the fixed-path timing control,
then run a frozen pilot. A local solver does
not provide a global optimality certificate.</p>
<h2>Original failure diagnosis</h2>
<p>The largest original residual is Panda joint 7 acceleration at 0.09266 s:
14.8601 rad/s² against the optimizer's 95%-scaled 14.25 rad/s² limit. Its
normalized residual is 0.042814. This peak falls between optimizer samples;
the physical limit is 15 rad/s², which this peak does not exceed. The initial
fit itself exceeded the physical acceleration and jerk limits. The table below
also records endpoint residuals, collision costs, locations, limits, and both grids.</p>
<img src="assets/acceleration.png" alt="Acceleration peak between optimizer samples" style="max-width:100%">
<pre>{dump(diagnosis)}</pre>
<h2>Recovery and fixed-geometry control</h2>
<p>The fitted seed has maximum absolute joint error of approximately 0.0012 rad at
saved baseline samples. Uniformly scaling its time produced a sampled-feasible seed
at approximately 0.4314 s. This preserves the fitted geometry because the saved
endpoint derivatives are zero; it is not a general fixed-path timing optimizer and
is not directly comparable to native TimeOpt. A denser-collocation SLSQP run
produced a sampled-feasible candidate at 0.401694 s. No IPOPT comparison was run.</p>
<pre>{dump(physical)}</pre>
<h2>Independent dynamics and collision checks</h2>
<p>Pinocchio and cuRobo initially differed because their model inertials differed.
After matching cuRobo's default inertials and excluded link, their torques agree
within 0.000017 N·m on the recovery grid. Native swept-world and sampled self
collision checks report zero cost; these checks are not independent of cuRobo,
and neither gives an analytic continuous-time certificate. The baseline has only
18 saved samples, so its dense behavior and the comparison remain unqualified.</p>
<pre>{dump(native)}</pre><h3>Recovery dynamics</h3><pre>{dump(pin_recovery)}</pre>
<h3>Baseline dynamics</h3><pre>{dump(pin_baseline)}</pre>
<h2>Derivative sweeps</h2>
<p>Selected VJP entries were compared using float32 central-difference steps 0.01,
0.003, and 0.001. The exploratory mixed check is absolute error ≤ 0.01 + 0.02 ×
|analytic|, in each weighted output's units. Relative error can be large near zero.
The entries and all step sizes are in derivatives.json. These tests do not cover
all Jacobian entries or contact regimes, and do not qualify the complete optimizer.</p>
<pre>{dump(derivative_summary)}</pre>
<h2>Original short-run summary</h2><pre>{dump(summary)}</pre>
<h2>Task and solver results</h2><p>The longer solve returned solver success at 43 iterations, but dense sampled validation rejected it. The largest normalized constraint violation was approximately 0.043. These runs concern the same generated task and do not form a representative cohort.</p><h3>Short run</h3><pre>{dump(rows)}</pre><h3>Longer run</h3><pre>{dump(longer_rows)}</pre>
<h2>Configuration and hardware</h2><pre>{dump(manifest)}</pre><h3>Longer run configuration</h3><pre>{dump(longer_manifest)}</pre>
<h2>Limitations</h2><p>The baseline check is sampled only; collision checking uses cuRobo callbacks; Pinocchio dynamics were independently cross-checked only after matching the model. This debugging run used an external PyTorch device-property compatibility
shim and is not a performance measurement. The generated cohort, multistart,
adaptive verification, swept validation, control retiming experiment, detailed plots,
and pilot and main studies remain incomplete.</p></html>"""
    destination = root / "reports/joint_time_oracle/index.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page)


if __name__ == "__main__":
    main()
