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
    current_dir = root / "research/joint_time_oracle/results/native_reconstruction_20261106"
    current = json.loads((current_dir / "qualification_summary.json").read_text())
    dense = json.loads((current_dir / "dense_validation.json").read_text())
    pin_dense = json.loads((current_dir / "pinocchio_dense.json").read_text())
    self_midpoints = json.loads((current_dir / "self_midpoints.json").read_text())
    cases = current["cases"]
    pilot = root / "research/joint_time_oracle/results/cohort_pilot_isolated_20261106/pilot.jsonl"
    cohort_notice = ""
    if pilot.exists():
        pilot_count = sum(bool(line.strip()) for line in pilot.read_text().splitlines())
        cohort_notice = (f'<h2>Multi-problem exploratory pilot</h2><p class="warn">'
                         f'{pilot_count} pilot tasks attempted. B, C and D and adaptive qualification '
                         'remain incomplete; confirmation was not launched. '
                         '<a href="cohort/index.html">View pilot results and per-task diagnostics</a>.</p>')
    timing = json.loads((current_dir / "timing_research_summary.json").read_text())
    jacobian = json.loads((current_dir / "pointwise_jacobian.json").read_text())
    def dump(value: object) -> str:
        return html.escape(json.dumps(value, indent=2))
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Joint path and time: debugging evidence</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:3em auto;padding:0 1em;line-height:1.5}}pre{{overflow:auto;background:#f3f4f5;padding:1em}}.warn{{border-left:5px solid #b43;padding:1em;background:#fff2ee}}</style>
<h1>Joint path and time: debugging and qualification evidence</h1>
{cohort_notice}
<p class="warn"><strong>Pilot decision: BLOCKED.</strong> The exact historical baseline
has been reconstructed, and A, B and C satisfy the physical limits on the sampled grids.
The 95%-scaled limits and joint-oracle derivative checks still need resolution, self collision has no
continuous certificate, and the independent timing solution reaches an artificial control bound. These observations are not a qualified improvement.</p>
<p><a href="../../research/joint_time_oracle/results/native_reconstruction_20261106/checksums.json">Raw artifacts and checksums</a></p>
<h2>New pointwise timing experiment</h2>
<p>On this task the fixed-path candidate is faster than the saved jointly optimized candidate.
This does not demonstrate a geometric improvement. The comparison is still blocked as a
common 95%-limit qualification: A and C have small scaled-limit violations, and the
full joint-optimizer CUDA derivative discrepancy classification is unfinished.</p>
<p>A: {timing["durations_s"]["A_native"]:.9f} s; B, restored: {timing["durations_s"]["B_restored"]:.9f} s;
C: {timing["durations_s"]["C_joint"]:.9f} s. A to B: {timing["A_to_B_percent"]:.3f}%;
B to C: {timing["B_to_C_percent"]:.3f}% (negative means C is slower).</p>
<img src="assets/timing_decomposition.png" alt="Durations for A, B and C; axis is truncated" style="max-width:100%">
<img src="assets/native_iteration_sweep.png" alt="Native timing duration versus iteration budget" style="max-width:100%">
<img src="assets/ipopt_convergence.png" alt="Pointwise IPOPT objective across iterations" style="max-width:100%">
<img src="assets/pointwise_profiles.png" alt="A, B and C joint 4 physical-time profiles" style="max-width:100%">
<p>The explicit pointwise IPOPT problem uses 32 cubic squared-speed controls, 2170
path samples and separate bounds, independent Pinocchio dynamics, and local-support
Jacobian sparsity. It converged at {timing["solver"]["duration_s"]:.9f} s with stationarity
residual {timing["solver"]["stationarity_inf"]:.3g} for its bounded, sampled NLP. Dense checks
found small violations of scaled acceleration and jerk; multiplying squared speed by
0.99998 gives the displayed B candidate, which passes both sets of sampled dynamic limits.
This restoration is not itself a stationary solution. Mild uniform time scaling of A and C yields {timing["scaled_restorations"]["baseline"]["duration_s"]:.9f} s and {timing["scaled_restorations"]["candidate"]["duration_s"]:.9f} s, respectively; these also pass 95% limits on 8193 samples, but need the full adaptive verification contract. A 1079-point solve from a different seed with tighter tolerance returned {timing["solver_crosscheck"]["duration_s"]:.9f} s, but still used the same upper bound. The maximum normalized speed control
is {timing["maximum_scaled_control"]:.3f} against an artificial upper bound of 10000: the bounded
solver status does not establish an unconstrained fixed-path optimum. The first path
segment is stationary, permitting large timing coefficients there. The earlier upper bound
of 100 was also active and made the earlier 0.4016 s IPOPT solutions too restrictive.</p>
<p>Native TimeOpt uses the same 32-control count, 512 uniform nodes plus path knots,
95% planning scales and a default budget of 10 iterations. On the frozen input its
reported durations decrease at larger budgets: 5, 10, 20, 40 and 80 iterations are
shown above. The 40/80 iteration output has not been independently validated using
native physical-time sampling, so it is evidence of iteration sensitivity, not a
qualified faster baseline. The initial native uniform seed is 0.517661 s; independent
IPOPT uses Gauss quadrature, an explicitly differentiated pointwise formulation, a different
path parameterization and aligned Pinocchio dynamics. Different grids and the native
FP32 acceptance tolerance also matter; the individual contribution of each has not been isolated.</p>
<p>Independent Pinocchio forward kinematics and analytic sphere-box/sphere-sphere tests
check 61 configured robot spheres, the table, configured self-ignore pairs and padding.
All three paths have positive sampled clearance. These checks do not cover robot meshes or
provide a continuous-time collision certificate.</p>
<p>Pointwise derivatives were checked entry by entry over three central-difference steps at
two profiles. The maximum best-step absolute error was {jacobian["optimized"]["max_best_absolute_error"]:.3g};
this qualifies these smooth timing derivatives only, not the joint optimizer's CUDA collision derivatives.</p>
<details><summary>Frozen timing, feasibility, collision and convergence data</summary><pre>{dump(timing)}</pre></details>
<details><summary>Pointwise Jacobian checks</summary><pre>{dump(jacobian)}</pre></details>
<p><a href="../../research/joint_time_oracle/results/native_reconstruction_20261106/timing_research_checksums.json">New artifacts and checksums</a></p>
<h2>Historical reconstruction and A/B/C candidates</h2>
<p>The rerun reproduced all 18 saved states exactly. Native TimeOpt was replayed using
captured controls, endpoint controls, timing coefficients and integrated times, with
641, 4097, 8193 and 16385 samples. The native replay agrees exactly at the 8193
shared points. The original geometric path was evaluated independently with maximum
position discrepancy 4.2e-7 rad on the measured interior samples.</p>
<table><thead><tr><th>Method</th><th>Duration (s)</th><th>Physical violations</th><th>95%-scaled violations</th></tr></thead>
<tbody><tr><td>A Native TimeOpt</td><td>{cases["baseline"]["duration_s"]:.9f}</td><td>{dump(cases["baseline"]["physical_max_violation"])}</td><td>{dump(cases["baseline"]["scaled_95_max_violation"])}</td></tr>
<tr><td>B Independent fixed-path timing</td><td>{cases["fixed_path"]["duration_s"]:.9f}</td><td>{dump(cases["fixed_path"]["physical_max_violation"])}</td><td>{dump(cases["fixed_path"]["scaled_95_max_violation"])}</td></tr>
<tr><td>C Joint candidate</td><td>{cases["candidate"]["duration_s"]:.9f}</td><td>{dump(cases["candidate"]["physical_max_violation"])}</td><td>{dump(cases["candidate"]["scaled_95_max_violation"])}</td></tr></tbody></table>
<p>This historical B uses a 32-control cubic squared-speed law and independent SLSQP/Pinocchio calculations.
The 800-iteration result was subsequently uniformly slowed by multiplying squared speed
by 0.997 to restore the sampled 95% limits. It is a feasible sampled timing candidate,
not a converged minimum. The independent retimer has not been applied to C's quintic path.</p>
<img src="assets/comparison_profiles.png" alt="Joint position and dynamic profiles" style="max-width:100%">
<img src="assets/constraint_utilization.png" alt="Constraint utilization profiles" style="max-width:100%">
<img src="assets/comparison_geometry.png" alt="Joint-space and end-effector geometry" style="max-width:100%">
<h2>Current dense and independent checks</h2>
<p>All three cases have zero sampled physical-limit violation. Pinocchio uses explicitly
aligned inertials and agrees with cuRobo to about 0.00002 N·m on 8193 A/C and 16439 B samples. World swept cost and
more than 16000 discrete self-collision samples per case are zero. Sampled checking and
swept sphere approximations are not continuous-time certificates. Endpoint acceptance
uses 1e-5 rad, 1e-4 rad/s and 1e-3 rad/s²; the saved quintic evaluated in higher precision
has acceleration residual approximately 1.4e-12 rad/s², whereas the earlier 0.000502 was
float32 matrix cancellation.</p><pre>{dump(pin_dense)}</pre><pre>{dump(self_midpoints)}</pre>
<details><summary>Per-grid, per-family detailed residuals and locations</summary><pre>{dump(dense)}</pre></details>
<h2>CUDA derivative evidence</h2>
<p>Full 7×21 RNEA Jacobians were swept at two trajectory states and a random state;
collision maxima were checked in free space, near contact and penetration. Complete
SLSQP constraint Jacobians were checked at seed and prior final spline controls. A
0.01 + 0.02 |analytic| exploratory absolute-plus-relative threshold is used; this is
not a complete proof. Many large central-difference discrepancies coincide with active
sample changes, and a step 0.003 discrepancy initially missed by the heuristic was traced to a switch
from active sample 8 to 7. Other flagged cases are not individually classified. See the
raw entries before drawing conclusions.</p>
<pre>{dump(current["oracle_jacobian_sweep"])}</pre>
<details><summary>Configuration, durations and outstanding qualification gates</summary><pre>{dump(current)}</pre></details>
<p class="warn"><strong>Not qualified.</strong> This is a single debugging attempt with a
three-iteration solver budget and a second longer run on the same task; later dense checks are reported above. No fully qualified
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
and neither gives an analytic continuous-time certificate. At that earlier stage, only 18 baseline samples were available; native dense reconstruction is described above.</p>
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
<h2>Limitations</h2><p>The baseline check is sampled only; historical collision checks used cuRobo callbacks; new independent sampled sphere checks are described above. Pinocchio dynamics were cross-checked only after matching the model. This debugging run used an external PyTorch device-property compatibility
shim and is not a performance measurement. The generated cohort, multistart,
adaptive verification, swept validation, control retiming experiment, detailed plots,
and pilot and main studies remain incomplete.</p></html>"""
    destination = root / "reports/joint_time_oracle/index.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page)


if __name__ == "__main__":
    main()
