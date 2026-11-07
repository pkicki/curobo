# Mechanisms of torque-aware geometric gains and a full native cuRobo baseline

**Status: IMPLEMENTED, NOT YET EXECUTED.** The parent torque-stress study
is frozen at bcbb85a709f06b2a934f8ba141137c33379c0450.
Do not overwrite or revise its original B, C, D results, trajectory NPZ
archives, independent validation receipts, or prior HTML reports.

## Scientific questions

1. WHY did the severe synthetic torque-derating cohort show 5.46% median
   matched-retimer B-to-D improvement on the original 20 cases and 3.47%
   on the 29 qualified wider-cohort pairs? Does the path reduce gravity,
   inertia-driven load, velocity-dependent loads, or the acceleration
   demands of particular joint-space curves? Are those changes tied to
   specific active torque inequalities?
2. DOES a full fresh native cuRoboV2-Fast C-space planner, built with
   the derated effort limits BEFORE planning and TimeOpt selection, find
   equally good geometric paths under the same constraints? How do its
   results depend on its geometric-search budget?

B = IPOPT retiming of original cuRobo geometry. D = same IPOPT solver on
C's optimized geometry. A10/A40 in the historical study merely retimed
pre-stress geometry under modified limits; they did NOT constitute a
fresh full-planner comparison. The new native-full arm is separate.

All stress limits simulate a *synthetically derated actuator*, NOT a
payload or the actual physical Panda limits. All source tasks,
collision geometry, original model inertials, and endpoint contracts are
unchanged.

## Freeze, problem selection and staging

Stage 0: confirm the source commit, task inventory, torque-profile SHA256,
and independent frozen B/D receipts. Compile CUDA extensions and run the
CPU tests before any new experimental work. Store new evidence in sibling
folders, not over the frozen archives.

Stage 1: preflight full native planner on nine predefined severe cases:
- pilot_free_00, pilot_free_02, pilot_free_04
- pilot_table_00, pilot_table_03, pilot_table_06
- pilot_cluttered_00, pilot_cluttered_03, pilot_cluttered_05

Also perform an independent RNEA breakdown for all 20 severe cases;
focus visual case studies on the 10 largest verified B/D gains PLUS
smaller/negative examples (selected examples are not summary statistics).
When working, expand to all 20 original severe cases, then 10 new
source-disjoint severe wide cases. Run six original moderate cases as a
separate negative-control study. Keep each cohort separate.

Full-planner budgets are nested 1, 4, 16 and 32 independently seeded
fresh MotionPlanner restarts, each requesting four internal TrajOpt seeds.
The number of restarts is NOT the number of individual geometric paths
or TimeOpt Newton steps. Preserve failures and all selected candidate
archives. Use the same restart seeds across nested budgets. Check actual
geometry diversity with a common arc-length-aligned joint-space metric.

Do not run 32-restart planning on all cases until 1/4/16 budgets are
qualified on the representative population. Do not infer that new seeds
necessarily produce distinct obstacle routes.

## Part A: physics mechanism investigation

Using the model-aligned independent Pinocchio RNEA workspace,
decompose torque exactly:

    gravity = RNEA(q, 0, 0)
    inertia = RNEA(q, 0, qdd) - gravity
    velocity_bias = RNEA(q, qd, 0) - gravity
    total = inertia + velocity_bias + gravity

The code samples B and D at 501 common normalized physical-time points
and records all four terms, residual identity error, max torque joint,
peak time and component values, RMS contributions, time near active
torque, path length changes, and source hashes.

Run in the independent Pinocchio environment:

    PIN_PY -m research.joint_time_oracle.torque_mechanisms \
      --root research/joint_time_oracle/results/torque_stress_final_20261107 \
      --regime severe

Output is in each case's mechanisms subfolder and mechanisms/summary.json.
The source trajectory outputs are already independently audited but
sampled; the 501-point mechanism traces are descriptive only.

Also investigate the path-coordinate torque equation:

    tau(s) = g(q(s)) + m(q,q_s,q_ss)*x(s) + 0.5*a(q,q_s)*x'(s),
    x = (ds/dt)^2, m = M(q)q_ss + C(q,q_s)q_s, a = M(q)q_s.

Plot upper/lower torque-allowable acceleration intervals x'(s) at the
B and D timing profiles. Pay attention to near-zero coefficients and
stationary spans, and check whether different joints become limiting.
Do not call RMS term differences a unique causal explanation.

The fixed-path IPOPT solver now exports per-family absolute multiplier
summaries for velocity, acceleration, jerk and torque. Historical
artifacts lack these values. To collect them, rerun chosen B and D
problems INTO A NEW RESULT ROOT, not the frozen cohort, and verify
duration/geometry/feasibility parity first. Check KKT residuals,
pointwise scaling, mesh stability and the appropriate multiplier sign
convention. Positive multipliers on torque constraints provide stronger
local sensitivity evidence than utilization alone.

## Part B: genuine full native planner rerun

Unlike earlier A10/A40, build a fresh RobotCfg from the Franka YAML for
each profile, validate its original effort limits against the frozen
nominal bounds, and overwrite ONLY the effort bounds with that task's
modified [lower, upper] values BEFORE building MotionPlannerCfg and
MotionPlanner. Ensure the TrajOpt kinematic joint limits and the TimeOpt
scaled effort limits both reflect the new values.

A new planning restart calls plan_cspace from the exact frozen endpoints,
scene and graph planner with a distinct deterministic random seed.
Retain all failures. Save the selected lane AND every alternate lane
passing native path, collision and dynamic validity masks, with their entire
spline geometry, exact 8193-sample physical-time reconstruction and checksums.
Alternate lanes are named F000L02 etc., share the parent F000 timing
buffers, and count as part of the same planner restart. Missing internal
validity masks fail closed for alternate lanes. Store setup/warmup time separately from
planning wall time. These are offline timing observations, not an online
latency benchmark.

Preflight one representative case:

    CUDA_PY -m research.joint_time_oracle.torque_full_planner \
      --root research/joint_time_oracle/results/torque_stress_final_20261107 \
      --regime severe --case pilot_free_00 --max-restarts 1

Then run the other eight predefined cases. If the single-restart
preflight passes, increase --max-restarts to 4, 16 and finally 32.
Escalation resumes within the same frozen configuration without
rerunning completed restarts.

The selected native candidate from each restart is not yet qualified.
Before verification, independently retime **every** saved Full geometry with
the SAME fixed-path IPOPT solver and speed parameterization used by B and D:

    PIN_PY -m research.joint_time_oracle.torque_full_retime \
      --root research/joint_time_oracle/results/torque_stress_final_20261107 \
      --regime severe --grid 513 --maxiter 500

This creates the matched Full-R arm (RF000, RF000L02, etc.). Full-R retains
exact native cubic geometry, changes only timing, and records IPOPT
convergence, collocation refinement and multipliers. Source hashes remain
linked to native F candidates. An original Full-native versus D comparison
mixes geometry and timing quality; **Full-R versus D is the primary fair
geometric comparison**. Do not equate Full-R solver success with independent
physical feasibility.

In the independent Pinocchio environment, audit BOTH Full and Full-R:

    PIN_PY -m research.joint_time_oracle.torque_full_validate \
      --root research/joint_time_oracle/results/torque_stress_final_20261107 \
      --regime severe

This writes native_full/independent_sampled.json for all successful
candidate labels F000,F001,... . It checks exact frozen modified limits,
aligned Pinocchio RNEA and independent sphere-model collision.

Replay native F time laws at extra times around independent extreme
measurements, using exactly frozen controls/timing buffers:

    CUDA_PY -m research.joint_time_oracle.torque_full_adaptive \
      --root research/joint_time_oracle/results/torque_stress_final_20261107 \
      --regime severe

Replay matched Full-R time laws using the EXACT same geometric spline
and IPOPT speed controls in the CPU Pinocchio environment:

    PIN_PY -m research.joint_time_oracle.torque_full_retime_adaptive \
      --root research/joint_time_oracle/results/torque_stress_final_20261107 \
      --regime severe

Audit BOTH the native and matched Full-R adaptive states independently:

    PIN_PY -m research.joint_time_oracle.torque_full_validate \
      --root research/joint_time_oracle/results/torque_stress_final_20261107 \
      --regime severe --adaptive

Only then generate checksum-bound adaptive receipts:

    PIN_PY -m research.joint_time_oracle.torque_full_receipts \
      --root research/joint_time_oracle/results/torque_stress_final_20261107 \
      --regime severe

Finally select by shortest *independently validated* duration for every
nested budget:

    PIN_PY -m research.joint_time_oracle.torque_full_select \
      --root research/joint_time_oracle/results/torque_stress_final_20261107 \
      --regime severe \
      --adaptive-receipts research/joint_time_oracle/results/torque_stress_final_20261107/severe/native_full/adaptive_receipts.json

This computes two separate statistics sets: physical vs 95%-scaled
limits, and sampled-only vs adaptive-verified. Missing or failed
validation is NA. No silent exclusion or fabricated 0% improvement.

## Primary contrasts and reporting

For each task, regime and planning-restart budget:

    B_to_Full = 100 * (T_B - T_Full) / T_B
    Full_to_D = 100 * (T_Full - T_D) / T_Full           (solver-mixed)
    Full_to_FullR = 100 * (T_Full - T_FullR) / T_Full     (timing only)
    B_to_FullR = 100 * (T_B - T_FullR) / T_B             (geometry with matched timing)
    FullR_to_D = 100 * (T_FullR - T_D) / T_FullR         (PRIMARY geometry comparison)
    B_to_D = 100 * (T_B - T_D) / T_B

Positive Full-R-to-D indicates D remains faster than the best independently
verified FULL native planner geometry after it is retimed using the SAME
IPOPT solver as D. Full-to-D (native TimeOpt) is a practical solver-mixed
comparison, not an isolated geometric advantage.
A native planner running with derated constraints is still NOT necessarily
using a torque-aware geometric objective; it does use derated TimeOpt
validation and selection. Report actual collision-free route diversity.

Summaries must include N attempted, planner success, valid outputs,
paired verified N, signed median/mean %, paired bootstrap 95% CI,
absolute differences in ms, improved/slower fractions, active effort
utilization, and offline wall time per budget. Show scene strata and
negative cases. Paired moderate/severe versions of the same source
are correlated and cannot be pooled naively.

Use at least 10 adaptive-verified Full-R/D pairs across two or more scene
classes before claiming an initial exploratory effect. New disjoint
confirmation cases are required for broad generalization. Neither
sampled dense+adaptive evidence nor an IPOPT status is a continuous-time
certificate or global optimality proof.

An optional second stage uses CEM/CMA-ES to search 2/4/8 intermediate
waypoints and the same fixed-path solver to retime every proposed path;
freeze a total compute budget, store every infeasible proposal and use
the same independent validation. Do not call derivative-free finite
budget search globally optimal.

## HTML, source, and limitations

Publish an evidence-only combined report from source JSON:

    PIN_PY -m research.joint_time_oracle.torque_followup_report \
      --source research/joint_time_oracle/results/torque_stress_final_20261107 \
      --report reports/joint_time_oracle/torque_mechanisms

This builds summary.json and index.html with N/A for unrun contrasts.
Extend it with torque time traces, physical configuration snapshots,
joint-specific active constraints and IPOPT multipliers as evidence
becomes available. All plots should be reconstructable from NPZ/JSON
input, with manifest SHA and source reference recorded.

Do not overwrite historical evidence or alter production defaults.
Do not merge into main and NEVER create a PR.

## New modules

- torque_mechanisms.py: independent RNEA decomposition and checksummed output.
- torque_full_planner.py: full native rerun with derated robot config.
- torque_full_validate.py: independent candidate audit.
- torque_full_adaptive.py: exact native dense+extrema replay.
- torque_full_receipts.py: SHA-linked adaptive audit receipts.
- torque_full_select.py: fail-closed cumulative budgets and paired effects.
- torque_followup_report.py: reproducible HTML publisher.
- cohort_retime.py: new per-family IPOPT multiplier summaries.
- tests/test_joint_time_torque_mechanisms.py and
  tests/test_joint_time_torque_full_receipts.py: CPU regression tests.

**No new GPU experiments or end-to-end test suite have been run by the
assistant.** Do not claim otherwise; the coding agent must run them.


## Matched-retimer software additions

The follow-up includes torque_full_retime.py and
 torque_full_retime_adaptive.py. The general independent validator,
 SHA-linked receipt generation, full-planner budget selector and HTML
 publisher all accept both native F and matched RF candidate labels.
 The numerator of the primary geometry contrast is the independently
 qualified Full-R trajectory, not the native TimeOpt trajectory.
