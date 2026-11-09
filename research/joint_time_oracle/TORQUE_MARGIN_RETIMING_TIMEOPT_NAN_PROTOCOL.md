# Margin-aware fixed-geometry retiming and TimeOpt NaN diagnosis

**IMPLEMENTED, NOT EXECUTED BY ASSISTANT.** Research source frozen at
be595eabea4d74abe5f7ef879c846b950f905e4a on the global-LP study.
NO PR; NO main merge; NO production defaults changed. The independent
source cases use synthetic Panda actuator limits, not hardware ratings.

## Goal and registered starting points

Three independent W geometries passed sampled Pinocchio physical and
Scaled95 checks. Source IPOPT P on the SAME geometries passed physical,
but FAILED Scaled95 on one or more kinematic/torque families:

| Case | Qualified W (s) | Unqualified P (s) |
| --- | ---: | ---: |
| trap_cluttered_08 | 8 | 2.906696 |
| confirm_geom_table_07 | 6 | 2.244931 |
| confirm_geom_cluttered_02 | 8 | 2.541856 |

Find a new retiming R satisfying independent physical AND Scaled95
dense/adaptive checks, on the exact frozen P/W geometric path, while
preserving most of P's execution-duration gain. A locally converged
IPOPT retime and preliminary sampled acceptance DO NOT qualify R.

Separate native question: why did native cuRobo TimeOpt output invalid
integrated times and NaN post-TimeOpt states on the selected synthetic
j1234_f0300 case? Record earliest OBSERVED invalid CUDA buffer; do not
claim an exclusive cause without controlled numerical reproduction.

## Fixed time-dilation arm

Existing P uses a 32-control cubic B-spline x(s)=sdot^2. For a fixed
geometry q(s) and slowdown rho>=1:

    x_rho(s) = x_P(s)/rho^2, T_rho=rho*T_P
    qdot_rho=qdot_P/rho, qddot_rho=qddot_P/rho^2
    jerk_rho=jerk_P/rho^3
    tau_rho=g(q)+(tau_P-g(q))/rho^2.

The gravity term means torque feasibility need NOT be monotonic in
rho. DO NOT binary-search on an assumed monotone property. Registered
discrete grid:

    1,1.005,1.01,1.025,1.05,1.075,1.10,1.15,
    1.20,1.30,1.50,2,2.5,3.

Evaluate all candidate pilot limits using fixed frozen P geometry and
independent Pinocchio RNEA/gravity. Save all attempted factors and
constraint-family signed violations. Once a factor passes conservative
sampled Scaled95 pilot checks, reconstruct R from exact squared-speed
controls, not interpolated numerical derivatives, and independently
audit dense and adaptive samples. Source W and P never change.
No registered rho => NO_CANDIDATE, not IPOPT infeasibility proof.

## Tightened-IPOPT fixed-geometry arms

Re-run existing IPOPT with the ORIGINAL P controls as initialization.
Change ONLY the allowable timing envelope, never geometric controls.
Additional registered Scaled95 tightening factors:

    beta=0.98,0.96,0.94,0.92.

Velocity, acceleration, jerk half-widths shrink about each original
midpoint; asymmetric torque limits shrink about ZERO. The resulting
torque envelope is [0.95*beta*lo,0.95*beta*hi]. Do not multiply
torque widths alone or scale torque around a nonzero midpoint.

Keep IPOPT grid=513 and max_iter=200; preserve original solver
convergence, primal margin, multipliers, iterations and wall time.
A converged solution is PRELIMINARY, and still requires both independent
physical/Scaled95 dense+adaptive checks. Save raw R speed controls and
exact source geometry hashes. Frozen IPOPT P remained unqualified
Scaled95; do not promote it retroactively.

## Stage P0: tests, provenance and source registration

Use pinned independent Pinocchio CPU environment; avoid Eagle CPU.
Separate native CUDA environment and compile CUDA extensions before
native diagnostic. Do not introduce the Pinocchio PYTHONPATH into the
CUDA test process. Run tests/test_joint_time_torque_margin.py, previous
research CPU and CUDA tests, Ruff E/F/I, Black, compileall and diff
checks. Test time scaling, nonmonotone gravity, asymmetric Scaled95,
IO/attribute schemas and TimeOpt trace flag correctness. Fix bugs
BEFORE development freeze in an unscored output root.

Verify previous SHA manifest and required local raw:
w_geometry.json, w_dense.npz, p_dense.npz, p_refined_dense.npz,
p_timing_controls.npz, ipopt_polish.json and torque_profile.json,
plus independently qualified source W receipt. Refuse missing or
different files; do not infer missing trajectory from compact summary.

Register source (PIN_PY means the pinned Pinocchio python executable):

    export SRC=$PWD/research/joint_time_oracle/results/torque_global_lp_screening_20261109
    export OUT=$PWD/research/joint_time_oracle/results/torque_margin_retime_20261109

    PIN_PY -m research.joint_time_oracle.torque_margin_prepare \
      --source "$SRC" --output "$OUT"

This freezes all 3 source task IDs, original W and P path hashes,
per-joint torque profiles, baseline durations and source W receipt.

## Stage P1: development on trap_cluttered_08 only

    PIN_PY -m research.joint_time_oracle.torque_margin_retime \
      --root "$OUT" --case trap_cluttered_08 --method dilation

    for BETA in 0.98 0.96 0.94 0.92; do
      PIN_PY -m research.joint_time_oracle.torque_margin_retime \
        --root "$OUT" --case trap_cluttered_08 \
        --method tighten --beta "$BETA" --grid 513 --maxiter 200
    done

Five arms: dilation, tight_b980, tight_b960, tight_b940, tight_b920.
Record all unsuccessful solver and margin cases; do not manufacture R.

For each arm with nonempty tasks.json, run in an INDEPENDENT Pinocchio
process (not the IPOPT scoring process):

    PIN_PY -m research.joint_time_oracle.cohort_validate \
      --input "$OUT/$ARM" --methods R \
      --torque-profile-dir "$OUT/$ARM" \
      --report independent_sampled.json

    PIN_PY -m research.joint_time_oracle.torque_margin_adaptive \
      --root "$OUT" --arm "$ARM"

    PIN_PY -m research.joint_time_oracle.cohort_validate \
      --input "$OUT/$ARM" --extra "$OUT/$ARM" \
      --methods R_adaptive \
      --torque-profile-dir "$OUT/$ARM" \
      --report independent_adaptive.json

    PIN_PY -m research.joint_time_oracle.torque_margin_receipts \
      --root "$OUT" --arm "$ARM"

Require physical+Scaled95 five-family joint limits, asymmetric RNEA
torque, world/self collision spheres, endpoint rest, 8193 dense and
exact event-refined adaptive samples, source geometry exact equality,
R/adaptive SHA and profile/task SHA. Dense+adaptive sampled checks do
NOT prove continuous-time torque/collision safety.

## Stage P2: freeze method, then two held-out tasks

Once all five development outcomes and any constructed R receipts are
final and tests pass:

    PIN_PY -m research.joint_time_oracle.torque_margin_freeze \
      --root "$OUT"

The freeze includes implementation source SHA, original registration,
five development result SHA and independent R receipts. Do not change
rho/beta grids, solver budget, code or development evidence afterward.
A violation blocks holdout and requires a NEW versioned campaign.

Then run the same five arms on each of:
confirm_geom_table_07, confirm_geom_cluttered_02.

    for CASE in confirm_geom_table_07 confirm_geom_cluttered_02; do
      PIN_PY -m research.joint_time_oracle.torque_margin_retime \
        --root "$OUT" --case "$CASE" --method dilation --holdout
      for BETA in 0.98 0.96 0.94 0.92; do
        PIN_PY -m research.joint_time_oracle.torque_margin_retime \
          --root "$OUT" --case "$CASE" --method tighten \
          --beta "$BETA" --grid 513 --maxiter 200 --holdout
      done
    done

Arm inventories allow append-only holdout entries but never edits to
previous case definitions. Repeat the independent dense/adaptive R
audits on all existing and newly added arm tasks. Verify the frozen
development receipt entries remain exactly unchanged. This is a
three-task, deliberately selected study, not population generalization.

## Stage P3: isolated native TimeOpt invalid-time localization

Prior native one-restart preflight in the cluttered task had five
failed planning attempts, 40 lanes, 18 distinct paths, invalid
integrated times and all post-TimeOpt sampled q/qd/qdd/jerk NaN.
Earlier stage masks were path_valid TRUE and collision/dynamics FALSE;
there was no valid same-post-state physical comparison.

DO NOT run more global restarts. Prepare separate native inputs:

    PIN_PY -m research.joint_time_oracle.native_pretimeopt_prepare \
      --output research/joint_time_oracle/results/native_timeopt_phase_probe_20261109

Pinned CUDA env, one restart only:

    CUDA_PY -m research.joint_time_oracle.torque_full_planner \
      --root research/joint_time_oracle/results/native_timeopt_phase_probe_20261109 \
      --regime j1234_f0300 --case confirm_geom_cluttered_02 \
      --max-restarts 1 --lanes 4 --max-attempts 5 \
      --capture-pre-timeopt --trace-timeopt-phases

Research-only hook reads: source path/input_valid, solver seed-feasible
and numerical flags, solver 32 squared-speed controls and duration,
finish optimization flag and controls, integrated times, TimeOpt status
and success, downstream collision/dynamic masks.

The native_timeopt_nan_trace module reports per-lane earliest OBSERVED
invalid buffer. It does not prove causal primacy: internal states may
be overwritten by finish kernels, CUDA graph reuse or buffer aliasing.
Capture errors are UNKNOWN. Never reclassify native planning errors as
geometric infeasibility or claim a production bug from a NaN alone.

Run a SEPARATE uninstrumented same-task, same-seed control in a
distinct output root. Compare failure status/attempt count, not
timings. If finite solve controls become NaN only in integration,
design a NEW narrowly scoped integration-kernel reproducer. If solver
controls are already NaN, inspect initialization/line-search, not
collision model. If seed feasible flag is false, document its actual
semantics before claiming a numerical error. Do not modify production
kernels or add large GPU campaigns automatically.

## Stage P4: report, artifact provenance and decision

    PIN_PY -m research.joint_time_oracle.torque_margin_report \
      --root "$OUT" \
      --native-arm research/joint_time_oracle/results/native_timeopt_phase_probe_20261109/j1234_f0300/native_full \
      --report reports/joint_time_oracle/torque_margin_retiming

Report all 15 task/arm cells, successful, unqualified, solver errors and
not executed. Compare original qualified W, original unqualified P,
new qualified R, rho/beta, physical/Scaled95 family margins, IPOPT
time/iterations, R duration and percent gain vs W. Preregistered
meaningful improvement is >=10% W duration reduction at independent
physical+Scaled95 qualification. Source P duration is a numerical
lower reference, not a qualified comparator. Do not claim a global
time optimum from IPOPT.

Native trace is a SEPARATE result and cannot establish an exclusive
root cause absent controlled numerical replay.

Build SHA inventory using existing geometry_feasibility_manifest.py
with root "$OUT" and report directory above. Keep giant raw NPZ local
and commit compact JSON/HTML/relevant figures with checksums. No
fictional success graphs, no false continuous-time certification.

Research decisions: GO_MARGIN_RECOVERY if >=1 independently qualified
R gains >=10% vs W with exact provenance; PARTIAL for smaller gains or
limited qualification; NO_GO if no R qualifies; BLOCKED for missing
input SHA, audit mismatch or failed holdout. No population CI from
three preselected tasks.

## Stage P5: closeout

Run all focused CPU/GPU regressions, Ruff/Black/compileall, artifact
SHA check. Record implementation/environment revisions and known
negative results. Commit and push to experimental branch only.
NO PR; NO main merge; NO production default change.
Report final SHA, tests passed/skipped/failed, exact 15-cell retiming
table, all independent receipts, native phase findings and uncertainties,
report and manifest paths, and GO/PARTIAL/NO_GO/BLOCKED decision.
