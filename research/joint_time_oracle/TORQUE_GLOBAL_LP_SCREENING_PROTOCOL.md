# Global torque-LP screening with common path proposals

**Research implementation; new experiments NOT EXECUTED.**
Source evidence remains frozen at
afe9f293d176145cdddc7074144403d6a4c37202.
No PR; no main merge; no production default changes.

## Scientific question

Does a NECESSARY pointwise torque feasibility LP make GLOBAL multi-path
search more computationally efficient by rejecting impossible paths
before expensive trajectory timing?

Previous registered LP-dual-gradient search: 0/3 rescues; random-local:
0/3; random-global: 2/3 (proposals 37 and 105; both 20-s trajectories).
Native pre-TimeOpt captures showed many intrinsically torque-infeasible
geometries and post-TimeOpt collision/dynamics masks rejecting all.
Do NOT infer a specific internal failure stage from a pre-TimeOpt curve
and a post-TimeOpt validation mask.

This test isolates the SCREEN rather than the direction optimizer.

## Fixed cases, input contract, study scope

- Development only: trap_cluttered_08.
- Frozen algorithm holdouts: confirm_geom_table_07 and
  confirm_geom_cluttered_02.
- Torque scenario j1234_f0300: synthetic joints 1–4 effort restriction
  to 30% nominal, with identical original model, endpoints and obstacles.
- 128 globally diverse proposals per task, shared by both algorithms.
- Interim proposal budgets: 16, 32, 64, 128.

The previously qualified source W trajectory is NEVER an algorithm
initialization, proposal, or result. Use it only for source admission.
These are three deliberately selected qualitative cases, NOT an
unbiased sample of robot tasks.

## Global proposal generator, no LP information

Module: torque_global_bank.py.

From the ORIGINAL nominal native cubic curve, fit common 22-control
quintic splines. Freeze a case-specific deterministic seed. Generate
a mixture of independent global detours, random global walks and
two-lobed perturbations with radii 0.25,0.5,0.8,1.2,1.6,2.4 radians.
Clamp first/last three controls to preserve endpoint position and
path derivatives. Do not use retiming, LP or W information to select
proposals. Save exactly 128 proposals in one bank NPZ with SHA256.

The earlier random-global method used a BEAM with LP-based scoring.
This NEW proposal generator has a different distribution, so the earlier
2/3 success rate is NOT its expected success rate. Do not tune proposal
radii after observing results. If no candidate succeeds, report it.

## Comparison: same candidates and order

Arm no_lp:
- 129-sample independent position/world/self collision check.
- Gravity torque utilization <=0.88 of Scaled95 joint effort.
- Geometric difference >=0.10 radians from original native curve.
- If cheap screens pass, attempt exact quintic smoothstep timing.

Arm lp_gate:
- IDENTICAL bank, order, cheap screen and constructive timing.
- After cheap screens, before timing, solve relaxed necessary torque LP
  at frozen 17- and 65-point interior path grids.
- If at ANY point a verified dual lower bound of normalized required
  torque relaxation exceeds 0.005, reject THIS candidate for timing.
- A zero LP gap is INCONCLUSIVE; proceed to timing.
- If LP solve/numerical certificate fails, DO NOT falsely reject the
  candidate: proceed to timing and log the uncertainty.

LP:
  tau = g(q) + m(q, q_s, q_ss) x + a(q,q_s) y,
  x = s_dot^2 >= 0, y = x'(s)/2 unrestricted.
  Minimize epsilon >= 0 subject to
    lower - epsilon * halfwidth <= tau <= upper + epsilon * halfwidth.

Positive dual-checked relaxation is a NECESSARY infeasibility witness
for that EXACT path, not all paths or continuous-time trajectory
safety certification.

Common constructive time law:
  q(t) = q(s(t/T)),
  s(u)=10u^3-15u^4+6u^5, u=t/T.
Durations 0.3,0.5,0.75,1,1.5,2,3,4,6,8,12,20 seconds.

All constraints remain identical: position, velocity, acceleration,
jerk, signed torque, independent Pinocchio RNEA, world and self sphere
collisions, rest-to-rest endpoints. Retimer screens and 8193-point W
are PRELIMINARY; only independent dense+adaptive physical and Scaled95
receipts qualify a success.

Both arms stop at their first preliminary witness. A no_lp qualified W
without an lp_gate qualified W on the SAME bank is a critical safety
investigation, NOT an acceptable screening tradeoff.

## Phase P0: code and numerical validation

Read:
- torque_global_bank.py
- torque_global_screen.py
- torque_global_freeze.py
- torque_global_report.py
- torque_global_retime.py
- native_timeopt_boundary.py
- native_pretimeopt_probe.py (opt-in post TimeOpt capture)
- torque_full_planner.py (opt-in diagnostic flag)

Run new test_joint_time_global_lp_screening.py and prior joint-time
oracle, geometry and native CUDA regression suites. Check Black, Ruff,
compileall, git diff --check, source SHA inventories.

Verify LP dual sign/normalization, asymmetric effort limits, exact
quintic chain rule, common bank equality/order, correct CPU/RNEA time
boundaries, provenance, fail-open-on-LP-error, W receipt fail-closed,
and no unintended production solver changes.

Use a SEPARATE debugging root before the scored research root.
Precompile cuRobo CUDA extensions before native tests. Use pinned
independent Pinocchio CPU environment; do not use Eagle CPU.

## Phase P1: development case, scored bank and two arms

    export OUT=$PWD/research/joint_time_oracle/results/torque_global_lp_screening_20261109

    PIN_PY -m research.joint_time_oracle.torque_global_bank \
      --root "$OUT" --case trap_cluttered_08 \
      --seed 20261109 --budget 128

    for ARM in no_lp lp_gate; do
      PIN_PY -m research.joint_time_oracle.torque_global_screen \
        --root "$OUT" --case trap_cluttered_08 \
        --arm "$ARM" --seed 20261109 --budget 128
    done

For EACH arm with nonempty tasks.json:

    PIN_PY -m research.joint_time_oracle.cohort_validate \
      --input "$OUT/$ARM" --methods W \
      --torque-profile-dir "$OUT/$ARM" \
      --report independent_sampled.json

    PIN_PY -m research.joint_time_oracle.geometry_feasibility_adaptive \
      --root "$OUT" --scenario "$ARM"

    PIN_PY -m research.joint_time_oracle.cohort_validate \
      --input "$OUT/$ARM" --extra "$OUT/$ARM" \
      --methods W_adaptive --torque-profile-dir "$OUT/$ARM" \
      --report independent_adaptive.json

    PIN_PY -m research.joint_time_oracle.geometry_feasibility_receipts \
      --root "$OUT" --scenario "$ARM"

Do not invent receipts for missing W. Preserve all failures and source
SHA. Record bank-generation elapsed time SEPARATELY and equally for
both arms, per-arm CPU search time including cheap checks, LP, RNEA and
constructive timing, and offline independent qualification time
separately.

## Phase P2: immutable development freeze and holdouts

After finishing development and any debugging, freeze algorithms,
source file hashes, common development bank SHA and both development
arm outcome/independent receipt records:

    PIN_PY -m research.joint_time_oracle.torque_global_freeze \
      --root "$OUT" --seed 20261109 --budget 128

Do not change source algorithm, LP tolerance, candidate order,
duration schedule or results after freezing. Changes require a
separately named new experiment root.

    for CASE in confirm_geom_table_07 confirm_geom_cluttered_02; do
      PIN_PY -m research.joint_time_oracle.torque_global_bank \
        --root "$OUT" --case "$CASE" --seed 20261109 --budget 128
      for ARM in no_lp lp_gate; do
        PIN_PY -m research.joint_time_oracle.torque_global_screen \
          --root "$OUT" --case "$CASE" --arm "$ARM" \
          --seed 20261109 --budget 128 --holdout
      done
    done

Repeat the independent W validation sequence for each arm. Check the
development receipt and holdout freeze did not change. Report the
full denominator of three tasks, not only successful cases.

## Phase P3: native same-stage TimeOpt diagnosis (separate)

Earlier native pre-TimeOpt geometric paths were compared with
post-TimeOpt native masks. That does NOT identify a failure stage.

Research-only probe optionally captures timing controls/integrated
times, and ACTUAL post-TimeOpt sampled q,qd,qdd,jerk. CUDA sampling
and tensor copies perturb timing, so this is NEVER a latency
qualification. Rejected lanes can fail to sample: label UNKNOWN.

Prepare isolated native inputs (never overwrite historical traces):

    PIN_PY -m research.joint_time_oracle.native_pretimeopt_prepare \
      --output research/joint_time_oracle/results/native_posttimeopt_probe_20261109

Preflight one restart on the cluttered case:

    CUDA_PY -m research.joint_time_oracle.torque_full_planner \
      --root research/joint_time_oracle/results/native_posttimeopt_probe_20261109 \
      --regime j1234_f0300 --case confirm_geom_cluttered_02 \
      --max-restarts 1 --lanes 4 --max-attempts 5 \
      --capture-pre-timeopt --post-timeopt-samples 513

Use SAME candidate lane, time samples and torque profile when comparing
native masks to independent physics. Independent post-state audit:

    PIN_PY -m research.joint_time_oracle.native_timeopt_boundary \
      --native-arm research/joint_time_oracle/results/native_posttimeopt_probe_20261109/j1234_f0300/native_full \
      --profile-arm research/joint_time_oracle/results/native_posttimeopt_probe_20261109/j1234_f0300 \
      --case confirm_geom_cluttered_02

Native collision masks can disagree with sampled independent sphere
checks due to padding, collision model, time grid, or validation
definition. DO NOT call the difference a bug without like-for-like
reproduction. Compare with uninstrumented status, NOT elapsed time.

Pay special attention to the two cluttered native geometries with
zero relaxed LP gap and sampled geometric validity; a zero LP result
does not establish a feasible timing. Investigate their prior
unqualified IPOPT retiming as an explicitly separate diagnostic.
If pilot adds no information, do not automatically run another 32
restarts. If successful, expand in NEW isolated roots.

## Phase P4: optional faster timing AFTER independent W

A 20-second feasible witness may be much slower than a useful robot
execution. Only once W is independently qualified:

    PIN_PY -m research.joint_time_oracle.torque_global_retime \
      --root "$OUT" --arm no_lp \
      --case confirm_geom_table_07 --maxiter 200 --grid 513

This runs IPOPT on FROZEN alternative quintic geometry with the SAME
synthetic torque limits; it initializes speed squared from constructive
smoothstep W. Solver convergence is NOT independent qualification.
Audit proposed P path separately using cohort_validate (method P)
and adaptive sampling if supported. Preserve solver failures and
primal margins. Do not include this offline polish time in the
original global screen search latency.

## Phase P5: HTML, SHA manifest and GO/NO-GO

    PIN_PY -m research.joint_time_oracle.torque_global_report \
      --root "$OUT" \
      --report reports/joint_time_oracle/torque_global_lp_screening

Report per case/arm and cumulative 16/32/64/128 budgets:
- exact bank SHA and common proposal order;
- LP exclusions, LP solver errors and timing calls saved;
- total CPU time and RNEA counts by cheap/LP/timing components;
- first preliminary W, independent physical and Scaled95 receipts;
- physical family/collision/headroom margins, W duration;
- optional independently qualified P duration and separate CPU time;
- native same-stage post-TimeOpt masks versus independent checks,
  unknown cases and captured source digests;
- frozen source task and torque scenario evidence hashes.

Generate raw artifact manifest using
geometry_feasibility_manifest.py and this output root. Keep large
raw NPZ locally with checksums; commit compact report and figures.

GO_RESEARCH only if LP gating preserves independently verified
feasibility and measurably improves actual CPU time under the matched
bank, not merely number of skipped timing calls. NO-GO if LP overhead
dominates or discards W. INCONCLUSIVE if no arm finds a W with the
pre-registered new generator or evidence is incomplete. No population
confidence intervals from three selected tasks.

Final: commit implementation and validated compact results, report
SHA, test counts, negative outcomes and scope limitations. No PR,
no merge to main, no production default changes.
