# Torque-LP-dual guided geometric rescue and native pre-TimeOpt attribution

**STATUS: IMPLEMENTED, NOT EXECUTED.** Starting evidence is frozen at commit
591e76a34abb97b9c68137e483d597cbdafc6649. Do not create PRs, merge
to main, change production defaults, or alter the historical trajectories.

## 1. Scientific objective

The previous study found three distinct geometry-dependent feasibility
counterexamples under a synthetic joint-selective actuator scenario with
joints 1–4 at 30% of nominal torque limits (j1234_f0300).

- trap_cluttered_08: original fixed-geometry LP dual bound 0.0679,
  independently sampled feasible W duration 4 s, 32/32 native failures.
- confirm_geom_table_07: LP dual bound 0.2153, W duration 6 s,
  32/32 native failures.
- confirm_geom_cluttered_02: LP dual bound 0.5148, W duration 4 s,
  32/32 native failures.

The geometric obstruction rules out *every timing* on the EXACT
original nominal native cubic path, subject to numerical model/LP
checks. It does not exclude every other path. Native failure is
finite-budget evidence, not necessarily failure in the geometric stage.
W passed sampled dense/adaptive independent physical and Scaled95
checks, not a continuous-time formal safety certificate.

Goals:
(A) test whether LP duals guide geometric proposals toward independently
    verified feasible alternatives more efficiently than random search;
(B) capture geometry and masks BEFORE TimeOpt even on native failures,
    independently audit those rejected curves and classify the stage.

## 2. Guided direction and fair comparison

For fixed q(s), x=sdot^2>=0 and y=x'/2 free,

    tau(s) = g(q) + m(q,q_s,q_ss)*x + a(q,q_s)*y.

Solve the necessary local LP

    minimize epsilon>=0
    low-epsilon*w <= g+m*x+a*y <= high+epsilon*w
    x>=0, y free; w=(high-low)/2.

At an obstructed path point, collect primal x,y and dual multipliers for
upper and lower torque inequalities. The local envelope derivative of
the LP optimum w.r.t. quintic controls theta is approximately

    grad_theta epsilon =
      (lambda_upper-lambda_lower)^T
      d_theta[g+m*x+a*y].

Compute the latter with central finite differences and independent
Pinocchio RNEA at the bottleneck only. This gradient is heuristic under
active-set changes. Reevaluate the FULL LP after any geometry change.

All three arms share:
- SAME 22-control quintic least-squares fit of nominal native cubic;
  never initialize from the known successful alternative W path.
- SAME 33 interior LP points; same three most obstructed bottlenecks.
- SAME four-element beam, endpoint-clamped controls (first and last 3
  fixed), deterministic seed and proposed radii 0.16,0.30,0.50,0.80 rad.
- SAME independent Pinocchio/scene collision and gravity screening.
- SAME constructive smoothstep time law, fixed duration grid, and
  independent dense/adaptive candidate verification.

Arms: dual gradient; random_local at same LP bottleneck support; and
random_global over all movable controls.

Score all candidate proposals. Track full CPU wall time, LP solves,
RNEA calls, actual torque LP gap, joint/collision checks and retiming.
The dual gradient costs additional RNEA work. A proposal-count win alone
is not an efficiency result; compare time-to-qualified-feasibility.

No algorithm is guaranteed to find a rescue. No positive new results
may be claimed before fresh execution.

## 3. Evaluation split and freeze

Treat ONE original counterexample (trap_cluttered_08) as algorithm
DEVELOPMENT only. The two task IDs beginning with confirm_geom are
held-out evaluations of the frozen algorithm. They were previously
used to establish geometric feasibility but must not guide any
algorithm hyperparameter choice in this follow-up.

Use one scored root per frozen algorithm/budget; maximum 128 proposals
stops at first preliminary candidate. Summarize cumulative gates at
16,32,64,128 using the proposal trace, not retroactively changed
settings. The registration includes scenario and source SHA256.

If a new source-disjoint problem set is added, define an immutable
JSON registry schema_version=1 with a cases array of case_id,
source_root,discovery_root,scenario_id; every case must have independent
positive original LP, W feasible and a completed native failure record.
Registry SHA is part of the experiment config. Do not mix source regimes
or allow unqualified tasks into this controlled search.

## 4. Stage 0 — tests, numerical gates, CUDA qualification

Read these modules: torque_dual_guidance, torque_guided_rescue,
torque_guided_freeze, torque_guided_report, native_pretimeopt_probe,
native_pretimeopt_prepare and native_pretimeopt_analyze.

Run tests/test_joint_time_torque_guided_rescue.py, the historical
geometry_feasibility tests, and relevant CPU Pinocchio/CUDA
planner and TimeOpt tests. Check Ruff, Black, compileall, git diff --check.
Compile native CUDA extensions before GPU experiments.

Before the scored run test the LP dual sign, primal/dual gap,
finite-difference geometry Jacobian, signed asymmetric torque bounds,
envelope direction at stable active sets, q_s regularity, and fallback
on degenerate gradients. Test exact endpoint preservation; source W
must NEVER leak into the candidate initialization. Fix bugs BEFORE
the development freeze and use a separate unscored pilot output root.
Retain negative numerical observations.

## 5. Stage 1 — development case (only ONE)

In pinned model-aligned Pinocchio Python, set:

    export OUT=$PWD/research/joint_time_oracle/results/torque_lp_guided_rescue_20261108

Run all arms using the same frozen maximum budget:

    for ARM in dual random_local random_global; do
      PIN_PY -m research.joint_time_oracle.torque_guided_rescue \
        --root "$OUT" --arm "$ARM" --budget 128 \
        --case trap_cluttered_08 --seed 20261108
    done

Record exact source hash, fitted-path error, every proposal's LP,
clearance/gravity, gradient fallback, total RNEA count, candidate wall
time and all constructed W candidates. Do not count candidate screens
as feasibility success.

### Independent qualification for EACH arm

For nonempty arm/tasks.json:

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

Every W must independently pass physical and Scaled95 joint position,
velocity, acceleration, jerk, torque (aligned Pinocchio RNEA), endpoint,
world sphere and self sphere constraints on original 8193 + adaptive
exact-analytic samples. Verify task/effort/geometry/time archive SHAs.
Dense/adaptive sampled checks are NOT continuous-time certificates.
Absent W or receipt means NO QUALIFIED SUCCESS.

## 6. Stage 2 — freeze and evaluate held-out cases

After finishing development and freezing the method, run:

    PIN_PY -m research.joint_time_oracle.torque_guided_freeze \
      --root "$OUT" --budget 128 --seed 20261108

This records method source SHA and all three development arm results.
Any later code/development change blocks the holdout until a separate
new experiment is explicitly registered.

Run WITHOUT any hyperparameter changes:

    for ARM in dual random_local random_global; do
      PIN_PY -m research.joint_time_oracle.torque_guided_rescue \
        --root "$OUT" --arm "$ARM" --budget 128 --seed 20261108 \
        --holdout --case confirm_geom_table_07 \
        --case confirm_geom_cluttered_02
    done

Repeat the independent validation commands for every arm, and preserve
both holdout successes and failures. The heldout tasks are a small fixed
qualification set, not independent proof of population generalization.

## 7. Stage 3 — capture native pre-TimeOpt geometries

Create an ISOLATED native probe input without touching historical raw
planning evidence:

    PIN_PY -m research.joint_time_oracle.native_pretimeopt_prepare \
      --output research/joint_time_oracle/results/native_pretimeopt_probe_20261108

In pinned CUDA Python, compile extensions first. Begin with one
restart on development case, then reproduce up to 32 frozen attempts
on all three task IDs if the probe passes basic parity checks:

    CUDA_PY -m research.joint_time_oracle.torque_full_planner \
      --root research/joint_time_oracle/results/native_pretimeopt_probe_20261108 \
      --regime j1234_f0300 --case trap_cluttered_08 \
      --max-restarts 1 --lanes 4 --max-attempts 5 \
      --capture-pre-timeopt

Research-only probe is installed AFTER warmup and wraps the native
TrajOptSolver finalizer without changing its returned result. It
records pre-TimeOpt all-seed JointState interpolation inputs, optional
pre masks/seed costs, exact native cubic geometry controls and masks
even when TimeOpt returns no accepted trajectory. Early failures
before finalizer are classified NOT REACHED. All torch-to-CPU copies
synchronize, invalidating native latency comparisons. Run a separate
NONINSTRUMENTED control and compare output/failure status, not time.

Independent post-mortem in Pinocchio environment:

    PIN_PY -m research.joint_time_oracle.native_pretimeopt_analyze \
      --native-arm research/joint_time_oracle/results/native_pretimeopt_probe_20261108/j1234_f0300/native_full \
      --profile-arm research/joint_time_oracle/results/native_pretimeopt_probe_20261108/j1234_f0300 \
      --samples 257

For every available candidate lane, audit exact cubic path pointwise
torque LP, collision, source masks, TimeOpt masks and geometric diversity.
LP zero is inconclusive; LP positive suggests THAT curve is
infeasible under any timing. If any LP-feasible candidate is found,
independently retime and validate it. Do not label solver stage
responsible from unavailable masks or failure alone.

## 8. Stage 4 — compare, report, preserve evidence

Run:

    PIN_PY -m research.joint_time_oracle.torque_guided_report \
      --root "$OUT" \
      --output reports/joint_time_oracle/torque_lp_guided_rescue

Report for each of three task IDs and three arms:
- attempted proposals and valid W count at budgets 16/32/64/128;
- FIRST independently qualified W, not preliminary screen;
- CPU time to first W and all RNEA/LP calls including gradient overhead;
- duration, geometry deviation, physical/Scaled95 family margins;
- source SHA and original native-to-quintic fitting error;
- all fallback, collision and numerical failure reasons.

No confidence intervals from three task IDs; do not pool correlated
records. Any positive claim about general efficiency should be tested
later on a truly new source-disjoint cohort with frozen algorithm and
a fixed common wall-time budget, not only common proposal budgets.
A negative outcome is a useful conclusion.

For native attribution, report per attempt/lane counts:
- did TrajOpt reach the pre-TimeOpt finalizer?
- was there pre-TimeOpt cubic geometry?
- which curves failed relaxed LP and which failed collision?
- which validation masks and stages are known vs unavailable?
- did the uninstrumented control reproduce the same failure?
- whether geometries were genuinely distinct.

Generate figures using REAL robot trajectories and environments only;
preserve both raw NPZ and compact checksum manifest. Committed report
must not invent new metrics, must label N/A if not executed, and
must retain prior results on main unchanged.

## 9. Optional independent validation and algorithm follow-up

If a useful guided-search improvement occurs, design a NEXT registered
ablation: sign-reversed dual, dual shuffled across joints, gravity-only
gradient, or random directions with matched RNEA/CPU cost. Then evaluate
on >=20 new nominally solved source-disjoint tasks with frozen actuator
constraints. A local dual sensitivity is not a globally valid geometry
planning solution, and cannot guarantee obstacle avoidance or success.

For a publication-quality counterexample, also consider conservative
continuous-time bounds on position/velocity/accel/jerk and interval
inverse-dynamics/collision bounds. Do not claim continuous-time
certification from sample checks.

## Final deliverables

Return branch and final commit SHA; CPU and CUDA test counts; actual
development and holdout per-arm outcomes at 16/32/64/128 budgets;
qualified witness checksums/physical margins; CPU timings/RNEA work;
native pre-TimeOpt stage/lane attribution and missing-stage reasons;
HTML report and artifact manifest paths; source environment revisions;
limitations and NO-GO/GO decision. No PR, no merge, no production
defaults changes.
