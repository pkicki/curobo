# Extreme torque: a feasible alternative while full native cuRobo fails

**STATUS: IMPLEMENTED BUT NOT EXECUTED.** Source study frozen at
508d29305ba76e60c273c5857e7a4329d58a9f95. Preserve all historical
trajectories, manifests and results. Never invent a trajectory or claim a
GPU test has run.

## Scientific hypothesis

For a torque-constrained manipulator task, there may exist a geometrically
different, independently verified collision-free path with a feasible
execution time while the standard full cuRoboV2-Fast C-space planner,
built with the SAME constraints, fails to find a trajectory within a
substantial preregistered planning budget.

This supports torque-aware geometric search, not a claim of global
optimality or mathematical impossibility for every conceivable native
geometric path.

The strongest possible qualitative result has ALL these components:

1. A frozen valid task with the SAME start, goal, scene, model, physical
   velocity/acceleration/jerk limits and derated torque limits in both arms.
2. A verified alternate quintic path from the preexisting D geometry,
   time-dilated if necessary, passing independent dense+adaptive Pinocchio
   dynamics and world/self collision checks, including a *Scaled95* margin.
3. A full native planner using the new torque limits from the moment of
   model construction, not merely retiming an old path.
4. All 32 registered planner restarts fail to find a solution; no
   exceptions, incomplete restarts, missing versions or unverified outputs.
5. The original severe-stress full-planner negative control succeeded.
6. Stronger geometry evidence: an independent relaxed fixed-path torque LP
   numerically rules out ANY retiming of the ORIGINAL cuRobo geometric curve.

A finite sampled/adaptive audit does not prove continuous-time safety;
explicitly qualify the claim unless a conservative continuous-time
certificate is additionally implemented.

## Preregistered factors and time dilation

Synthetic effort factors relative to the source severe profile:

    1.000 0.950 0.900 0.850 0.800 0.750 0.700 0.650

Time dilation factors for the original already qualified D trajectory:

    1.0 1.05 1.10 1.20 1.35 1.50 1.75 2.0 2.5 3.0 4.0

Both lists are frozen BEFORE new native success/failure outcomes.

The same nominal robot inertias and all other physical limits remain
unchanged. This is synthetic actuator derating, NOT real Panda torque
rating or an added payload.

If q_rho(t)=q(t/rho), velocities, accelerations and jerk scale by
1/rho, 1/rho^2, 1/rho^3, while rigid-body inverse dynamics obeys:

    tau_rho = gravity(q) + (tau_original - gravity(q))/rho^2

This preserves geometry and collisions. Select shortest preregistered
slowdown passing a preliminary Scaled95 torque screen with 0.5% extra
headroom; REAUDIT the complete resulting motion independently afterward.
The slowdown duration must be disclosed; an extremely slow witness
proves existence, not a good performance tradeoff.

REJECT tasks with static endpoint gravity exceeding 85% of derated bounds
for the rest-to-rest endpoint contract. Never count trivially infeasible
endpoints as evidence of a planner failure. Require the original frozen D
trajectory to have a valid historical physical adaptive receipt.

## A useful necessary geometry test: relaxed pointwise torque LP

For path q(s), x=sdot^2>=0 and y=x'/2 free,

    tau(s) = g(s) + m(s)*x + a(s)*y

At EACH original cuRobo path point solve the numerical LP

    minimize epsilon>=0
    lower-epsilon*w <= g+m*x+a*y <= upper+epsilon*w
    x>=0, y free, w=(upper-lower)/2.

If the minimum expansion epsilon is strictly positive beyond numerical
uncertainty, no possible timing law can make THAT EXACT fixed geometric
path torque-feasible at that configuration. This is a sufficient
INFEASIBILITY witness for one geometry, not a continuous proof for
all geometries. A zero result is inconclusive. Use at least 257 interior
path points and refine any positive case with tighter tolerances.

Audit both the original cuRobo cubic geometry and the alternative
quintic D geometry using the independent Pinocchio coefficients.
Contradictions with independent W feasibility BLOCK the claim.

## Stage 0: freeze and validate source data

Pin exact cuRobo, CUDA, Pinocchio, Python and experiment Git commit.
Confirm original severe task inventory, robot limits, D source NPZ
checksums, source independent receipts, scene files and URDF hashes.
Original source root:

    research/joint_time_oracle/results/torque_stress_final_20261107/severe

Run the new CPU tests and relevant regression tests first.
Compile CUDA extensions before GPU experiments.

No background work, no PR, no merge to main.

## Stage 1: prepare independently screened alternatives

Set environment paths for the actual pinned Python executables:

    export SRC=$PWD/research/joint_time_oracle/results/torque_stress_final_20261107/severe
    export OUT=$PWD/research/joint_time_oracle/results/extreme_torque_witness_20261108

Run in independent Pinocchio environment:

    PIN_PY -m research.joint_time_oracle.extreme_torque_prepare --source "$SRC" --output "$OUT"

This creates factor arms named extreme_f1000, extreme_f0950 ...,
extreme_f0650, with task-specific profile checksums, all admitted/
rejected source cases, source NPZ checksums and preliminary witness
candidates. Preserve every factor-case outcome.

For each factor, e.g. the 0.850 regime:

    export REG=extreme_f0850
    export ARM=$OUT/$REG

    PIN_PY -m research.joint_time_oracle.cohort_validate \
      --input "$ARM" --methods W --torque-profile-dir "$ARM" \
      --report independent_sampled.json

    PIN_PY -m research.joint_time_oracle.extreme_torque_adaptive \
      --source "$SRC" --root "$OUT" --regime "$REG"

    PIN_PY -m research.joint_time_oracle.cohort_validate \
      --input "$ARM" --extra "$ARM" --methods W_adaptive \
      --torque-profile-dir "$ARM" --report independent_adaptive.json

    PIN_PY -m research.joint_time_oracle.extreme_torque_receipts \
      --source "$SRC" --root "$OUT" --regime "$REG"

W is not qualified until the final receipt explicitly passes physical,
Scaled95, collision, endpoints and checksum checks. A failed source
exact-spline replay invalidates W. Do not infer a certificate from the
initial gravity/dynamic time-dilation identity alone.

Repeat on ALL preregistered factors, not only promising ones.

## Stage 2: identify when original geometry is impossible

    PIN_PY -m research.joint_time_oracle.extreme_torque_path_lp \
      --root "$OUT" --source "$SRC" \
      --regime extreme_f0850 --samples 257

Record the worst path-progress coordinate and minimum normalized torque
relaxation on the original native cubic curve and on the alternate
quintic curve. Recompute any candidate positive slack with a denser
path grid, stricter LP tolerances and independently aligned dynamics.
Inspect dual evidence if available. A positive relaxed-LP lower bound is
a stronger geometric-infeasibility diagnosis than failure of an
optimization algorithm.

## Stage 3: full native planner challenges with identical constraints

Begin with THREE frozen cases:

    pilot_free_00
    pilot_table_00
    pilot_cluttered_03

First rerun the 1.000 negative control and verify native success,
profile and model parity before more extreme regimes. For suitable
endpoint-admitted, independently qualified witnesses, start with
f=.95,.85,.75. All other registered factor/case cells are reported
even if their witness fails or no GPU challenge is attempted.

Use native full planner from this branch; it now accepts extreme_fXXXX
regimes. It builds a new RobotCfg with derated limits BEFORE planner
construction and checks native TimeOpt's internal scaled limits:

    CUDA_PY -m research.joint_time_oracle.torque_full_planner \
      --root "$OUT" --regime extreme_f0850 \
      --case pilot_cluttered_03 --max-restarts 1

Increase to 4,16,32 on the SAME frozen seed/configuration and
same task, resuming existing work. Each restart has four internal
TrajOpt lanes and a fixed planning-attempt budget. Register all
success/failure/error statuses. NO changing the native timeout,
Newton iterations, seed set or effort limits after seeing failure.
If increasing the search budget as a diagnostic, call it a new arm.

If there are ANY native success candidates, independently audit them
with the existing tools:

    PIN_PY -m research.joint_time_oracle.torque_full_validate \
      --root "$OUT" --regime extreme_f0850

    CUDA_PY -m research.joint_time_oracle.torque_full_adaptive \
      --root "$OUT" --regime extreme_f0850

    PIN_PY -m research.joint_time_oracle.torque_full_validate \
      --root "$OUT" --regime extreme_f0850 --adaptive

    PIN_PY -m research.joint_time_oracle.torque_full_receipts \
      --root "$OUT" --regime extreme_f0850

A verified native success NEGATES a failure demonstration for that
source task and search budget. An unverified native candidate is
INDETERMINATE, not a forced failure.

An exception, initialization failure, invalid model, CUDA OOM, timeout,
or missing restart is NOT the qualitative native failure we seek.
Such trials stay in the inventory and must be fixed or separately reported.

If native fails all 32, independently attempt to retime the ORIGINAL
native geometric spline using the B IPOPT solver under the extreme
profile. A verified feasible B indicates a native TimeOpt/pipeline
weakness, not a necessary geometry change. An IPOPT failure does not
prove path infeasibility; the relaxed LP is the stronger check.

If possible preserve the native's REJECTED TrajOpt geometry before
TimeOpt and independently retime it to distinguish failure to discover
geometric alternatives from failure to solve retiming on generated paths.
Record geometric diversity; 32 seeds may still explore only one path.

## Stage 4: classification and HTML

    PIN_PY -m research.joint_time_oracle.extreme_torque_report \
      --root "$OUT" --control "$SRC" \
      --report reports/joint_time_oracle/extreme_torque --restarts 32

Main categories:
- NATIVE_BUDGET_FAILURE_WITH_FEASIBLE_ALTERNATIVE (ALL strict gates)
- STRONG_GEOMETRIC_FAILURE_EXAMPLE (above + original path LP infeasible)
- NATIVE_FOUND_FEASIBLE_TRAJECTORY
- NATIVE_RETURNED_UNQUALIFIED_CANDIDATES
- INCOMPLETE_NATIVE_BUDGET
- BLOCKED_NATIVE_EXECUTION_ERRORS
- ENDPOINT_STATIC_GRAVITY_REJECTED
- NO_INDEPENDENT_FEASIBLE_WITNESS
- BLOCKED_CONTRADICTORY_ALTERNATIVE_LP
- NATIVE_BUDGET_FAILURE_BUT_CONTROL_NOT_VERIFIED.

Do not report 20 tasks x 8 factor levels as 160 independent tasks.
This is an exploratory adversarial discovery experiment: all attempts
remain visible, and a highlighted witness must be described as an
observed finite-budget counterexample with independent sampled checks.

For each convincing case create a qualitative figure including:
- Original native and alternative geometric robot configurations and
  end-effector paths, obstacles and minimum clearance.
- The same original and extreme upper/lower actuator limits.
- Independent torque profiles for all seven joints, with limiting joint
  highlighted and source timestamps.
- Original cuRobo path's infeasible relaxed LP witness at progress s,
  including normalized margin and limiting joint inequalities.
- Native failed restarts with counts and failure stage; exact frozen
  method/model/configuration and SHA256.
- Alternative time dilation rho, execution duration, and full family
  qualification summary.
- Clear label that sampled checks are not continuous-time certificates.

## Promotion and next algorithmic experiment

Only if at least one strict finite-budget failure with an alternative
is obtained, compare torque-guided geometric rescue to a fixed-budget
random multi-candidate baseline. Use active torque LP/infeasibility
bottleneck locations as geometry search guidance and the same independent
fixed-path retimer to evaluate every candidate. Freeze task and GPU budget.
This separate phase evaluates how to SEARCH for feasible alternatives.

No PR; no main merge; production defaults unchanged.
New numerical measurements are unavailable until the agent runs these
experiments in the proper environments.
