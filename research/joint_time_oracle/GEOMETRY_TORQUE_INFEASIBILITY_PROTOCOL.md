# Geometry-dependent torque infeasibility — preregistered implementation and experiments

Status: IMPLEMENTED, NOT EXECUTED BY THE ASSISTANT. Historical severe and uniform-derating results are frozen; no new original-path LP, independently feasible alternative, or native failure has been measured. Do not create a PR or merge to main.

## Hypothesis and three separate claims

A. A specific nominal cuRobo cubic geometry is torque-infeasible under EVERY timing law: at some regular path point the independent Pinocchio relaxed torque LP has a strictly positive expansion even when x=s_dot^2>=0 is arbitrary and y=x'/2 is free.

B. A DISTINCT quintic geometric path with identical start, goal, obstacles, inertias and limits is demonstrably feasible using constructive, smooth rest-to-rest timing. Dense and exact adaptive Pinocchio and sphere checks with Scaled95 margin are required; these are sampled, not continuous-time certificates.

C. A NEW full native cuRobo planner, built with the exact same derated effort limits BEFORE planning, fails to find any independently feasible path within a preregistered finite budget of 32 restarts. This is separate from A+B: native may find a different path and succeed.

A+B is already a meaningful qualitative demonstration of why geometry matters under torque constraints. A+B+C is a stronger algorithmic example but never proves all possible native paths infeasible.

## Why new tasks instead of more uniform torque derating

The previous frozen study registered 160 source-task/factor cells. Endpoint gravity rejected 106; all other 54 were solved by native planning on the first restart. None gave positive relaxed original-path torque LP expansion. Uniform reduction preferentially made endpoint configurations infeasible.

This protocol changes BOTH the source distribution and actuator scenario:
- Generate low-gravity start/goal configurations with larger joint travel and a nominal straight-line interior gravity-demand barrier.
- Test a FIXED grid of synthetic JOINT-SELECTIVE actuator weakenings rather than a single uniform scalar.
- Keep position/velocity/acceleration/jerk limits, collision environment, model inertias, source endpoints and rest-to-rest contract identical.

Synthetic limits do NOT describe Panda manufacturer ratings and are NOT a payload experiment.

## Preregistered actuator scenarios and gates

Nine joint groups, 1-based indices:

    (2,3,4), (1,2,4), (2,4,6), (3,4,5), (1,3,5),
    (2,3,5,6), (1,2,3,4), (3,4,6,7), (2,4,5,7).

Frozen factors on selected joints: 0.20, 0.30, 0.40, 0.50, 0.65, 0.80.
All unselected joints retain their original torque limits. There are
54 scenarios per source task. The joint limits retain nominal sign
asymmetries. Scaled95 planning torque limits scale around zero.

Require endpoint gravity utilization <=80% of Scaled95 limits.
Reject all failed endpoint cells; never count as native planning failure.
Original native geometric path LP prescreen needs normalized required
torque-limit expansion >=0.01 before proposing alternate geometry.

The discovery is deliberately ADVERSARIAL, with complete attempt
history. Multiple scenarios per task are correlated. Do not treat them
as independent trials or estimate population-wide failure prevalence
from cherry-picked examples.

## Stage 0 — Freeze nominal source tasks and GPU geometry

First smoke-test old original 20-task source (already planned and
certified), in a separately labeled preview root. The scored discovery
source should be newly generated tasks with benign endpoint gravity.

Use Pinocchio environment:

    PIN_PY -m research.joint_time_oracle.geometry_feasibility_taskgen \
      --output research/joint_time_oracle/results/geometry_trap_source_20261108 \
      --seed 20261108 --per-population 10

This samples fresh start/goal pairs in free/table/cluttered scenes,
records every rejected proposal, enforces source-disjointness from
historical pilot and wide-task inventories, and requires larger travel,
low endpoint gravity and greater interior gravity demand along a
straight joint-space screening line. It does NOT inspect new derated
native outcomes. Record any incomplete task-generation quota.

Under the ORIGINAL nominal robot, run native source planning in CUDA:

    CUDA_PY -m research.joint_time_oracle.cohort_torque_wide_plan \
      --source research/joint_time_oracle/results/geometry_trap_source_20261108

Every available A10 native selected path must be captured exactly in
the usual native timing/geometry buffers. Independently audit the
source original A10/A40 trajectories:

    PIN_PY -m research.joint_time_oracle.cohort_validate \
      --input research/joint_time_oracle/results/geometry_trap_source_20261108 \
      --methods A10,A40 --report independent_sampled.json

Nominal source planning failures remain explicit attrition, not torque
geometry obstruction evidence. Freeze the resulting source inventory
and all nominal geometric-path and model hashes.

## Stage 1 — Discover fixed-path torque obstructions

Use a separate output root from source tasks and old frozen results:

    export SOURCE=$PWD/research/joint_time_oracle/results/geometry_trap_source_20261108
    export OUT=$PWD/research/joint_time_oracle/results/geometry_feasibility_discovery_20261108

First use a different preview root with 2–3 tasks and 10–20
alternative candidates, inspect numerical status and fix bugs. Do not
call that preview a scored cohort.

Freeze implementation and settings before the scored run:

    PIN_PY -m research.joint_time_oracle.geometry_feasibility_discover \
      --source "$SOURCE" \
      --reference research/joint_time_oracle/results/torque_stress_final_20261107/severe \
      --output "$OUT" --lp-samples 65 --max-candidates 100

For each source task and 54 preregistered actuator scenarios, the
discovery records all endpoint rejects, source-planner failures,
numerical LP results, candidate geometry attempts and constructed
witnesses. The original native cubic path uses the EXACT frozen native
geometry; no fitting or user-edited detours are allowed in that arm.

LP coefficient model:

    tau(s) = g(q(s)) + m(q,q_s,q_ss) * x(s) + a(q,q_s)*y(s),
    x=s_dot^2>=0, y=x'/2 unrestricted.

At each interior s minimize eps>=0 subject to
low-eps*w <= g+m*x+a*y <= high+eps*w, w=(high-low)/2.
A positive min eps is a NECESSARY pointwise obstruction for that
EXACT geometry under any timing law. A zero result is inconclusive.
Do not treat IPOPT nonconvergence as proof of geometry infeasibility.

## Stage 2 — Construct distinctly different feasible geometry

Only when original path LP margin is positive, propose up to 100
deterministically seeded quintic controls, including endpoint-clamped
smoothstep shapes, prior frozen C/D geometry if endpoint-compatible
and coordinated 0.25/0.50/0.80/1.20 rad detours.

Every alternate must differ by >=0.10 rad under arc-length-aligned
joint-space comparison. Screen joint position, analytic independent
Pinocchio robot sphere world/self collisions with >=5 mm sampled
clearance, and gravity effort <=88% of Scaled95 bounds.

For alternate q(s), use a CONSTRUCTIVE timing law:

    u=t/T, s(u)=10*u^3-15*u^4+6*u^5.

Compute qdot, qddot, jerk by exact chain rule. Endpoint velocity and
acceleration vanish. Dynamic rigid-body torques decrease with T^-2;
gravity does not. Scan registered durations:

    0.3,0.5,0.75,1,1.5,2,3,4,6,8,12,20 seconds.

A witness may be very slow; disclose its duration. A timing screen
is NOT a feasibility certificate. Save exact 8193-point witness,
candidate controls and SHA. Preserve all failed candidates and
rejection reasons. Do not fit a new curve after qualification.

## Stage 3 — Independent witness qualification and LP refinement

For each scenario arm with nonempty tasks.json, e.g.:

    export SCEN=j234_f0200
    PIN_PY -m research.joint_time_oracle.cohort_validate \
      --input "$OUT/$SCEN" --methods W \
      --torque-profile-dir "$OUT/$SCEN" --report independent_sampled.json

    PIN_PY -m research.joint_time_oracle.geometry_feasibility_adaptive \
      --root "$OUT" --scenario "$SCEN"

    PIN_PY -m research.joint_time_oracle.cohort_validate \
      --input "$OUT/$SCEN" --extra "$OUT/$SCEN" \
      --methods W_adaptive --torque-profile-dir "$OUT/$SCEN" \
      --report independent_adaptive.json

    PIN_PY -m research.joint_time_oracle.geometry_feasibility_receipts \
      --root "$OUT" --scenario "$SCEN"

The receipt requires BOTH independent dense and independent adaptive
physical and Scaled95 limits, position/velocity/acceleration/jerk/torque,
world/self sphere collision, endpoints, exact phase reconstruction,
profile SHA, task SHA and source geometry identity. Any missing
evidence fails closed. Sampling is not a continuous-time proof; for
a strong paper example, try validated interval error bounds on
all constraints and signed collision clearance.

Also independently rescan the ORIGINAL native cubic path:

    PIN_PY -m research.joint_time_oracle.geometry_feasibility_refine \
      --root "$OUT" --source "$SOURCE" --scenario "$SCEN"

This evaluates 257 and 1025 path points, refines the strongest region,
and checks a numerical dual lower bound, primal/dual residuals, and
duality gap. Require lower bound >0.01 and stable numerics.
Recompute independent RNEA coefficients and check q_s regularity at
the witness point. A numerically positive LP gap is a fixed-geometry
infeasibility diagnosis, not continuous-time geometric global proof.
Investigate any contradiction between alternative LP and W audits.

## Stage 4 — Freeze and execute full native challenge

Freeze only cases with independent Scaled95 W receipts and numerically
strong positive original-path LP evidence, BEFORE inspecting any
derated native-planner result:

    CUDA_PY -m research.joint_time_oracle.geometry_feasibility_challenge \
      --root "$OUT" --max-cases 3

Without --run the command writes an immutable challenge plan but does
not execute the native planner. Select eligible cases by deterministic
task/scenario IDs, not by future native success or duration.

Run staged nested native full-planner budgets with original frozen
seeds/configuration and SAME task/robot limits:

    CUDA_PY -m research.joint_time_oracle.geometry_feasibility_challenge \
      --root "$OUT" --max-cases 3 --restarts 1 --run

Escalate to 4,16,32 registered independent planner restarts, without
changing task selection, internal TrajOpt lanes (four), numerical
iterations or planning attempts. Avoid excessive full-cohort GPU costs
before nine representative preview tasks and 1/4 budgets are qualified.

Full planner uses torque_full_planner.py and builds the modified
RobotCfg BEFORE TrajOpt and TimeOpt. Verify nominal vs derated effort
and native scaled95 internal buffers; retain every native selected and
internally valid alternate lane. Do not silently count an OOM, API
exception, missing restart or failed bootstrap as native planner
feasibility failure.

Independently audit ALL native successes, on each scenario arm:

    PIN_PY -m research.joint_time_oracle.torque_full_validate \
      --root "$OUT" --regime "$SCEN"

    CUDA_PY -m research.joint_time_oracle.torque_full_adaptive \
      --root "$OUT" --regime "$SCEN"

    PIN_PY -m research.joint_time_oracle.torque_full_validate \
      --root "$OUT" --regime "$SCEN" --adaptive

    PIN_PY -m research.joint_time_oracle.torque_full_receipts \
      --root "$OUT" --regime "$SCEN"

A qualified native trajectory is success even if geometrically
different from the original. If the native planner fails, attempt
independent retiming of the original geometry as a diagnostic; IPOPT
failure itself is not geometric impossibility. Recover pre-TimeOpt
rejected native geometries if possible to separate generating
inappropriate paths from numerical TimeOpt failure. Report distinct
geometric routes, not just restart counts.

## Stage 5 — Independent outcome classification and HTML

Build the result from raw evidence:

    PIN_PY -m research.joint_time_oracle.geometry_feasibility_report \
      --root "$OUT" --source "$SOURCE" \
      --report reports/joint_time_oracle/geometry_feasibility --restarts 32

Main distinct outcomes:
- NOMINAL_SOURCE_UNAVAILABLE and ENDPOINT_REJECTED
- ORIGINAL_LP_NOT_INFEASIBLE
- NO_CONSTRUCTIVE_ALTERNATIVE_DISCOVERED
- NO_QUALIFIED_ALTERNATIVE
- NO_STRONG_ORIGINAL_GEOMETRY_INFEASIBILITY_PROOF
- GEOMETRY_OBSTRUCTION_BUT_NATIVE_REPLANNED_SUCCESSFULLY
- GEOMETRY_OBSTRUCTION_NATIVE_BUDGET_FAILURE
- GEOMETRY_OBSTRUCTION_NATIVE_BUDGET_INCOMPLETE
- GEOMETRY_OBSTRUCTION_NATIVE_EXECUTION_BLOCKED
- GEOMETRY_OBSTRUCTION_NATIVE_FAILURE_CONTROL_UNVERIFIED.

A strong native-budget failure requires verified original nominal
control, strong dual-validated original geometry LP obstruction,
different independently qualified Scaled95 alternative, and 32
documented FAILED parent restarts with zero native successes, errors
or missing attempts. A native success invalidates a failure claim,
but does not invalidate the fixed original-path obstruction.

For each example, visualize original geometry and alternative in a
REAL robot scene, the actual LP constraint point and limiting joints,
joint torque traces vs limits, signed collision clearances, witness
duration and all attempt statuses. Generate from SHA-linked artifacts;
do not draw hypothetical trajectories.

## Stage 6 — Fresh source-disjoint confirmation

After development choose at most THREE fixed scenario IDs based only
on DEVELOPMENT results, and freeze them before fresh planning data.

Generate new source-disjoint tasks under a new seed and ID prefix:

    PIN_PY -m research.joint_time_oracle.geometry_feasibility_taskgen \
      --output research/joint_time_oracle/results/geometry_trap_confirm_20261109 \
      --seed 20261109 --prefix confirm_geom \
      --exclude-root research/joint_time_oracle/results/geometry_trap_source_20261108 \
      --per-population 10

Run nominal native planning and independent A10/A40 validation.
Run geometry_feasibility_discover on a NEW confirmation output root,
passing ONLY the preselected repeated --scenario-id arguments. No
case-specific actuator tuning and NO source-case cherry-picking. Then
repeat witness checks, refined LP and native challenges on the
source-disjoint cases. Report all exclusions and failures.

Target at least 20 nominal-success source tasks across two or more
obstacle populations. If fewer qualify, report the shortfall instead
of altering scored thresholds. The discovery result is qualitative
existence evidence; confirmatory frequency/coverage requires this
prospectively frozen source cohort.

## Optional future algorithm after real counterexamples

Compare:
- full native C-space planning,
- random waypoint geometry + same TOPP retimer,
- LP-dual/retiming-infeasibility-guided waypoint geometry + same retimer,
- offline C/D reference,

under matched total GPU/CPU budgets and identical independent checks.
The LP dual identifies joints and progress intervals to target; it
does not directly give a guaranteed collision-free geometry update.
Do not report performance advantages without controlled comparisons.

## Implemented files (branch-specific)

geometry_feasibility_core.py, geometry_feasibility_scene.py,
geometry_feasibility_taskgen.py, geometry_feasibility_discover.py,
geometry_feasibility_adaptive.py, geometry_feasibility_receipts.py,
geometry_feasibility_refine.py, geometry_feasibility_challenge.py,
geometry_feasibility_report.py, tests/test_joint_time_geometry_feasibility.py.

Also changed research-specific torque_active.read_profile to support
vector joint effort scaling and cohort_torque_wide.py to support fresh
task IDs and historical-exclusion roots. Existing uniform profiles
and all production solver defaults are unchanged.

No new numeric results or GPU tests have been run in this branch by
the assistant. Agent must run test suite, fix failures and publish
only source-grounded evidence. No PR; no merge to main.


## Pilot versus extended challenge manifests

A challenge plan cannot be edited after it is frozen. Use a separate
plan filename for a three-case preflight and for later expanded eligible
cases, preserving both as evidence. Example:

    CUDA_PY -m research.joint_time_oracle.geometry_feasibility_challenge \
      --root "$OUT" --max-cases 3 \
      --plan-file "$OUT/native_challenge_pilot.json"

After qualifying the pilot, an extended arm must be frozen separately
without discarding the pilot:

    CUDA_PY -m research.joint_time_oracle.geometry_feasibility_challenge \
      --root "$OUT" \
      --plan-file "$OUT/native_challenge_extended.json"

Then execute each plan with the corresponding --plan-file argument and
the registered 1/4/16/32 restart budgets. The challenge code checks the
frozen profile, task, base+adaptive witness and refined LP receipt hashes
before launching native planning. It refuses changed inputs.

## Local NPZ evidence checksums

The compact HTML/JSON report is committed, whereas large raw trajectories
can remain local and ignored by Git. Create a SHA256 inventory including
the source nominal files:

    PIN_PY -m research.joint_time_oracle.geometry_feasibility_manifest \
      --root "$OUT" --source "$SOURCE" \
      --report reports/joint_time_oracle/geometry_feasibility

This writes artifact_manifest.json with relative paths, sizes and SHA256.
Rebuild after completing/scoring the campaign. Do not include the
generated report manifest itself among its own source files. Retain
raw NPZ archives for later reproduction.
