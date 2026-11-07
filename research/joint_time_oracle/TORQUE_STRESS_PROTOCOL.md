# Torque-active joint path-and-time study

**Implementation status:** experimental code committed; GPU campaigns **not run in
this branch**. No performance/feasibility improvement is claimed. Historic
20-task pilot remains unchanged.

## Scientific objective and design

The historic 20-problem pilot showed that independent fixed-path timing was
consistently faster, but the maximum effort utilization on B was only about
21.5–66.2% of the original limits (median ~30.9%). Velocity, acceleration
and jerk were near their 95%-scaled planning limits. This study asks whether
geometry provides more value when **torque actually constrains time**.

Use two intentionally distinct populations:

1. **Controlled actuator-derating stress test:** reuse the original 20 frozen
   geometric tasks but replace *effort limits only*, keeping identical
   inertials, gravity, collision geometry, endpoint contract and all other
   dynamic limits. New constraints define a **synthetic derated actuator**,
   not the real Panda rating or a loaded Panda.
2. **Wider-excursion tasks:** generate 10 new source-disjoint endpoint pairs
   per scene (free/table/cluttered), screen for significant joint travel,
   endpoint validity and nontrivial static effort. Then run the existing
   native planner and independent sampled torque export to freeze a distinct
   source cohort. Do not mix this second population with the historical data
   until it is independently frozen and validated.

Within each population evaluate **moderate** and **severe** torque constraints.
Both torque regimes on one source trajectory are *paired*, not two independent
random tasks. Never pool 20×2 as n=40 independent observations.

## Exact derating contract

The reference is the historical A40 motion, *not B/C/D*, to prevent choosing
stress limits based on the outcome of the new methods. For each source task:

    u_A40 = max_t,j signed_side_utilization(tau_A40(t), original_effort_j)
    u_static = max_j signed_side_utilization(tau_g(q_start),tau_g(q_goal))

Choose a **uniform** scalar multiplier, preserving all joint-specific upper
and lower asymmetries:

    alpha = max(0.15, u_A40 / target, u_static / 0.85)

If alpha > 0.95, or the resulting A40 peak is <0.78 of the new effort limit,
reject and document the task; do not silently retune it.

Two targets:

- moderate: target=0.90; original A40 peak is about 90% of derated limit;
- severe: target=1.10; original A40 trajectory nominally violates the
  new physical effort limit, so a genuinely different timing or path must
  restore it.

Limits are [alpha * original_lower, alpha * original_upper] and straddle 0.
The 0.85 gravity reserve at endpoints is mandatory. Feasibility at
intermediate configurations is deliberately *not* assumed. If a path is
gravity-infeasible in its interior, a different geometry may be required.

All methods optimize with an additional 0.95 planning scale and are checked
separately against both the synthetic physical limit and its 95%-scaled
version. Existing original robot effort limits should be reported only as a
secondary reference, **not** the stress-test physical contract.

**This is not an attached payload experiment.** A genuine payload benchmark
requires a jointly modified cuRobo and Pinocchio inertial model (mass,
center of mass, inertial tensor), matched robot sphere collision geometry,
and unchanged real actuator effort limits. Do not add mass to one model only.

## Methods

A10/A40: CUDA native TimeOpt with 10 or 40 iterations, using the EXACT
original TrajOpt geometry and interpolation inputs under frozen derated
effort bounds. The original planner is NOT reoptimized under new effort
limits; this is intentionally a fixed-geometry control.

B: independent IPOPT fixed-path timing of the same original geometry.

C: joint quintic path-time optimization. Torque-stress runner enables more
expensive multistart (default 14/22/34 controls; four starts; 300 iterations;
trust radius 1.5 rad and seed noise 0.2 rad), but this is STILL a local
search around one geometric seed. The agent must validate C's CUDA
derivatives, convergence and independently qualified status. Never
advertise C as a global minimum-time oracle.

D: independent IPOPT fixed-path timing on C's EXACT quintic geometry,
using the same solver accuracy and effort limits as B.

The primary geometry comparison is B vs D. B vs C additionally diagnoses
coupled-solver timing weaknesses. For stronger geometric exploration,
generate diverse collision-free trajectories from distinct planning seeds,
alternative IK branches and obstacle routes, and optionally run CEM/CMA-ES
with B/D as the inner retimer. Fix the total compute budget and preserve
every attempted candidate.

## Frozen protocol and execution

Use one clean code commit and run all tasks on that commit. Compile cuRobo
CUDA extensions before the experiment. Do not use the notebook's CPU
environment for the CUDA phase.

Preflight (two existing Python environments, customize these paths):

    export CUDA_PY=/home/pkicki/anaconda3/envs/curobo-timeopt/bin/python
    export PIN_PY=/tmp/oracle-ipopt-env/bin/python
    export TORQUE_OUT=$PWD/research/joint_time_oracle/results/torque_stress_20261107
    export SOURCE=$PWD/research/joint_time_oracle/results/cohort_pilot_isolated_20261106

Generate original-source torque stress cases with **independent Pinocchio**:

    "$PIN_PY" -m research.joint_time_oracle.cohort_torque_prepare \
      --source "$SOURCE" --output "$TORQUE_OUT"

Run a three-source-case preflight in one regime (CUDA setup first):

    "$CUDA_PY" -m research.joint_time_oracle.cohort_torque_execute \
      --regime moderate --output "$TORQUE_OUT" --source "$SOURCE" \
      --cuda-python "$CUDA_PY" --pin-python "$PIN_PY" \
      --cuda-visible-devices 1 --shim-dir /tmp/joint_oracle_compat \
      --max-cases 3

The device number and compatibility shim must be verified for the target
workstation; do not treat shim-based runs as latency comparisons.

After the three-case preflight succeeds, run both regimes **without**
--max-cases, with --resume. Inspect all subprocess logs, solver statuses,
feasibility restoration and geometry replay, not just exits.

Independently evaluate every available output at dense samples:

    "$PIN_PY" -m research.joint_time_oracle.cohort_validate \
      --input "$TORQUE_OUT/moderate" --extra "$TORQUE_OUT/moderate" \
      --methods A10,A40,B,C,D \
      --torque-profile-dir "$TORQUE_OUT/moderate" \
      --report independent_sampled.json

Repeat validation for severe.

Aggregate into independent sampled and STRICT independent/adaptive views:

    "$PIN_PY" -m research.joint_time_oracle.cohort_torque_report \
      --regime moderate --output "$TORQUE_OUT"

This generates per-arm index.html, torque_summary.json, torque_sampled.jsonl
and torque_qualified.jsonl. The strict view intentionally has **zero
qualified pairs until adaptive qualification receipts are present**.

Generate NEW wider excursion sources on the CUDA workstation:

    "$CUDA_PY" -m research.joint_time_oracle.cohort_torque_wide \
      --output /tmp/torque_wide_proposals --per-population 10

Run the existing native planning harness against the frozen proposal file
IN A DISTINCT OUTPUT DIRECTORY:

    "$CUDA_PY" -m research.joint_time_oracle.cohort_explore \
      --frozen-tasks /tmp/torque_wide_proposals/tasks.json \
      --output /tmp/torque_wide_planned

Then use cohort_validate on that source to write a40_pin_torque.npz and
run cohort_torque_prepare with --source /tmp/torque_wide_planned and a
NEW output root (not TORQUE_OUT). Pass that source through --source when
calling cohort_torque_execute, so B/C/D load the same frozen geometry.

## Adaptive and independent qualification (required before claiming gain)

The provided cohort_validate diagnostics use 8193 dense samples,
model-aligned Pinocchio torques, independently computed sphere/scene
clearance and endpoint residuals. These are NOT continuous-time certificates
and must not be counted as fully qualified by default.

The coding agent must extend the existing adaptive validation scripts to:

- accept task-specific derated torque profiles and output directories;
- include all spline knots, active dynamic peaks, near-contact points and
  bracketing neighborhoods; carry over SHA256 source/profile linkage;
- verify world AND self collision using the independent sampled sphere model;
- crosscheck torque and endpoint conditions with an independent Pinocchio
  model; isolate non-smooth CUDA derivative discrepancies;
- produce per-case/method adaptive_qualification.json receipts with explicit
  physical/scaled95/collision/endpoint booleans, profile hash and reason.

Only pass --adaptive-receipts to cohort_torque_report once those receipts
have been generated from real validation. Missing evidence fails closed.
Do not call sampled checks a continuous-time proof.

A solver termination status and a feasible incumbent are distinct. Collect
IPOPT primal/dual/KKT residuals, arbitrary speed coefficient bound activity,
SLSQP C convergence, collocation refinement and runtime per method.
If B/D fail, retain the failure. Never substitute an unverified original
torque-limit trajectory as though it satisfied derated limits.

## Analysis and acceptance

Report separately per regime, scene and source population:

- attempted/admitted tasks, rejected endpoint/profile proposals, failures;
- independently sampled physical/Scaled95 and adaptive-qualified counts;
- torque utilization distribution after retiming; fractions >=0.85 and >=0.95
  of the DERATED physical limits; verify torque becomes limiting;
- paired relative duration (%) and absolute duration (ms) for A10->A40,
  A40->B, B->C, and especially B->D;
- uncertainty for signed paired medians (bootstrap 95% CI), fraction of
  paired cases improved (>0, >1, >5, >10%), and explicit failures;
- geometry path length, turning, collision clearance, velocity/jerk/torque
  utilization, solver convergence and compute budget.

If torque utilization remains below 0.85 in most qualified cases, label
that regime **NOT TORQUE ACTIVE** rather than interpreting geometric results.

Pre-register decision criteria: at least 10 valid matched B/D physical pairs
per informative regime and adequate coverage of at least two scene classes
for an initial exploratory conclusion. Larger source-disjoint confirmation
follows only if this is met and C's derivative/convergence issues are
resolved. No significance claims based on n=3 preflight.

Do not manufacture measured improvements or globally optimality claims.

## Files implemented here

- torque_active.py: pure-CPU signed/asymmetric normalization, calibration,
  source hashing, static endpoint reserve, fail-closed profile read/write.
- cohort_torque_prepare.py: freeze moderate/severe profiles against source A40
  and independent endpoint gravity; preserve every exclusion.
- cohort_torque_native.py: rerun native A10/A40 on exact geometry with
  explicit torque bounds, preserve failures and exact timing buffers.
- cohort_torque_execute.py: clean-tree, per-task execution of A/B/C/D,
  provenance, logs, source-selectable and resumable.
- cohort_torque_wide.py: generate larger-excursion, source-disjoint endpoint
  tasks and rejected proposals.
- cohort_retime.py/cohort_joint.py: inject identical case-specific limits.
- cohort_validate.py: independent sampled Pinocchio/sphere verification with
  frozen derated limits.
- cohort_torque_report.py: separately label sampled diagnostic versus
  adaptive-qualified comparisons, and report torque activity.
- tests/test_joint_time_torque_active.py: CPU-only smoke tests.

No PR. No production defaults changed. No full confirm cohort launched.
