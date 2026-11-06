# Frozen multi-problem TimeOpt versus joint-path/time study

Status: IMPLEMENTATION SCAFFOLD ONLY. Cohort experiments have NOT yet run.
The historical single-task numbers are NOT evidence of cohort generalization.

## Methods and hypothesis

Compare the same frozen task, robot, payload, scene, endpoint derivatives and
limits for all methods:

- A10: cuRoboV2-Fast TimeOpt default 10 iterations.
- A40: same original geometry and TimeOpt input with 40 iterations.
- B: independently optimized path speed on exactly A's geometry. Use
  pointwise inequalities, IPOPT and a mathematically justified treatment of
  stationary-path speed controls. Record control-bound activity.
- C: multistart joint path-and-time optimization.
- D: retime exactly C's geometry with the SAME solver, constraints and
  accuracy used for B. Reject D if path conversion changes geometry beyond
  the predeclared tolerance.

A10-to-A40 isolates native iteration budget. A40-to-B isolates fixed-path
timing solver/representation. B-to-D compares *geometry* under the same
timing method. B-to-C is exploratory: mixed representations and solver
convergence prevent assigning that gap solely to geometric optimization.

Historical diagnostic-only example: A10 0.412843 s, B restored 0.398760 s,
C 0.401694 s. A40 previously reported ~0.399673 s but its faster result
needs independent dense validation. These are NOT cohort statistics.

## Cohort registration and source independence

Pilot: 20 distinct tasks (6 obstacle-free; 8 table/simple; 6 cluttered).

Confirmation: 100 NEW, source-disjoint tasks (30 obstacle-free; 35
table/simple; 35 cluttered). Do not reuse pilot start-goal pairs, near
duplicates, the historical diagnostic trajectory, or source-identical
obstacle geometry unless that is explicitly the population definition.

Freeze task IDs, YAML/URDF, scenes, payloads, joint ordering, endpoints,
planner seeds and fixed per-task solver budget BEFORE observing results.
For generated tasks, validate goal and start positions and log rejected
proposals as well as accepted tasks. Do not silently exclude failed
planners or numerical solvers.

Run 20-pilot first. Only launch 100-confirm after method/derivative/validator
issues are resolved and the exact implementation/configuration is frozen.

## Same two acceptance gates for all methods

PHYSICAL: original position, velocity, acceleration, jerk, torque and
collision limits; start/end positions, velocities and accelerations;
dense reconstruction; independently evaluated Pinocchio torques with
aligned inertials and independent world AND self-collision checks.

SCALED95: PHYSICAL plus 95%-scaled dynamic limits.

Both gates must be reported. Record sampled versus continuous-time safety;
sampled feasibility is not a continuous-time safety certificate. If a
trajectory fails SCALED95, store a separate, explicitly identified
restored trajectory and its increased duration. Do not overwrite the
original data.

A local feasible nonconverged candidate may count as a feasible duration
incumbent but not as evidence of optimality. Document IPOPT KKT status,
active artificial squared-speed coefficient bounds, and mesh sensitivity.

A10 and A40 must be verified after native TimeOpt sampling, independently
of the solver's internal success flags.

## Statistics

Primary question: B->D duration difference under the same fixed-path
timing solver, directly testing whether improved geometry helps.

Secondary: A10->A40, A40->B, B->C, A10->C.

Report for EACH contrast and feasibility gate:
- attempted N and paired independently validated N
- duration differences in ms, paired percentage changes, median and mean
- bootstrap 95% CI on median; fractions >0%, >1%, >5%, >10%
- negative differences, missing counts and reasons
- per scene population; independent confirmation versus exploratory pilot
- solver convergence, active bounds and compute budget
- improvement distribution and selected illustrative trajectory plots.

Do not pool pilot into the confirmation headline or report a missing
pair as zero improvement. Do not imply global optimality.

## Input format for the statistics tool

Create one JSON object per attempted task in a JSONL file (or a JSON array):

{
  "case_id": "confirm_table_0007",
  "cohort": "confirm",
  "population": "table",
  "task_sha256": "<frozen task checksum>",
  "methods": {
    "A10": {
      "duration_s": 0.5,
      "solver_converged": false,
      "control_bound_active": false,
      "verification": {
        "physical": true,
        "scaled95": false,
        "endpoint": true,
        "dense": true,
        "collision_independent": true,
        "dynamics_independent": true
      }
    }
  }
}

The duration above is an ILLUSTRATIVE PLACEHOLDER, not an observation.
Include A40, B, C, D as they are attempted; omit unrun results or record
errors separately. Missing gate flags are false by construction.
The collision_independent flag requires an independently evaluated
world and self collision contract, not cuRobo zero collision cost.

To generate statistics and an offline HTML table:

python -m research.joint_time_oracle.cohort_statistics --input /path/to/cohort.jsonl --output /path/to/cohort_report

This produces summary.json and index.html without requiring CUDA.
The existing historical HTML report must remain intact until new cohort
results are available.

## Required executor work

The current reconstruct_baseline.py, fixed_path_pointwise.py and validator
scripts hard-code the single diagnostic task. DO NOT run them repeatedly
and interpret the outputs as different tasks. First expose task-specific
input/output paths and move their core logic to reusable callable APIs.

For every task:
1. run A10 and export original B-spline geometry and TimeOpt buffers;
2. run A40 on the SAME geometry and original timing input;
3. run B with independent variable-speed optimization on that geometry;
4. run multistart C on the same task, then D on C's EXACT geometry;
5. independently validate full physical-time states, RNEA and collision;
6. store full trajectories, solver histories, failure cases and checksums;
7. emit the JSONL case record and update the report.

A pair only contributes to physical/scaled95 statistics when BOTH outputs
pass that identical verification gate.

No PR, no merge into main, no production default changes.
