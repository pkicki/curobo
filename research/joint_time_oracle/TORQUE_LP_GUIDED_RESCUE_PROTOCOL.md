# Torque-LP-dual geometric rescue — implementation and experimental protocol

**STATUS: implementation committed; NEW EXPERIMENTS NOT EXECUTED by the assistant.**
Frozen source result: 591e76a34abb97b9c68137e483d597cbdafc6649.
No PR; no merge to main; no production defaults changed.

## 1. Scientific hypothesis and frozen tasks

The previous study found three task/scenario combinations where an original
nominal cuRobo cubic geometric path has a positive, dual-checked relaxed
pointwise torque infeasibility LP (no timing x=sdot^2>=0, y=x'/2 free can
satisfy all joint bounds at the obstructing path point), while a different
quintic path passes independent sampled physical/Scaled95 feasibility and
the full native cuRobo planner failed to return a verified trajectory
in 32 registered restarts.

Frozen development tasks, all with synthetic derated torque limits
j1234_f0300 (joints 1-4 at 30% of nominal):
  trap_cluttered_08         prior witness 4 s, dual LP bound 0.0679
  confirm_geom_table_07    prior witness 6 s, dual LP bound 0.2153
  confirm_geom_cluttered_02 prior witness 4 s, dual LP bound 0.5148

These three examples, including two prior source-disjoint examples, have
ALREADY BEEN SEEN by the method designer. They are now DEVELOPMENT tasks,
not independent confirmation of a newly developed algorithm. Existing
alternative W controls and trajectories are restricted to eligibility
checking and final oracle reference. NEVER initialize or steer a search
method with previously found W controls.

Hypothesis: An active torque-infeasibility LP supplies local joint/path
information that can improve geometrically distinct feasible-path discovery
against random geometric perturbations, when measured under fair CPU-time
and proposal-count budgets. A subsidiary study instruments native cuRobo
to distinguish pre-TimeOpt geometry generation from timing rejection.
A partial conservative whole-domain kinematic check improves the
verification evidence; collision and torque remain sampled unless
formally bounded separately.

## 2. Three controlled search arms and mathematical guidance

All arms begin from exactly the same 22-control quintic least-squares
approximation of the frozen original nominal cubic geometry.
The fit is NOT the exact original; report its max path deviation and LP
feasibility at proposal 0. If fitting alone makes it feasible, this is
initialization success, NOT an improvement from LP guidance.

Each method preserves first/last THREE quintic controls exactly, retaining
endpoint position and zero path derivatives. All share:
 - 33 interior progress LP screening points;
 - top three LP bottlenecks, beam size 4;
 - search step radii .16/.30/.50/.80 rad;
 - exact same independent Pinocchio RNEA + joint/world/self collision
   screening, gravity demand and geometric-distinctness thresholds;
 - constructive rest-to-rest phase s(u)=10u^3-15u^4+6u^5 and frozen
   duration ladder 0.3/.5/.75/1/1.5/2/3/4/6/8/12/20 s;
 - no oracle or source W geometry in initialization.

  dual: the relaxed local LP is min eps subject to
    low-eps*w <= g + m*x + a*y <= high+eps*w,
    x>=0, y free, eps>=0.
    With primal x,y and upper/lower LP multipliers, the local
    envelope sensitivity w.r.t. geometric controls is approximated by
        d eps/dtheta = (lambda_upper-lambda_lower)^T
                       d[g+m*x+a*y]/dtheta.
    Finite-difference the independent RNEA at the limiting path
    progress point, propose a negative-gradient control update and
    REEVALUATE the actual LP, collisions and all screens. Near LP
    active-set changes this is only a heuristic, not a differentiable
    global time-optimal retimer.

  random_local: same LP bottleneck, same local quintic basis support,
    same step radii, with random search direction.

  random_global: same LP bottleneck and step radii, but all interior
    control points may be perturbed randomly.

Scored nested proposal budgets: 16/32/64/128, with a 128-proposal
registered run giving nested prefix outcomes. Stop on first
constructively retimed W candidate. If its later independent audit
fails, the result remains an UNQUALIFIED rescue; do not selectively
restart that method or extend the frozen scored budget.

FAIRNESS: LP-dual finite differences require substantially more RNEA
work per proposal. Equal proposals do not mean equal computation.
Record total CPU wall time (including guidance), number of RNEA calls
and LP evaluations. Evaluate wall-time as well as proposal efficiency.
If only proposal count improves but wall time regresses, report the
tradeoff instead of calling the new method faster.

## 3. Phase P0: source/configuration audits and tests

Verify branch and starting source SHA. Confirm local large NPZ input
archives match the frozen artifact manifest. Pin aligned Python,
Pinocchio, SciPy, cuRobo, CUDA, GPU, CPU affinity and thread count.

Run:
    pytest tests/test_torque_lp_guided_rescue.py
    pytest tests/test_joint_time_geometry_feasibility.py
    pytest tests/test_joint_time_extreme_torque.py
    pytest tests/test_joint_time_torque_mechanisms.py

Also run relevant CUDA TimeOpt and TrajOpt tests, compileall, Ruff,
Black, git diff --check, regression-source SHA checks.

Before scoring, review numerical issues:
 - LP primal/dual residuals and dual-sign finite difference against
   a simple analytical synthetic LP test;
 - correct Pinocchio path coefficient definitions a=M(q)q_s,
   m=M(q)q_ss+C(q,q_s)q_s, g(q);
 - identical asymmetric effort limits, .95 scaling, collision and
   endpoint contracts in all arms;
 - no source W geometry leakage into fitted initial controls;
 - correct fitted-vs-exact original path deviation reported;
 - deterministic seeds, frozen parameters and batch/test ordering;
 - independent witness audit and output file SHA consistency.

Use a PREVIEW results root to fix code bugs before scored runs. Preview
outcomes may not be pooled with the scored cohort.

## 4. Phase P1: preview and frozen scored 3-task campaign

Set:
    export PREVIEW=$PWD/research/joint_time_oracle/results/torque_lp_guided_preview_20261108
    export OUT=$PWD/research/joint_time_oracle/results/torque_lp_guided_rescue_20261108

On the preview root, one case and 16 proposals per arm:
    PIN_PY -m research.joint_time_oracle.torque_guided_rescue \
      --root "$PREVIEW" --arm dual --case trap_cluttered_08 \
      --budget 16 --seed 20261108

Repeat for random_local and random_global. Verify source checksums,
endpoint parity, q(s) fitting, LP/gradient directional signs,
collision/gravity screens, cost logs and preliminary W generation.
Freeze implementation and numerical parameters BEFORE the scored run.

Scored run: same 3 frozen tasks and same proposal budget:
    PIN_PY -m research.joint_time_oracle.torque_guided_rescue \
      --root "$OUT" --arm dual --budget 128 --seed 20261108

    PIN_PY -m research.joint_time_oracle.torque_guided_rescue \
      --root "$OUT" --arm random_local --budget 128 --seed 20261108

    PIN_PY -m research.joint_time_oracle.torque_guided_rescue \
      --root "$OUT" --arm random_global --budget 128 --seed 20261108

Every arm writes per-task search_result.json, best_controls.npz,
torque_profile.json, full per-proposal ledger and optional W trajectory.
The new runner binds original native cubic buffer SHA, source task SHA,
effort profile SHA, source refined LP SHA, original independently
qualified W receipt SHA, and 32 frozen failed-native restart results
BEFORE commencing the controlled search.

Additional runs with new seeds must use SEPARATE results roots and be
registered explicitly. Counterbalance arm order across repeated
blocks. A method may be faster per proposal but worse in wall time;
always present both cost bases.

## 5. Phase P2: independent candidate W qualification

A constructive screening result is NOT a rescue success.
For every arm with nonempty tasks.json, independently run:

    PIN_PY -m research.joint_time_oracle.cohort_validate \
      --input "$OUT/dual" --methods W \
      --torque-profile-dir "$OUT/dual" --report independent_sampled.json

    PIN_PY -m research.joint_time_oracle.geometry_feasibility_adaptive \
      --root "$OUT" --scenario dual

    PIN_PY -m research.joint_time_oracle.cohort_validate \
      --input "$OUT/dual" --extra "$OUT/dual" \
      --methods W_adaptive --torque-profile-dir "$OUT/dual" \
      --report independent_adaptive.json

    PIN_PY -m research.joint_time_oracle.geometry_feasibility_receipts \
      --root "$OUT" --scenario dual

Repeat the four commands for random_local and random_global.
All five dynamic constraint families, endpoint conditions, self/world
collision and original/adaptive exact analytic replay must pass both
physical and Scaled95 checks. Check every SHA. Any missing/error/mismatch
receipt is an UNQUALIFIED rescue.

Attempt conservative whole-domain kinematic spline hull verification:
    PIN_PY -m research.joint_time_oracle.torque_kinematic_hull \
      --root "$OUT/dual"
Repeat for random arms. The hull uses exact spline derivative
coefficient convex bounds and analytic maxima of smoothstep derivatives
to obtain SUFFICIENT whole-domain position/velocity/acceleration/jerk
bounds. Failure to pass a loose hull is UNKNOWN, not a demonstrated
constraint violation. Float-padding is not a formal interval proof.
Torque and collisions are NOT continuously certified by these bounds.
For publication, attempt truly validated interval RNEA and signed
sphere collision bounds on a selected example, or state the limitation.

## 6. Phase P3: fail-closed report and paper figures

    PIN_PY -m research.joint_time_oracle.torque_guided_report \
      --root "$OUT" \
      --output reports/joint_time_oracle/torque_lp_guided_rescue

The report checks the exact source registry match across arms, source
task/profile/geometry provenance, independent W receipts, geometric
controls SHA and exact adaptive replay. Report every case and arm,
proposal count and wall time to first qualified W, total CPU time,
RNEA work and 16/32/64/128 proposal-prefix success. Three adversarial
cases cannot yield a meaningful random-population confidence interval.
Report complete failure reasons and do not quote unverified prescreens
as positive results.

Use torque_guided_figures.py for qualified finalists. It plots
actual model-aligned right-gripper paths, the original-path LP torque
conflict and independent RNEA torque traces from saved W trajectories.
Include actual robot and obstacle geometry and LP dual constraints in
the final paper-quality figures; do not substitute hypothetical
illustrations. Keep file SHA provenance and separate unqualified plots.

## 7. Phase D1: instrument the native failure stage

This stage is a diagnostic RERUN, not editing the original frozen
32-restart failure evidence. Native planning time measurements are
NOT valid when instrumentation synchronizes CUDA.

Start with:
    CUDA_PY -m research.joint_time_oracle.torque_native_stage_trace \
      --case confirm_geom_cluttered_02 --restart 0

The diagnostic wrapper intercepts TrajOptSolver._finalize_result,
snapshots get_interpolation_input BEFORE native TimeOpt, and
captures post-TimeOpt cubic controls and validation masks only if
those buffers actually exist. The snapshot is saved even on an
unsuccessful final plan. Every stage is SHA-linked.

If replay success/failure status disagrees with the frozen result,
label NONREPRODUCIBLE; do not attribute the original failure to any
internal stage. If pre-TimeOpt path data cannot be accessed, label
NO_PRETIMEOPT_GEOMETRY_OBSERVABLE instead of inferring infeasibility.

After passing preflight, repeat for all registered restarts 0..31
on all three cases, with the same seeds and planner configuration
in a separate diagnostic root.

Then independent offline path audit:
    PIN_PY -m research.joint_time_oracle.torque_native_stage_audit \
      --case confirm_geom_cluttered_02 \
      --root research/joint_time_oracle/results/torque_lp_native_stage_trace_20261108

For every reconstructed native cubic lane, compute independent
path-local torque LP obstruction, joint limits and world/self collision.
Record identical versus distinct geometries. A path that passes the
relaxed LP is INCONCLUSIVE, not necessarily time-retimable. Try IPOPT
fixed-path retiming of rejected native lanes when possible; a certified
feasible timing on such a lane implicates native TimeOpt/pipeline
rather than a geometrically inadequate candidate.

Outcome categories must include:
   pre-TimeOpt never observable;
   pre-TimeOpt available but TimeOpt geometry unavailable;
   geometric path violates necessary torque LP;
   geometrically collision-infeasible;
   relaxed torque LP inconclusive;
   independent retiming passes/fails;
   replanning outcome changed under instrumentation.

Do not claim a precise native failure stage unless the evidence identifies it.

## 8. Phase Q: source-disjoint qualification of the NEW algorithm

The three scored tasks are DEVELOPMENT, even though two were source-
disjoint in the PREVIOUS feasibility experiment: they were exposed
during the design of this new method.

After method freeze, generate a NEW nominal task population and
torque-feasibility discovery using the previous
GEOMETRY_TORQUE_INFEASIBILITY_PROTOCOL, without adjusting scenario
or method parameters in response to new search outcomes. Freeze the
eligibility cohort before applying guided/random methods. Include all
tasks (including ineligible tasks) in the source denominator.

Provide an independent registry JSON of this form:
    {
      "schema_version": 1,
      "cases": [
        {
          "case_id": "<fresh_case_id>",
          "source_root": "research/joint_time_oracle/results/<frozen_source>",
          "discovery_root": "research/joint_time_oracle/results/<frozen_study>",
          "scenario_id": "j1234_f0300"
        }
      ]
    }

The runner supports --registry, and pins SHA256 of the registry and
every native/LP/effort/W source evidence file. It does not initialize
from W controls. Run all three arms in NEW result roots using identical
algorithm budgets and multiple registered independent seeds.

Target at least 20 source-disjoint nominal-source tasks, and report the
full exclusion/admission counts. If only a few meet the strong original
LP/alternative/native-failure contract, report the actual denominator
without extending until success is observed. Compare independent
qualified rescue success and wall time; any publication claim about
generalization must come from this prospectively frozen cohort.

## 9. What counts as success (and negative outcomes)

Development target (NOT a guaranteed finding): torque-dual guidance
produces independently qualified Scaled95 rescues on at least 2/3
frozen cases within 128 proposals, AND is at least competitive in wall
time versus both random baselines. A benefit only in proposals but not
wall time must be reported as a tradeoff.

If it fails, diagnose whether the signed LP-dual gradient is poor
near active-set changes, collides with an obstacle, suffers from
an inaccurate quintic source fit, or explores too small a trust region.
Preserve the negative result. A revised multiscale beam, line search,
multi-bottleneck objective or collision-aware gradient is a NEW,
separately preregistered experiment.

Do not confuse the envelope derivative of a local LP with a gradient
of globally optimal execution duration. This project studies
TORQUE FEASIBILITY search; not deadline filtering or duration racing.

## 10. Finalization

Run tests, lint, compileall, independent GPU/CPU qualification,
report/artifact provenance validation. Use existing
geometry_feasibility_manifest.py to inventory raw NPZ/JSON files with
paths, sizes and SHA256. Large binary trajectories may stay local;
compact JSON, source, protocol, HTML and figure metadata are committed.

Report exact branch and final SHA; passed/failed/skipped tests; source
tasks, approved/failed solver budgets, first qualified rescue proposal,
total wall time/RNEA counts, complete Scaled95/physical validations,
native stage capture/attribution, kinematic whole-domain PASS/UNKNOWN,
source-disjoint confirmation and all remaining limitations.

DO NOT create PRs. DO NOT merge into main. Production defaults unchanged.

## Implemented files

- torque_dual_guidance.py: independent pointwise LP, dual-checked local
  gradient via RNEA, no requirement to solve a time-optimal program.
- torque_guided_rescue.py: matched initialization, local/global random
  baselines, dual-guided search, candidate W generation, SHA provenance.
- torque_guided_report.py: independent receipt-gated comparisons.
- torque_native_stage_trace.py: opt-in pre-TimeOpt native snapshots.
- torque_native_stage_audit.py: independent audit of rejected geometry.
- torque_kinematic_hull.py: conservative whole-domain kinematic bounds.
- torque_guided_figures.py: real robot and RNEA provenance figures.
- tests/test_torque_lp_guided_rescue.py: pure LP, recorder and hull tests.
- reports/joint_time_oracle/torque_lp_guided_rescue/index.html:
  explicitly NOT EXECUTED until the agent qualifies new results.

No numerical tests or GPU campaigns have been run by the assistant.
