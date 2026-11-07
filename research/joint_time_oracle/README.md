# Joint path-and-time opportunity study (experimental)

**Research question:** How much shorter could the trajectory become if cuRoboV2-Fast
were allowed to change *joint geometry* while simultaneously optimizing execution
time, instead of passing a fixed geometric path to TimeOpt?

## Direct nonlinear program

For normalized physical time \(u=t/T\), jointly optimize all free quintic
joint-spline controls and a free positive final time \(T\):

\[
q(u)=\sum_i B_i^{(5)}(u)c_i,\quad
\min_{c,T>0} T .
\]

Joint velocity, acceleration and jerk are spline derivatives divided by
\(T,T^2,T^3\), respectively. The four *TimeOpt* constraint families are
velocity, acceleration, jerk and torque. We also constrain joint positions,
world collision and self collision. We use the same cuRobo inverse dynamics,
payload inertial parameters, scene and sphere-collision masks as the planner.
Six boundary conditions per joint (initial/final position, velocity,
acceleration) are eliminated analytically from the spline coefficients.

The first seed is the selected cuRoboV2-Fast *retimed* trajectory at its
retimed duration. Additional nonlinear-program starts perturb interior controls
within a bounded coefficient neighborhood. The backend is selectable:
SciPy SLSQP (default, no extra compiled dependency) or cyipopt/IPOPT
(optional local interior-point solve with a conservative smooth approximation
of the worst sampled dynamics margin). The IPOPT Jacobian is assembled from
cuRobo/Torch automatic differentiation; for this small pilot it is dense,
not yet a sparse mesh-collocation implementation. Both modes require
an independent dense post-solve check. The solver minimizes duration plus a
large penalty on a nonnegative collision slack; only solutions with zero
collision slack and zero collision cost can pass the dense audit.

**This is a local nonlinear optimization prototype, not a global minimum-time
oracle.** An audited solution with smaller duration establishes an achievable
improvement and hence a *lower bound on the improvement ceiling*. Failure to
find improvement proves nothing about the true ceiling. This restriction
matters especially around obstacles and when different path homotopy classes
are involved. The trust region currently explores a local neighborhood only.

## Run on a CUDA workstation

Use a pinned cuRoboV2-Fast fork installation with a CUDA-capable GPU.
From the repository root:

    python -m pytest -q tests/test_joint_time_oracle.py

    python -m research.joint_time_oracle.run_study \
      --output /tmp/joint_time_ceiling \
      --robot franka.yml --scene collision_table.yml \
      --number 8 --seed 14 --controls 18 --attempts 4 --maxiter 250

    # After installing cyipopt and libipopt in a compatible CUDA environment:
    python -m research.joint_time_oracle.run_study \
      --output /tmp/joint_time_ipopt --solver ipopt \
      --number 8 --seed 14 --controls 18 --attempts 4 --maxiter 400

The default cohort samples deterministic goal perturbations from the robot's
home configuration. For a fixed study cohort, write a JSON list:

    [
      {"start": [joint values in active order], "goal": [joint values]}
    ]

and pass --tasks /path/to/tasks.json. The order is recorded in the manifest.
Study the same scene, tool, payload and actuator limits for each method.
Default optimizer dynamic limit scale 0.95 matches the published TimeOpt
configuration; validation records the original physical limits.

Results:
- manifest.json: input settings, GPU and model metadata.
- per_task.json: *all* attempted tasks and solver starts, original TimeOpt
  duration, local solution, feasibility, endpoint contract and failures.
- summary.json: the reported paired statistics on **mutually verified**
  baselines and local solutions, with exclusions counted separately.
- task_NNNN.png: joint profiles versus normalized physical time.
- task_NNNN_fast.npz / task_NNNN_joint.npz: original and optimized joint
  trajectories; SHA256 hashes are recorded per task for independent replay.

Treat results as **NOT QUALIFIED** until CUDA runs and an independent
dynamics/collision checker confirm them. A dense sampled cuRobo check is not
an analytic continuous-time safety certificate or an independent
cross-engine verification.

## Protocol for a publishable test

1. Freeze fork commit, robot URDF/YAML, joint ordering, payloads, scenes and
   task IDs. Include realistic collision-rich and high-torque cases. Log
   failures instead of deleting them.
2. Verify cuRobo RNEA and collision custom gradients against finite
   differences on small, isolated inputs. This prototype's GPU autograd has
   not been exercised in the current environment.
3. Use the same physical initial and terminal position, velocity, and
   acceleration for both candidates. Compare only paths with matching
   contracts and valid original-limit audits.
4. Vary the joint spline resolution (14,18,26,34 controls), number of
   solver starts, and duration initialization. Import multiple
   *geometrically distinct* cuRobo seeds, not only random perturbations
   of one trajectory. Check whether the best duration stabilizes.
5. Re-evaluate candidate trajectories on finer grids (8x and 16x), including
   knot locations, swept world/self collision and independent Pinocchio
   dynamics. Reject any trajectory with new peak violations.
6. For any shorter joint trajectory, retime its **fixed optimized geometry**
   using TimeOpt/K107 before attributing savings to changes in geometry
   rather than a suboptimal baseline timing law.
7. Report the *paired* reduction \(100(1-T_{\rm joint}/T_{\rm Fast})\)
   together with confidence intervals and all missing/failed cases.
   Positive reduction is a demonstrated lower bound on possible savings,
   not the global optimum gap.

## A stronger numerical oracle for the full study

This first implementation uses SciPy SLSQP with Torch CUDA autodiff and
multiple local starts to determine whether there is an opportunity worth
pursuing. For an expensive high-quality reference, I recommend an independent
**free-final-time sparse direct-collocation** formulation in CasADi + IPOPT,
or SNOPT, using Pinocchio rigid-body dynamics and explicit time-varying
trajectory states. Increase the mesh and run many homotopy-distinct
initializations. When possible, generate certified lower bounds on small
obstacle-free cases. IPOPT and SNOPT find *local* optima, not guaranteed global
optima. Distinguish best feasible *upper bounds on optimum duration* from
certified lower bounds.

**Current scope:** same four retiming constraint families as cuRoboV2-Fast
TimeOpt. No torque-rate constraint is implemented in this oracle. Adding a
fifth family would change the problem and must be studied separately.

**Status:** implementation committed as a research experiment; CPU
mathematical smoke tests run locally. No GPU experiment, collision-gradient
qualification, 100-path cohort, or numerical gain claim has been completed.

## Debugging evidence, 2026-11-06

`results/debug_20261106` and `results/debug_long_20261106` preserve two
attempts on the same generated Franka/table task. The first used three SLSQP
iterations. In the second, SLSQP reported success after 43 iterations, but the
much denser sampled audit rejected the result (maximum normalized violation
approximately 0.043). Neither is an independently feasible comparison.
`python -m research.joint_time_oracle.generate_debug_report` renders the
corresponding preliminary report without rerunning optimization.

The available PyTorch 2.5.1 environment is missing the
`shared_memory_per_block` device property expected by the current planner.
These debugging runs used a temporary Python compatibility shim returning
49152 bytes for that property; the shim is recorded in the evidence directory.
This changes kernel selection, so these runs must not be used for latency
comparisons. The source base was `adda90b0e2f9f66ca4daba904c180b112ffdca6f`
with the experimental changes in this commit applied; result files predate
subsequent status-label edits. The manifest's GPU is the selected PyTorch
GPU; CUDA device ordering can differ from `nvidia-smi` ordering.

Native collision backward operators require upstream gradient scaling. The
adapter now requests it for both world and self collision, but native CUDA
collision and RNEA gradients still require finite-difference qualification.
The baseline audit checks only stored samples, not a dense or swept path.
The oracle audit uses cuRobo's own models and point samples; it is not
independent verification or a continuous collision certificate. A fixed
95% dynamic limit scaling can conflict with nonzero endpoint derivatives
at physical limits. IPOPT, independent dynamics, adaptive refinement, and
representative multistart experiments remain unqualified. Do not interpret
sampled status labels in historical JSON as independent verification.

## Recovery diagnosis, 2026-11-06

The original final spline's worst normalized residual, 0.042814, was Panda
joint 7 acceleration between collocation points. At 0.09266 s it reached
14.8601 rad/s², exceeding the 95%-scaled 14.25 limit but not the physical
15 rad/s² limit. The original fitted spline did exceed physical acceleration
and jerk limits. The diagnostic records each family, grid, endpoint residual,
physical limit, and location in `results/diagnosis_20261106/diagnosis.json`.

The fitted path uniformly stretched to 0.4314 s passed the sampled check.
A fixed-budget, 16-samples-per-span SLSQP recovery reached 0.401694 s and
passed a 641-point sampled physical-limit check and native swept-world
collision check. Its maximum endpoint acceleration residual was approximately
0.000502 rad/s²; self collision was checked only at samples. This is a local
candidate, not a continuous-time safety certificate. The original baseline
archive contains only 18 samples and cannot independently establish its
behavior between samples. Uniform time scaling is not an equivalent control
for TimeOpt's variable timing law. No IPOPT solve was performed.

Pinocchio 4.1.0 torques match cuRobo within 1.7e-5 N·m on the recovery grid
after matching the models. cuRobo gives four retained, otherwise massless
links default 0.01 kg inertials and excludes the off-chain `right_gripper`;
the raw URDF model differs by up to 0.257 N·m. Both aligned and unaligned
results are retained. Derivative sweeps and all selected VJP entries are in
`derivatives.json`; they are not exhaustive CUDA Jacobian qualification.

To regenerate, use the same CUDA environment and device-property shim described
above, then run the `diagnose`, `qualify_derivatives`, `recover`, and
`export_validation` modules. Run `verify_pinocchio recovery` and
`verify_pinocchio baseline` in a separate Pinocchio environment, followed by
`summarize_recovery`, `plot_diagnosis`, and `generate_debug_report`. These
scripts use the original checked-in artifacts and do not start a benchmark.
A complete fixed-path timing experiment, dense baseline reconstruction,
independent collision check, further derivative testing, and pilot remain
outstanding; do not launch the full benchmark yet.

## Native reconstruction and independent timing control

`results/native_reconstruction_20261106` contains the frozen task rerun,
original action knots and knot timing, selected TimeOpt geometry and squared
speed controls, integrated timing, native replay outputs, and SHA256 checksums.
The rerun's 18 stored states exactly reproduce the historical archive and its
0.412843227 s TimeOpt duration. Replaying the saved buffers reproduces the
8193-point samples exactly. The selected original action knot interval is
approximately 0.02680891 s. The 0.000502 rad/s² reported previously at the
candidate endpoint is float32 matrix cancellation: evaluation of its saved
quintic coefficients yields about 1.4e-12 rad/s². For endpoint checks we use
1e-5 rad, 1e-4 rad/s and 1e-3 rad/s²; the latter covers the observed float32
cancellation and is about 0.0067% of the 15 rad/s² physical limit.

`fixed_path_control` analytically evaluates the native uniform cubic geometry,
uses a cubic squared-speed spline, Pinocchio RNEA coefficients, 311 sampled
constraints and SciPy SLSQP. The 800-iteration solve stopped at its budget;
its candidate was slowed by multiplying squared speed by 0.997 and then
sampled at 16439 positions, including points on both sides of geometric knots.
The restored duration is 0.402526355 s. It is a sampled feasible timing
candidate, not a certified or converged optimum. The analytic path matches
native sampling within 4.2e-7 rad on the sampled interior. No timing control
has yet been implemented for the original quintic joint candidate.

On sampled physical limits, native TimeOpt, restored fixed-path timing and
the joint candidate all passed; sampled self-collision and swept-world costs
were zero. Their durations are approximately 0.412843, 0.402526 and 0.401694 s.
Native TimeOpt and the joint candidate nevertheless exceed the stricter 95%
jerk and acceleration/jerk limits by the amounts in `dense_validation.json`.
Model-aligned Pinocchio RNEA agreed with native torque to about 2e-5 N·m.
Collision verification still uses cuRobo and self-collision remains sampled.

`cuda_derivative_sweep.json` covers RNEA full joint Jacobians at two path
states and one random state, plus free, near-contact and penetrating collision
maxima. `oracle_jacobian_sweep.json` contains the full SLSQP constraint
Jacobian at the fitted seed and original final trajectory. Large differences
at switching minima are not smooth derivatives. The initially unflagged
0.003-step discrepancy was reproduced and traced to an active sample changing
from 8 to 7 (`active_sample_diagnosis.json`); other flagged changes were not
all individually examined. This evidence is not complete qualification of all
input shapes or contact regimes.

Reproduce with the same CUDA environment and compatibility shim documented
above, running `reconstruct_baseline`, `replay_native`, `fixed_path_control`
(in a Pinocchio/SciPy environment), `sample_fixed_control`, `validate_dense`,
`validate_pin_dense` (Pinocchio), `validate_self_midpoints`, `export_ee`,
`qualify_cuda_full`, `qualify_oracle_jacobian`, `investigate_switch`,
`summarize_comparison`, `plot_comparison`, and `generate_debug_report` in that
order. The frozen comparison remains **blocked** for the pilot pending a
qualified treatment of nonsmooth derivatives and collision verification,
resolution of the scaled-limit violations, and a stronger fixed-path solve.
No full benchmark was run.

### Pointwise fixed-path follow-up

The single-task follow-up is generated from the frozen `native_reconstruction_20261106`
artifacts. An isolated environment with Python, NumPy, SciPy, Pinocchio, PyYAML and
`cyipopt`/IPOPT is required; no production defaults are changed. The 2026-11-06 run
used conda-forge IPOPT 3.14.19 and cyipopt 1.7.0. The historical 311-point
`fixed_path_attempt.npz` supplies the path samples and aligned torque coefficients.

```bash
OPENBLAS_NUM_THREADS=1 python -m research.joint_time_oracle.qualify_pointwise
OPENBLAS_NUM_THREADS=1 python -m research.joint_time_oracle.validate_collision_independent
OPENBLAS_NUM_THREADS=1 python -m research.joint_time_oracle.verify_pointwise_torque
python -m research.joint_time_oracle.summarize_timing_research
python -m research.joint_time_oracle.generate_debug_report
```

`fixed_path_pointwise` reconstructs Pinocchio coefficients for `--grid-count`; its
`--refine` option reads `refinement_input_2049.npz`, the dense output of the saved
2049-grid solve. The existing attempt names preserve the bounds and options in effect
at the time: early solves used a scaled upper bound of 100; current runs use 10000,
which is also active. A large bound is not a proof of an unrestricted optimum. The
native-iteration sweep requires the same cuRobo GPU environment and compatibility
shim documented above. Native 40/80 outputs are solver diagnostics, not independently
qualified native trajectories. The new checksum manifest is
`results/native_reconstruction_20261106/timing_research_checksums.json`; the original
manifest remains unchanged.


## Frozen multi-problem protocol (not executed)

See [COHORT_PROTOCOL.md](COHORT_PROTOCOL.md) for the 20-task pilot and
100-task source-disjoint confirmation protocol, method definitions A10/A40/B/C/D,
independent physical and Scaled95 acceptance gates, and the task-level result
schema. [cohort_statistics.py](cohort_statistics.py) produces fail-closed paired
statistics and an offline HTML summary from frozen JSONL inputs; it does not
run optimization. The historical single-task artifacts are not part of the new
holdout. CPU-only aggregation tests are in
`tests/test_joint_time_cohort_statistics.py`. No cohort improvements are
claimed until real GPU runs and independent validation complete.

## Exploratory cohort execution (blocked)

`results/cohort_pilot_exploratory_20261106` records 20 frozen generated pilot
problems (six free, eight `collision_test.yml`, six `collision_primitives_3d.yml`)
and all 23 endpoint proposals. The first run exposed an error from changing the
planner-owned retimer's sampling shape; `cohort_pilot_retry_20261106` records a
second failed attempt that reset a sampler inside the planner's captured graph.
`cohort_pilot_isolated_20261106` preserves a third run using an independent
replay instance. The task definitions are identical across these runs. Each
successful native attempt stores the selected lane, input knots, timing buffers,
8193 native physical-time samples, and A40 output from the same input.

`cohort_validate` evaluates the saved samples with aligned Pinocchio dynamics,
robot spheres, the configured table/cuboids/spheres/capsules, and independent
sphere-pair distances with configured ignore masks and padding. These are sampled
checks. They do not establish swept or continuous collision safety, cover exact
adaptive peaks, or qualify a method for the cohort protocol. None of the
trajectory `verification` flags is set to true. The exploratory scripts were
modified in the working tree around the run; the manifest identifies this
provenance limitation. The checked-in generated tasks are not a representative
benchmark, and source-disjoint confirmation tasks have not been generated.

`cohort_geometry` exposes exact native cubic and oracle quintic path evaluation,
including an explicit leading/trailing stationary-span trimming operation. It is
not yet wired into the IPOPT retimer. In particular, the artificial speed-control
bound remains unresolved, the matched B/D executor is missing, and C multistart,
CUDA derivative classification, and adaptive verification are incomplete.
Consequently, no pilot comparison or confirmation launch is qualified.

With the environments described above, reproduce the independent sampled audit
and regenerate the offline report and fail-closed statistics with:

```bash
OPENBLAS_NUM_THREADS=1 python -m research.joint_time_oracle.cohort_validate \
  --input research/joint_time_oracle/results/cohort_pilot_isolated_20261106
python -m research.joint_time_oracle.cohort_report
python -m research.joint_time_oracle.generate_debug_report
```

### Exact-geometry timing pilot

`cohort_retime` uses the exact captured native cubic or joint-oracle quintic
geometry. Native geometry trims only leading and trailing cubic spans whose four
support controls are bitwise identical; no magnitude threshold is used. It
optimizes a positive 32-control squared-speed spline with pointwise dynamic
inequalities, aligned Pinocchio inverse dynamics, no finite upper speed-control
bound, and up to two bounded adaptive collocation refinements. A final uniform
slowdown is saved separately if the dense scaled-limit audit requires it. The
positive lower coefficient bound is numerical and its activity is reported.

`cohort_joint` restores fitted baseline seeds and tries two spline resolutions
and two starts per resolution. It retains the best native-sampled feasible seed
or restored solver iterate and labels its origin. CUDA gradients and collision
sampling are not independently qualified, so every candidate still requires the
separate independent verifier. Solver status alone does not establish feasibility.

The final runner `cohort_execute` records its source commit, frozen task checksum,
iteration limits and subprocess failures before executing. The prior three-task
experiments are exploratory and may have used different refinement settings.

## Completed frozen 20-task exploratory pilot (2026-11-07)

`results/cohort_joint_final_20261106/` contains the end-to-end A10/A40/B/C/D
rerun for all 20 tasks frozen in `cohort_pilot_isolated_20261106/tasks.json`.
The task checksum and pre-rerun implementation commit are in
`frozen_config.json`. B and D use the same pointwise IPOPT speed-law NLP and
exact captured cubic/quintic geometry, respectively; no finite arbitrary upper
bound is imposed on squared-speed controls. This remains a small generated
exploratory cohort, not the 100-task confirmation study.

Run `cohort_analyze` to rebuild physical and Scaled95 records from stored
trajectories and independent validation JSON. Run `cohort_joint_report` to
regenerate `reports/joint_time_oracle/cohort/index.html` and per-task plots.
`cohort_restore` writes separate time-stretched Scaled95 trajectories and
`cohort_validate` independently evaluates Pinocchio torque and analytical
sphere/scene clearance. The adaptive collision pass samples neighborhoods
around identified clearance minima; it does not certify continuous-time
collision avoidance. Dynamic adaptive samples include spline knots and
identified constraint extrema. The report explicitly distinguishes these
sampled checks from continuous-time certification.

The frozen pointwise solves use 257 base progress samples, up to two adaptive
passes, 32 cubic squared-speed controls, IPOPT tolerance 1e-7 and a 150-iteration
cap for B/D. C used 10 and 14 quintic controls, two starts per resolution, and a
40-iteration cap. A separate 513-sample check on one free, one table, and one
cluttered case is saved under `mesh_checks/`; it is a convergence sensitivity
check, not a post-hoc retuning of the 20-task cohort. `checksums.json` lists
SHA256 for stored evidence. Local solvers provide no global optimality
certificate, and CUDA collision-gradient/continuous swept-collision
qualification remains incomplete. Do not start the confirmation cohort from
this exploratory result without resolving those gates.


## Torque-active stress extensions (2026-11-07)

See [TORQUE_STRESS_PROTOCOL.md](TORQUE_STRESS_PROTOCOL.md). The completed
frozen exploratory study has separate reports for the historical 20-task
cohort and the generated 30-task wider-excursion cohort:

- `reports/joint_time_oracle/torque_active/index.html`
- `reports/joint_time_oracle/torque_active_wide/index.html`

Raw artifacts are under `results/torque_stress_final_20261107/` and
`results/torque_wide_stress_20261107/`; source-disjoint wide task definitions
are under `results/torque_wide_source_20261107/`. `.npz` trajectory archives
are kept outside Git because the raw cohorts exceed 3 GB. The report artifact
manifests list exact paths, sizes, and SHA-256 hashes.

On the original 20 tasks, severe derating produced a median +5.46% B→D gain
(paired bootstrap 95% CI +1.79% to +7.39%); moderate derating produced no
demonstrated geometry gain (median −0.39%). The wider cohort reproduced a
smaller positive severe-regime gain (+3.47%, 29 pairs; CI +2.79% to +3.83%)
and a small moderate gain (+0.44%, 27 pairs; CI +0.33% to +0.53%). The original
20-task severe gain was larger in table and cluttered populations than in free
space. In these improved groups D's median joint path was longer than B's, so
path shortening alone does not explain the duration reduction.

Claims are limited to independently sampled physical checks; this is not a
continuous-time certificate. C is a local multistart search: individual SLSQP
starts terminated successfully in 77/480 trials on the original cohort and
218/684 trials on the wider cohort. Feasible nonconverged incumbents remain
distinct from converged local solutions. A full geometric planner rerun under
the derated effort constraints and exhaustive homotopy exploration were not
completed. Derating simulates a lower-capacity actuator, **not** an actual
payload; Panda inertials and production limits were not changed.

## Torque physics and full native replanning follow-up (2026-11-07)

See [TORQUE_MECHANISM_FULLPLANNER_PROTOCOL.md](TORQUE_MECHANISM_FULLPLANNER_PROTOCOL.md)
for the staged torque decomposition and full-planner rerun protocol.
This experimental branch starts from frozen torque-stress commit
bcbb85a709f06b2a934f8ba141137c33379c0450 and preserves the original
A10/A40/B/C/D evidence. It adds:
- independent Pinocchio RNEA gravity / inertial / velocity-bias breakdown;
- full native cuRobo C-space planning with derated effort limits installed
  **before** constructing the planner;
- 1/4/16/32-restart budgeted selection across all internally valid native lanes;
- dense and adaptive independent Pinocchio/sphere audits and SHA-linked receipts;
- a deterministic N/A-aware HTML report at
  reports/joint_time_oracle/torque_mechanisms/index.html.

**Unexecuted:** no new GPU experiments or new independent verification
outcomes are claimed by this implementation. Run the documented tests and
GPU qualification chain before interpreting full-planner comparisons.

The principal new comparison is **Full-R versus D**, where Full-R independently
retimes every internally valid native full-planner geometric candidate using
the *same IPOPT fixed-path solver* as B and D. Native Full versus D is retained
as a separate practical, mixed-timing-solver comparison. Run
torque_full_retime.py and torque_full_retime_adaptive.py before issuing strict
native-full receipts and selecting the best validated matched trajectory.
