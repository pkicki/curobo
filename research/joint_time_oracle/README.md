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
