# Prompt for the coding agent (experiments, tests, fixes, report)

You are working in `/home/pkicki/curobo` on branch `exp/timeopt-guided-planner-20261110`
(GPU workstation: A100 = cuda:0, RTX 3090 = cuda:1). Read `AGENTS.md`,
`research/timeopt_guided/PLAN.md` and `research/timeopt_guided/README.md` first.
The goal is to show whether fast retiming FEEDBACK (K107 TOPP-RA+Newton) used to
modify path geometry beats cuRoboV2-Fast and an identical blind search, under
the ORIGINAL Franka actuator limits with rigid payloads. Do not open a PR, merge
to `main`, or change cuRobo production defaults or kernels. Commit to this
branch, and push only this branch.

## Ground rules

* Two interpreters, never mixed. CUDA:
  `/home/pkicki/anaconda3/envs/curobo-timeopt/bin/python` with
  `PYTHONPATH=$TOC/src:$PWD`. Pinocchio: `/tmp/oracle-ipopt-env/bin/python`
  with `PYTHONPATH=$PWD`.
* `$TOC` is a worktree of `time_optimal_curobo` at
  `94d25ecde95bc7f31851b0b9a78f61bd5f7cb65e`. Create it as shown in the README.
  Record both repo SHAs, the GPU and the driver in `provenance.json` in every
  result directory.
* No result counts unless `independent_validate.py` passes it (Scaled95). K107
  results alone are preliminary. A failed or numerically erroring retimer is
  never proof of infeasibility.
* Tune ONLY on the development split. Run `freeze.py` before touching the
  confirmation split. Report every negative or unresolved outcome. Never select
  cases because an alternative happened to succeed.
* Keep raw NPZ local under `research/timeopt_guided/results/` (git-ignored).
  Commit code, tests, small JSON summaries and HTML.
* Fix bugs you find in the harness (with a test), and note each fix in the
  final report. If a fix would change cuRobo library code, put it behind a
  research-only wrapper and describe it instead.

## Steps

0. **Setup and tests.** Create `$TOC`. Run `tests/test_timeopt_guided.py` in
   both interpreters (expected: 7 passed, 1 skipped in each). Run one warm
   K107 smoke and one native-planner smoke (the README commands with `--limit`).

1. **Hard cases under original limits** (PLAN P2). Write everything to
   `OUT=research/timeopt_guided/results/campaign_20261110`.
   * `task_bank` with default seed and sizes (180 tasks, 1980 cells).
   * `mine_native` R1 on all admissible cells.
   * Escalate failures to R4, then R16.
   * Sensitivity check: rerun R1 on the failure cells and a 10% success sample
     with upstream `main` (`5e4f23d`), in a separate worktree, under the same
     interpreter.
   * Report rates by payload, COM, scene and stratum. Report separately:
     endpoint-inadmissible cells, `timeopt_seed_infeasible_all_lanes`, other
     failure stages, and the native false successes
     (`success_flag_with_nonfinite_trajectory`).

2. **Native TimeOpt issue** (PLAN P1). Run `seed_diagnosis` for every budget.
   * Report lane classes and the agreement between the replay and CUDA
     `seed_feasible`. The pilot measured 96.7%, below the 99% bar: find the
     cause (grid points, stationary spans, float32, captured lanes vs reported
     lanes, the boolean dtype of the stored flag) and fix or explain it.
   * Investigate `TOPPRA_FEASIBLE_NEWTON_START_FAILED` (34% of pilot lanes):
     - extract the raw TOPP-RA profile;
     - validate it independently on a dense grid;
     - retry with stride 1 and a larger `fit_k`;
     - decide whether it is a genuine law through a gravity overload or a
       grid artefact.
   * Root-cause the native false successes: success flag set with a NaN
     trajectory, 15 of 1239 pilot successes, all at 4–5 kg. Produce a minimal
     reproducer (task, payload, seed) and identify the first non-finite buffer
     and where the success mask is set.
   * Optional, opt-in: a non-uniform native seed prototype in a wrapper,
     reporting how many native failures it fixes on the SAME geometry.

3. **Guided search on the development split** (PLAN P3). First fix the
   screen/audit mismatch from the pilot (PLAN 2.3: 16 of 167 search exports
   failed the independent world-collision check). Options: denser
   `PathCollisionChecker` sampling, a margin that covers the sphere-model
   difference, or a final dense cuRobo check before export. Show that rejected
   exports drop to about 0 without hiding real collisions.
   * Run `run_campaign` on the development split with `--search-seeds 3
     --max-proposals 128`, once without and once with `--native-seeds`. Use
     separate output directories.
   * Then run `independent_validate` and `report`.
   * Tune at most 3 configurations on development: `global_fraction`,
     `directed_fraction`, amplitudes and elites. Use `--config` JSON and log
     all of them. Pick one and run `freeze.py`.
   * Report the following per arm:
     - rescue@{0.1, 0.3, 1, 3, ∞} s;
     - proposals to first feasible;
     - validated duration ratio vs native;
     - the retimer-only ratio (K107 on the exact native geometry);
     - oracle calls and wall time;
     - paired Wilcoxon tests (D vs B, C vs B, D vs C) by source task.

4. **Practical planner and latency** (PLAN P4).
   * Implement hybrid planner E: native attempt 1; if all TimeOpt seeds are
     infeasible, stop native and run the frozen guided search under a
     wall-time budget.
   * Add an oracle mode with TOPP-RA-only scoring and Newton only for elites
     and the final candidate. Report the rank correlation against full K107.
   * Compare A vs E on total request latency (GPU-synchronized, warm, the same
     GPU, compile time excluded and reported separately), success, and
     validated duration.

5. **Confirmation split** (PLAN P5). Run with the frozen configuration and
   `--freeze`, with no changes. Apply the GO / PARTIAL / NO_GO rule from
   `report.py` and PLAN P3 to these numbers.

6. **Report.**
   * Write `reports/joint_time_oracle/timeopt_guided_planner/index.html` (from
     `report.py` output plus prose) and add a "Results" section to `PLAN.md`.
   * Include:
     - denominators at every stage;
     - the class tables;
     - budget curves;
     - paired statistics with confidence intervals;
     - the native-bug findings;
     - every fix you made;
     - the final decision with its justification;
     - limitations (sampled validation, payload model, 4–5 kg above rating,
       torque rate not enforced).
   * Commit and push the branch. Reply with: the decision, the 10 most
     important numbers, the bugs found and fixed, and anything that blocked you.
