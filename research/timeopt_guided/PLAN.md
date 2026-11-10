# TimeOpt-guided geometric planning: implementation and experiment plan (2026-11-10)

Research branch `exp/timeopt-guided-planner-20261110` (cuRobo). No PR, no merge to
`main`, no change to production defaults.

## 1. Goal

Build a planner extension that **uses fast, dynamics-aware retiming to choose and
modify path geometry**, and show that it beats cuRoboV2-Fast:

* **rescue**: native fails under its budget, the extension finds an
  independently validated trajectory under the same task, payload and limits;
* **quality**: native succeeds, the extension finds a validated trajectory that is
  >= 5% shorter;
* **attribution**: retiming feedback must beat an identical search that does not
  use it (blind proposals, same budget). Otherwise the contribution is "global
  search + a better retimer", not "TimeOpt-guided geometry".

All experiments use the ORIGINAL Franka actuator limits. Hard cases come from
physical rigid payloads (0–5 kg, two COM offsets), not synthetic derating.

## 2. Where we start from

### 2.1 Inherited work (ChatGPT session, 2026-11-09/10)

Four parallel, never-executed curobo branches (`exp/timeopt-feedback-nominal-payload-20261109`,
`exp/timeopt-guided-nominal-payload-20261109`, `exp/timeopt-guided-path-payload-hardcases-20261110`,
`exp/timeopt-guided-payload-hardcases-20261110`) and two time_optimal_curobo
branches. They are left untouched. This branch is a fresh, executed consolidation
from `1597f12`, for these reasons:

* the curobo branches' tests fail at import (they import Pinocchio from the CUDA
  environment, where it is not installed); none of their pipelines has been run;
* the K107 bridge (`time_optimal_curobo:exp/joint-time-oracle-k107-bridge-20261109`)
  runs, but its payload COM/inertia mutation **has no effect** (see 2.2). Only
  the mass, set before Dynamics construction, was applied;
* the protocol content worth keeping is kept: matched four-family constraint
  contract, exact native path for the TimeOpt diagnosis, nominal effort plus
  payload, broad and enriched strata, dev/confirmation split by source task,
  arms A–D, independent validation, and the GO/PARTIAL/NO_GO rule.

### 2.2 Verified in this session (code in this directory)

1. **Native TimeOpt failure mechanism.** `timeopt_io_kernel.cuh`
   (`fused_coefficients_and_seed_bounds`, `fused_seed`) seeds with ONE constant
   squared path speed x. Where gravity alone exceeds 0.95·τmax, the torque row
   imposes a positive lower bound on x. Elsewhere, V/A/J/τ cap x from above. The
   interval empties, and all lanes are rejected before Newton runs.
   `seed_diagnosis.uniform_seed_interval` reproduces the CUDA flag (100%
   agreement on the smoke set). This is an initialization restriction, not proof
   that the path is infeasible.
2. **Payload plumbing.** `MotionPlanner.update_link_inertial()` raises for the
   default Franka planner ("Cannot update link inertial properties without
   inverse dynamics"), because some IK/TrajOpt rollouts have no dynamics.
   Mutating `link_masses_com` and rebuilding `Dynamics` is ignored, because the
   reduced model is cached in `KinematicsParams._dynamics_model`. What works:
   `Dynamics.update_link_inertial` on a Dynamics bound to the planner's
   KinematicsParams. It refreshes the shared reduced model in place, so it is
   CUDA-graph safe and native TimeOpt sees it. Verified: native TimeOpt vs K107
   gravity parity is 0.0 Nm, and cuRobo vs independent Pinocchio parity is
   < 1e-4 Nm.
3. **K107 TOPP-RA+Newton runs in-process** (pinned time_optimal_curobo
   `94d25ec`). Warm cost per path: about 100 ms when feasible (Newton about 80 ms)
   and about 20 ms when infeasible, including an 8-step torque-expansion bisection.
   It accepts quintic search paths and the exact native cubic. Native lanes are
   converted to a clamped cubic by knot insertion, with an error of about 1e-15.
4. **Smoke numbers** (30 tasks, 231 admissible cells, R1). 24 native failures,
   all `timeopt_seed_infeasible_all_lanes`, at 3–5 kg, mostly in the enriched
   stratum. On exact native lanes: many `GEOMETRY_TORQUE_INFEASIBLE` (TOPP-RA
   finds no law), some `TOPPRA_FEASIBLE_NEWTON_START_FAILED`. Search on 8
   failures: 7 rescued by every arm within the shared first round, so they are
   initialization-hard, not geometry-hard. The remaining one is hard for all
   arms, but C/D push torque expansion to 1.003 vs 1.196 for B.
5. **Native false successes.** In the pilot (180 tasks, 1980 cells, 1369
   admissible, R1), 15 of 1239 native-reported successes (about 1.2%, all
   4–5 kg) return `success=True` with an all-NaN interpolated trajectory. The
   harness now counts them as failures (`success_flag_with_nonfinite_trajectory`)
   and captures their lanes. Pilot R1: 130 seed-infeasible failures + 15 false
   successes = 145 native failures (10.6% of admissible cells).
6. **Latency gap.** A warm native plan takes 15–25 ms on success and 0.4–1.1 s on
   a seed-infeasible failure (20 internal attempts). A 128-proposal search takes
   2–9 s. A practical planner therefore needs Phase 4.

### 2.3 Pilot (2026-11-10, development split, 1 search seed: PRELIMINARY)

180 tasks, 1980 cells, 1369 admissible. Native R1: 145 failures (130
seed-infeasible + 15 NaN false successes). Escalating to R4 rescued only 1 of
130, so more native restarts barely help. Seed diagnosis on 520 exact native
lanes:

* 62% are `GEOMETRY_TORQUE_INFEASIBLE`;
* 34% are `TOPPRA_FEASIBLE_NEWTON_START_FAILED`;
* 3.5% had a native seed flagged feasible;
* none were fully retimed by K107 on the native geometry;
* replay vs CUDA flag agreement is 96.7%, which is below the bar and needs
  investigation.

Search on 40 development cells (28 native failures, 12 success controls),
128 proposals:

| | B blind | C ranked | D guided |
|---|---|---|---|
| validated rescue, task-averaged, ∞ budget | 0.58 | 0.72 | **0.78** |
| rescue within 1 s / 3 s | 0.58 / 0.58 | 0.72 / 0.72 | 0.72 / 0.78 |
| median duration vs native (successes) | 0.988 | 0.988 | 0.988 |
| median search wall time | 4.6 s | 8.1 s | 8.5 s |

* 23 confirmed hard failures (validated rescue) and 6 unresolved.
* On native successes, the arms do not differ (D vs B paired p = 0.55).
* The report rule says GO_RESEARCH, but n is small and there is one seed:
  treat it as a signal only.
* Validation issues to fix: 16 of 167 search exports fail the independent
  world-collision check. The search screen uses 257 samples and a 5 mm cuRobo
  activation distance; the audit uses 2049 samples. One native "success"
  collides in the independent audit, and one marginally exceeds Scaled95.
* The retimer-only decomposition is missing from this pilot: the native runs
  predate selected-lane capture.

### 2.4 Are the TOPP-RA laws behind `TOPPRA_FEASIBLE_NEWTON_START_FAILED` real? (P1 item 3, done)

Tool: `toppra_audit.py`. Inputs: 178 lanes in 46 pilot cells, which reduce to
59 unique geometries. Each was re-solved on the oracle grid (2049, stride 3),
stride 1, and a fine grid (8193, stride 1), audited 8× between nodes with
cuRobo RNEA, and checked independently with Pinocchio.

* **56/59 geometries (45/46 cells) have a REAL second-order law.**
  * Feasible on all grids; duration converges (median change 0.4%, max 3.4%).
  * Between nodes: torque/velocity/acceleration utilization ≤ 0.995 of Scaled95.
  * Pinocchio agrees: 56/56 torque, velocity and acceleration ≤ Scaled95.
* **3/59 are grid artefacts:** TOPP-RA becomes infeasible after refinement.
* **The laws "swing through" a static gravity overload.** Median overload is
  5.7% (max 34.9%), lasting a median 0.30 s (max 1.24 s). Minimum path speed
  inside the overload is median 0.34 (min 0.11).
* **Jerk is not the blocker.**
  * The reconstructed law reaches 2.49× the jerk limit, but only at the final
    rest instant. That is a generic artefact of piecewise-linear x near rest.
  * Excluding the first and last 5% of time: median 0.42×, max 1.02×. Inside
    the overload region: max 0.43×.
* **Root cause of the K107 failure:** its Newton start is restored to the
  strict interior by scaling the law down (x → λx). Under gravity overload,
  slowing down *raises* torque, because the dynamic term was compensating
  gravity. Fraction of laws still torque-feasible after slowing:
  * 1.00 at λ = 0.99, 0.68 at 0.95, 0.54 at 0.9, 0.16 at 0.8, 0.02 at 0.5;
  * a larger `fit_k` (29/55/107) and stride 1 do not help (0/56).
  * This is the SAME "slower is always safer" assumption as native TimeOpt's
    constant-speed seed (see 2.2). Both initializations are incomplete for
    paths that cross a static overload.
* **Practical reading.**
  * Native TimeOpt (and K107) failed on paths that DO admit a timing.
  * Those timings have zero torque slack through the overload. A 5% slowdown
    (e.g. tracking lag) makes 32% of them torque-infeasible.
  * They are physically real but fragile. Geometry that avoids the overload
    (the guided search) is the robust fix.
  * Retiming candidates: an overload-aware start for K107 (restore by moving
    toward the TOPP-RA law, not toward x = 0); for native, the opt-in
    non-uniform seed of P1 item 4.
* Two of the 56 native geometries also collide in the independent world audit
  (consistent with 2.3).

### 2.5 Overload-aware K107 start: Phase-I LP in timing-spline space (done, opt-in)

`K107Oracle(..., overload_aware_start=True, start_method="lp")`. It runs only
after the standard start fails, so regular paths are unchanged.

* **Idea.** The TOPP-RA feasible set is convex in x(s). K107's velocity²,
  acceleration and torque rows are linear in its 107 timing coefficients; jerk
  is linearized at the TOPP-RA law. One HiGHS LP on every 4th grid point
  maximizes speed subject to ≥ 0.3% relative slack on all rows. That gives a
  strictly interior Newton start without slowing the overload stretch down.
  The few tiny between-point residuals are left to K107's own restoration,
  which then slows down by ≪ 1%.
* **Pilot result** (59 unique native geometries from 46 failed cells):
  * 54/56 real-law geometries are now timed by the full K107 route, Newton
    plus C3 screen. The 3 grid artefacts correctly still fail.
  * 43/46 cells now have a K107-timed native path.
  * Independent Pinocchio: 54/54 dynamically valid (V/A/J/τ max 0.9985 of
    Scaled95); 2 fail only on world collision (the known colliding native
    paths).
  * Duration is a median 0.9% (max 1.6%) above the second-order TOPP-RA bound.
* **Cost:** median 253 ms per call (LP ≈ 160 ms), vs ≈ 100 ms for a regular
  K107 call. Rejected alternatives:
  * tightened-margin TOPP-RA alone: 0/56 (spline-fit overshoot at bang-bang
    kinks exceeds the slack);
  * exact cutting planes: degenerate, slow;
  * two-stage max-slack LP: same success, 1.8× slower.
* **Caveat.** These laws keep only about 0.3% torque slack through a static
  overload. Execution robustness (tracking lag, model error) still favours
  geometry that avoids the overload. The fallback makes the retimer complete
  on such paths; it does not make them good paths.
* **Next** (agent): enable it as a search-oracle option and compare rescue and
  robustness. Port the idea to native TimeOpt as an opt-in non-uniform seed
  (P1 item 4). Speed up the LP: warm start, fewer rows outside overload
  windows, or a GPU first-order LP.

### 2.6 Idea A: gravity-aware TrajOpt and the hybrid planner (done, development pilot)

* **Finding.** `franka.yml` loads no dynamics (`RobotCfg.load_dynamics=False`), so
  TrajOpt and the success metrics never evaluate torque; only TimeOpt does.
  Geometry is chosen torque-blind.
* **Variants** (`native_runner.VARIANTS`, research only):
  * `torque_aware`: load dynamics, so TrajOpt's existing torque-bound cost
    becomes active.
  * `gravity_aware_<ρ>`: the optimizer rollouts see gravity torque / ρ instead
    (velocity and acceleration zeroed), which keeps |g(q)| ≤ ρ·τmax along the
    path. Metrics rollouts are unchanged.
  * `hybrid`: native `max_attempts=1`. On `timeopt_seed_infeasible_all_lanes`
    (or a NaN false success) it switches to `gravity_aware_85`; on any other
    failure, native continues with default attempts.
* **Frozen comparison set** (pilot development tasks): 145 native R1 failures,
  100 heavy (≥ 3 kg) successes and 60 light successes, all run on the same
  A100.

  | planner | R1 failures, physically valid | heavy successes valid | light valid | median latency |
  |---|---|---|---|---|
  | native | 2/145 | 100/100 | 60/60 | 16–18 ms (fail: 388 ms) |
  | torque_aware | 64/145 planned | 84/100 planned | 60/60 | — (different GPU) |
  | gravity ρ = 0.85 | 116/145 | 85/100 | 59/60 | 22 ms |
  | **hybrid** | **117/145** | **100/100** | **60/60** | 16–19 ms; rescue 42 ms |

  * ρ sweep (planned): 0.95 → 76, 0.90 → 114, 0.85 → 123 of 145.
  * Durations on cells both planners solve: hybrid identical; gravity-only
    median +0.1–0.6%.
  * The gravity-only variant loses about 15% of heavy successes to collision:
    the gravity cost pulls the payload toward the arm. The hybrid avoids this
    by construction.
  * 5 of 122 hybrid rescues fail the independent world-collision audit (the
    same sphere-model gap as native).
  * Jerk: all planners, native included, exceed Scaled95 jerk by up to 0.5%
    (cubic jerk sampling), within the physical limit.
* **Harness fixes:** the eager TimeOpt replay can return NaN where native's
  interpolated output is finite; it now falls back to the interpolated output.
* **Not yet done:** held-out confirmation split, full-cohort rates (the
  frozen set is enriched with failures), and a ρ / cost-weight sweep for the
  standalone variant.

### 2.7 Ideas B and C: better retiming and retimer-guided path refinement (done, development pilot)

All numbers below are independently validated (Pinocchio; Scaled95 with 0.5%
jerk sampling tolerance, collisions, endpoints), on the frozen idea-A cell set,
on top of the hybrid planner.

* **B, `lane_retime.py`.** K107 retimes all 4 candidate lanes of the
  successful solve (exact cubic) and keeps the fastest collision-valid one.

  | cells | median shorter | mean shorter | ≥ 5% shorter |
  |---|---|---|---|
  | rescued (gravity fallback) | 3.7% | 7.9% | 41% |
  | heavy successes | 2.2% | 4.4% | 20% |
  | light successes | 2.0% | 2.6% | 7% |

  * Most of the gain is the retimer: same lane, K107 vs native TimeOpt is
    2.1–3.5% shorter.
  * Cost: 0.4 s sequential (4 lanes), which needs batching.
  * 15 cells had no K107-timeable lane.
* **C, `joint_refine.py`.** Starts from B's best lane, then alternates: K107
  law → request it σ = 1.04 faster → Adam on the free cubic controls against
  the V/A/J/τ excess at the faster law (differentiable cuRobo RNEA) plus
  differentiable collision and joint limits → accept only if the collision
  screen passes and K107 is feasible and faster. 20 cells per role, 8
  iterations.

  | cells | C vs B (median, p10) | C vs planner (median, mean, p10) | ≥ 5% shorter vs planner |
  |---|---|---|---|
  | rescued | 0.958, 0.693 | 0.934, 0.850, 0.580 | 58% |
  | heavy successes | 0.999, 0.910 | 0.972, 0.940, 0.848 | 32% |
  | light successes | 1.000, 0.995 | 0.979, 0.978, 0.975 | 0% |

  * 54/54 refined paths pass validation.
  * Cost: 1.4–1.9 s and 6–9 K107 calls per request; 16 iterations gain ≤ 1%
    more.
  * Reading: geometry refinement pays off where torque is active (heavy
    payload, and especially the gravity-aware rescues, whose paths are
    conservative). With light payloads the retimer alone gives the about-2%
    gain.
* **Next:**
  * Latency: batched K107 over lanes; TOPP-RA-only inner scoring for C;
    fewer inner steps.
  * Compare C vs the population search (arms B/C/D of `guided_search`) at
    matched wall time.
  * Rerun A+B+C on the held-out confirmation split.

## 3. Software (all in `research/timeopt_guided/`)

| Module | Env | Role |
|---|---|---|
| `common.py` | any | payload contract/grid, scenes, split, hashing, quintic helpers |
| `geometry.py` | any | exact native-cubic→clamped conversion, quintic eval/fit, proposal operators |
| `k107_oracle.py` | CUDA | in-process K107 oracle with structured feedback (duration, utilizations, bottleneck profile, torque expansion) and dense sampling |
| `native_runner.py` | CUDA | one warmed `MotionPlanner` per scene; payload install; per-solve TimeOpt lane capture; exact dense replay |
| `collision_screen.py` | CUDA | batched cuRobo world/self/limit screen with the payload sphere |
| `task_bank.py` | CUDA | 180 tasks (broad 40 + enriched 20 per scene), all rejections logged, endpoint admissibility per payload |
| `mine_native.py` | CUDA | native R1 → R4 → R16 ladder; failed cells frozen before each escalation |
| `seed_diagnosis.py` | CUDA | uniform-seed replay + K107 on the exact native lanes; lane classes |
| `guided_search.py` | CUDA | arms B/C/D with identical first round, blind stream and budget |
| `run_campaign.py` | CUDA | frozen selection, matched arms × seeds, exports, payload parity samples, retimer-only decomposition |
| `independent_validate.py` | Pinocchio | RNEA + payload, V/A/J/τ physical and Scaled95, sphere world/self collisions, endpoints/rest, parity |
| `report.py` | any | cell classes, arm metrics, paired tests, budget curves, decision; JSON + HTML |
| `tests/test_timeopt_guided.py` | both | pure, CUDA-gated and Pinocchio-gated tests |

Environments: `CUDA_PY=/home/pkicki/anaconda3/envs/curobo-timeopt/bin/python`
with `PYTHONPATH=<toc>/src:<curobo root>`, and `PIN_PY=/tmp/oracle-ipopt-env/bin/python`,
where `<toc>` is a worktree of time_optimal_curobo at
`94d25ecde95bc7f31851b0b9a78f61bd5f7cb65e`. Never mix them.

## 4. Phases, experiments and acceptance criteria

### P0: environment, pinning and tests

* Create the pinned K107 worktree and run the tests in both environments.
* Record `git rev-parse HEAD` of both repos, GPU model and driver in every result
  directory (`provenance.json`).
* **Accept:** 7 pass / 1 skip in each environment (the skip is the other
  environment's test).

### P1: native TimeOpt issue (user item 1)

1. Run `seed_diagnosis` on every native failure at every budget.
2. Report lane classes per payload/stratum, and the agreement between the
   uniform-seed replay and CUDA `seed_feasible`. **Accept:** >= 99% agreement,
   otherwise investigate the grid mismatch.
3. Investigate `TOPPRA_FEASIBLE_NEWTON_START_FAILED`. Extract the raw TOPP-RA
   profile, check it against an independent dense validation, and determine
   whether it is a real "swing-through-overload" law or a stride-3 grid
   artefact. Retry with a finer TOPP-RA stride and a larger `fit_k`.
4. Optional, opt-in only: prototype a non-uniform native seed (initialize native
   TimeOpt from a TOPP-RA profile, or from a piecewise-uniform x that is slowed
   only where the path is gravity-feasible), behind a research flag in a
   wrapper. Report how many native failures become native successes on the SAME
   geometry. Never change default kernels.
5. Root-cause the native false successes (`success_flag_with_nonfinite_trajectory`).
   Rerun those cells with `use_cuda_graph=False`, and locate the first non-finite
   buffer (TimeOpt `_times`, `_timing`, interpolation `dt`) with
   `research/joint_time_oracle/native_timeopt_nan_trace.py`-style probes. Check
   whether the success mask is computed before or after the non-finite
   integration. This is a native correctness bug, independent of the search
   contribution. Report it separately, with a minimal reproducer (task, payload,
   seed).
6. **Deliverable:** a decomposition of native failures into
   (a) initialization restriction (a law exists on the native geometry),
   (b) native geometry torque-infeasible (geometry change needed), and
   (c) unresolved.

### P2: hard cases under original limits (user item 2)

1. `task_bank` (seed 20261110) → `tasks.json`, `rejected_proposals.json`,
   `cells.json` (1980 cells; about 1370 admissible in the pilot).
2. `mine_native` R1 on all admissible cells, then `--restarts 4 --escalate-from 1`,
   then `--restarts 16 --escalate-from 4`. Persist the frozen to-do list before
   each stage.
3. Classification (`report.py`): ENDPOINT_INADMISSIBLE, NATIVE_SUCCESS,
   CONFIRMED_HARD_FAILURE (needs an independently validated alternative),
   UNRESOLVED_FAILURE, NATIVE_SUBOPTIMAL (>= 5% shorter and validated). Report
   broad and enriched separately, and by payload and scene.
4. Sensitivity check: rerun R1 for the failure cells and a control sample on
   upstream `main` (`5e4f23d`, new RRT growth), so a stale baseline is not
   blamed.
5. **Accept:** >= 30 confirmed hard failures or native-suboptimal cells from
   >= 10 source tasks in the development split. If fewer exist, report the true
   rate. Do not manufacture cases.

### P3: TimeOpt-guided search (user item 3, the contribution)

Arms, all at matched `max_proposals` with an identical round 0:

* **A** native cuRoboV2-Fast at its top budget;
* **B** blind global proposals, with K107 used only for scoring;
* **C** K107-ranked elites, with children that are global + elite-local;
* **D** C + bottleneck-directed detours, at the K107 bottleneck (s*, j*), along
  −∇|g_j*| when torque-limited, or toward the chord when kinematically limited.

Protocol:

1. Development split only. Run `run_campaign` with `--search-seeds 3
   --max-proposals 128`, with and without `--native-seeds`.
2. Tune ONLY on development: `global_fraction`, `directed_fraction`, amplitudes
   and elite count, for at most 3 tuning rounds. Log every configuration. Then
   freeze `SearchConfig` and the source SHA in `freeze.json`.
3. Validate: run `independent_validate` (Pinocchio) on all exports, then
   `report.py`.
4. Metrics:
   * rescue@{0.1, 0.3, 1, 3, ∞} s, using the validated first-feasible candidate;
   * proposals to first feasible;
   * validated best duration vs native (paired by task);
   * oracle calls;
   * wall time;
   * retimer-only gain (K107 on the exact native geometry), which separates
     "better retimer" from "better geometry".
5. Paired by source task, average payload cells per task first, Wilcoxon
   signed-rank for D vs B, C vs B and D vs C.
6. **Accept (GO):** D > B in validated rescue rate at matched budget, or a
   significantly shorter paired validated duration (p < 0.05). PARTIAL: only
   fewer proposals to first feasible. NO_GO: B matches or beats D.

### P4: practical planner and latency

1. Hybrid planner **E**: native attempt 1. If its TimeOpt seed is infeasible on
   all lanes, stop native early and run the guided search with a wall-time
   budget. Compare E vs A on total request latency, including the failed native
   attempts that A spends.
2. Make the oracle cheaper:
   (a) a TOPP-RA-only score (about 12 ms) for ranking, with Newton only on
       elites and the final candidate;
   (b) batch candidates in one K107 call (the K107 campaign code has batched
       routes);
   (c) a coarser grid (1025) inside the loop and full resolution for the final
       result.
   Report the rank correlation of cheap vs full scores.
3. Measure GPU-synchronized latencies, excluding first-call compilation and
   reporting it separately. Use the same GPU for all arms.
4. **Accept:** at a 1 s budget, E rescues more native failures than A, and on
   native successes it is not slower than A by more than the configured
   improvement budget.

### P5: confirmation (held out)

Only after `freeze.json` exists: run P2 (if not already run) and P3/P4 with the
frozen configuration on the confirmation split. No tuning. Report all
negatives. The decision rule applies to the confirmation numbers.

### P6: reporting

`report.py` HTML/JSON plus a short README section. Commit code, small JSON
summaries and HTML. Keep raw NPZ local, with paths and SHA256 recorded. Push
to the research branch only.

## 5. Known limitations

* Validation is sampled (2049–8193 points), not a continuous-time certificate.
* The payload is a solid sphere with one collision sphere. 4–5 kg exceeds the
  Panda rating (3 kg): research-hypothetical, with nominal actuator limits.
* K107's constraint contract is C3 without torque rate. Torque rate is reported
  only as a post-hoc audit if added.
* Search candidates are quintic. A native-path seed is a least-squares fit:
  never report search results as exact-native-path results.
* Native latency in research runs includes the lane-capture hook (a few ms of
  device→host copies). Report it, or rerun A without capture for latency.
