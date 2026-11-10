# TimeOpt-guided path planning with nominal-effort payload hard cases

STATUS: RESEARCH HARNESS IMPLEMENTED; NEW EXPERIMENTS NOT EXECUTED.
No PR; no merge to main; do not change production defaults.

## Motivation and model contracts

The user wants a planner that rescues default cuRoboV2-Fast failures
and improves native execution duration by using fast retiming feedback
to actively change geometry. Previous three synthetic joint-selective
30%-effort cases proved existence of alternative geometries and
qualified timings, but were not nominal-actuator problems or a
coupled TimeOpt-guided algorithm. They are development/debug sources,
not prospective holdout evidence.

CuRobo TimeOpt 2026-11-09 trace: input and numerical flags valid,
40/40 seed infeasible, empty seed intervals, invalid finished times.
Inspect curobo/_src/curobolib/kernels/trajectory/timeopt/timeopt_io_kernel.cuh:
the fused_coefficients_and_seed_bounds and fused_seed kernels intersect
ALL local velocity, acceleration, jerk, and torque limits into ONE
UNIFORM squared speed x=(ds/dt)^2. An empty uniform-x interval does
not rule out a nonuniform x(s), so it is not a global path
infeasibility certificate.

Retimer reference: pkicki/time_optimal_curobo master, initial
investigation SHA 94d25ecde95bc7f31851b0b9a78f61bd5f7cb65e.
Its default K107 TOPP-RA+Newton handles V/A/J/torque/torque-rate.
Native TimeOpt handles V/A/J/torque (no torque rate).
Match FOUR physical dynamic families and the position+collision
contract as the primary head-to-head. Torque-rate is a separately
reported posthoc audit, unless an explicitly separate five-family
contract is registered. Pin exact commits and compare same original
signed torque bounds, gravity, payload inertia and endpoints.

## P0: implement, pin and test exact K107 adapter

Read these research modules:
payload_contract.py; payload_native.py; payload_geometry_audit.py;
payload_parity.py; nominal_payload_task_bank.py;
native_uniform_seed_replay.py; retimer_oracle.py;
timeopt_feedback_search.py; run_timeopt_feedback.py;
payload_hardcase_registry.py; tests/test_joint_time_payload_feedback.py.

The cross-repo strict payload-timeopt-bridge-v1 JSON contract is
implemented on the cuRobo side, but the real K107 backend is NOT
yet connected. Implement and commit a GPU-capable research adapter
in time_optimal_curobo (e.g. scripts/research_payload_k107_adapter.py).
It must accept --request/--response and use the pinned TOPP-RA+Newton
code, not substitute IPOPT. Pin source commit SHA in every result.

Accept exact 22x7 quintic and original native CUBIC paths as
separate geometry types; the existing quintic bridge is for search,
and the exact native comparison requires a separately added exact
cubic schema. Enforce q,qs,qss,qsss replay parity on a dense grid.
Implement strict-rest vs native endpoint contract alignment.
Build GPU payload RNEA/JVP from the actual rigid inertia and COM,
not effort scaling. Produce time law samples and structured
feasibility/infeasibility diagnostics, with model/path/limit/request
SHA echo. No output is independently qualified by the oracle alone.
For native TimeOpt implement a counterpart adapter as a separate
research wrapper; do not change default kernels.

Tests: numerical start, TOPP2-RA seed, Newton/fallback,
early exit, finite reconstruction, signed effort, same source
geometry, endpoint contract, GPU/Pinocchio RNEA+JVP under payload,
C3 online screen and independent sampled S8193/S32768.
Compile CUDA once before scored timing; log warm/first-call time.

## P1: exact native uniform seed failure diagnosis

The native kernel uses one constant x across its sampled path.
native_uniform_seed_replay.py reproduces its global x interval
from independent q,qs,qss,qsss and Pinocchio gravity+dynamic torque.
Use ACTUAL native path grid and original signed limits (the old
synthetic 30%-effort cases are marked historical). Compare its
per-point lower/upper bounds, tile reductions and final interval to
captured native CUDA seed_bounds with FP32 tolerance and knot-event
points. Preserve limiting joint, s location, family and sign.

On the EXACT SAME native cubic path (not a fitted quintic),
run native TimeOpt versus K107 TOPP-RA+Newton and optionally an
offline IPOPT reference. Compare native uniform seed, necessary
path LP, K107 start validity, Newton outcome, duration and
independent physical+Scaled95 S8193/event-aware S32768 audit.
If native constant-x seed is empty but K107 independently certifies
a NONUNIFORM law on exactly the same path/limits, this isolates an
initialization completeness restriction, not necessarily a Newton
bug. If K107 also fails and the necessary LP gap is positive,
the path is intrinsically torque infeasible; if LP zero and no
constructive law, mark unresolved. Never claim K107 solves a
different fitted geometry as exact native-path rescue.

Optionally research a nonuniform native seed injection under an
opt-in flag, leaving production defaults untouched; only after
verified exact-path K107 timing evidence.

## P2: MANY new hard tasks, ORIGINAL actuator limits, physical payloads

Nominal signed joint effort limits from the original franka.yml,
UNMODIFIED. Use actual rigid-tool payload rather than synthetic
torque derating. Registered mass grid:
0, 0.5, 1, 2, 3, 4 kg;
COM (0,0,.12) or (.04,0,.16) meters;
positive diagonal inertia, 6-cm attached sphere.
Mass 4 kg is research hypothetical, not an authorized hardware load.

payload_native.py combines original hand inertia and tool payload
via parallel-axis theorem; payload_contract.py adds the identical
rigid inertia to independent Pinocchio; payload_geometry_audit.py
adds the same sphere to independent world/self collisions.
Before ANY scored run, log >=32 identical q,qd,qdd samples per
payload and verify native vs independent torque <=0.15 Nm + 1%
relative; native sphere world/self signed distance <=1mm versus
independent. Confirm ORIGINAL actuator effort bytes unchanged.
Missing collision parity, payload dynamics mismatch, or changed
limits means BLOCKED, not a native-planner failure.

Generate task bank with nominal_payload_task_bank.py:
40 broad + 20 separately gravity-enriched start/goal pairs in
free/table/cluttered = 180 source tasks, paired across 12 payload
profiles = 2160 task/scenario cells before endpoint exclusions.
Store ALL rejected endpoint proposals and source SHA. Enriched cohort
uses a nominal 2-kg gravity-interior heuristic; do not pool it with
unbiased broad cohort or relabel it random. Historical task endpoint
overlap excluded. Task-level deterministic hash split:
80% development /20% confirmation, SAME task in one split for all
payload masses, no confirmation-driven algorithm tuning.

Stage GPU resources: all admitted cells at native 1 and 4
restarts, independent static endpoint gravity screening.
Freeze suspected native failures and a source-randomized native-
success control subset BEFORE escalating to 16/32 restarts,
retimer exact-path diagnostic and global-search rescues.
Do not select only examples whose alternatives already succeeded.
Preserve no-result, collision, numeric errors, and failures
before TimeOpt separately. Target >=30 independently witnessed
hard failure/suboptimal cells from >=10 source tasks; if fewer
exist, report the true rate rather than manufacturing cases.

Classification in payload_hardcase_registry.py:
Native finite-budget failure with no independent feasible
alternative = UNRESOLVED, not confirmed.
Native failure + independently physical/Scaled95-qualified
alternative under SAME task/payload/limits = confirmed hard
failure. Native success + separately qualified alternative at
>=5% shorter duration = confirmed native suboptimality.
Endpoint static gravity infeasible is a separate non-planner class.
All models and constraint/scene/endpoint SHA must match.
Require strict 8193 + event-aware 32768 audits and
payload self/world collisions; sampled is not continuous time
certification.

## P3: TimeOpt guided geometric search

Controlled arms:
A native complete cuRoboV2-Fast;
B blind global proposal generation plus retiming after proposals;
C retimer ranked global parent selection without local bottleneck;
D retimer ranked plus broad two-lobe bottleneck-directed geometry
perturbations, retaining independent global exploration.

The three search arms implemented in timeopt_feedback_search.py
share the same first 16 global proposals and deterministic
generation, and a common 128-proposal maximum. They use K107
duration/structured infeasibility information to update next
geometry only for C/D. An unsupported/numerically failed retimer
does NOT imply geometry infeasible. The global-breadth fraction is
maintained because previous local LP gradient guidance failed.

Run matched oracle call budgets and FULL wall-time budgets
100,300,1000,3000 ms, with offline research 10s clearly separate.
Measure CPU/GPU timings, preparation, source transfer and
graph capture separately; do not pretend an untested
cross-repo subprocess interface has online latency.
For native success, seek shorter independently certified
duration; for native failure, seek independently qualified
feasible rescue. Retimer result marked preliminary until
independent Pinocchio with SAME payload and native CUDA
cross-verification, V/A/J/torque and separately torque rate,
world/self collision and strict endpoint checks.

Freeze algorithm source/settings, payload distribution,
source/model checksums and development results BEFORE touching
confirmation tasks. Do not claim gains based on the three
historical synthetic cases; use them only for debugging.

## P4: reporting and decision

Per case/mass/COM, record source seed and native result,
exact path and timing (including failure), native seed interval
witness, exact-path K107 outcome and independent audit,
global blind/ranked/feedback progress, first certified rescue,
quality difference on native successes, full planning time,
number of oracle calls, constraint bottlenecks, payload/scene
collision margin and both model SHAs.

Report all rejection denominators, model parity blocked cells,
static endpoint inadmissible cells and uncertified candidate
counts. Broad and enriched strata separate. Use paired results
by source task, not correlated payload scenarios as independent
samples, for any confidence interval. Report every negative and
numeric-only outcome. User target is practical better planner;
one-off rescued paths are preliminary algorithm evidence.

Final decision:
GO_RESEARCH only with verified source-disjoint improvement
in native rescue and/or qualified duration at matched runtime
plus NO increased independent feasibility failure;
PARTIAL if feedback improves proposal efficiency but not
wall-time or only synthetic cases;
NO_GO if same-budget blind global matches or beats feedback;
BLOCKED if native payload/parity, exact path retimer adapter,
independent certification or holdout protocol fails.

Commit and push implementation, tests and compact HTML/JSON
and SHA manifest only to research branches in both repos;
retain huge raw NPZ locally with exact hash/path.
No PR; no merge; no production defaults.
