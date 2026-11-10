# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Why does native TimeOpt fail? Exact-native-path diagnosis (CUDA env).

Native TimeOpt (``timeopt_io_kernel.cuh``: ``fused_coefficients_and_seed_bounds``
+ ``fused_seed``) initializes with ONE constant squared path speed ``x`` for the
whole path: per sample it intersects velocity/acceleration/jerk upper bounds and
the torque interval ``lower <= g + x*B <= upper`` (B = M q'' + C(q') q'), then
reduces over the path. If gravity alone violates the limit somewhere (|g| > 0.95
tau_max), a positive LOWER bound on x appears there, while other samples cap x
from above; the interval is often empty and every lane is rejected
("seed infeasible") before Newton runs. An empty uniform interval does NOT prove
that no non-uniform x(s) exists.

For every captured native lane of a failed cell this tool:
  1. converts the lane EXACTLY to a clamped cubic (no fitting);
  2. replays the uniform-x seed interval independently (cuRobo RNEA on a dense
     grid) and compares with the captured CUDA ``seed_feasible`` flag;
  3. computes the static gravity-overload profile;
  4. runs K107 TOPP-RA + Newton on the SAME curve and limits.
Classification per lane:
  NATIVE_INIT_RESTRICTION   uniform seed empty, K107 finds a law (screen pass)
  TOPPRA_FEASIBLE_NEWTON_START_FAILED  TOPP-RA finds a non-uniform law but K107's
                            Newton start cannot be made strictly interior
                            (unresolved; often a gravity overload "swung" through)
  GEOMETRY_TORQUE_INFEASIBLE TOPP-RA (complete for the discretized 2nd-order
                            problem with jerk cap) finds none; torque_expansion>1
  K107_SCREEN_REJECTED / K107_ERROR   unresolved
  NATIVE_SEED_FEASIBLE      native seed was non-empty (other failure mode)
All K107 outcomes are preliminary until independently validated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import LIMIT_SCALE, Payload, read_json, write_json
from .geometry import native_cubic_to_clamped, native_support, native_uniform_spline, nonstationary_interval


def uniform_seed_interval(oracle, spline, s: np.ndarray, vlim, alim, jlim) -> dict:
    """Replay of the native constant-x seed interval on progress grid ``s``."""
    q, qs, qss, qsss = (spline(s, nu=k) for k in range(4))
    g = oracle.gravity(q)
    B = oracle.torque(q, qs, qss) - g
    tau = oracle.torque_limit  # already 0.95-scaled
    hi = np.full(q.shape, np.inf)
    lo = np.zeros(q.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        v, a, j = np.abs(qs), np.abs(qss), np.abs(qsss)
        hi = np.minimum(hi, np.where(v > 0, (vlim / v) ** 2, np.inf))
        hi = np.minimum(hi, np.where(a > 0, alim / a, np.inf))
        hi = np.minimum(hi, np.where(j > 0, (jlim / j) ** (2.0 / 3.0), np.inf))
        left = (-tau - g) / B
        right = (tau - g) / B
        dyn = B != 0
        lo = np.where(dyn, np.maximum(lo, np.minimum(left, right)), lo)
        hi = np.where(dyn, np.minimum(hi, np.maximum(left, right)), hi)
    static_bad = (~dyn) & (np.abs(g) > tau)
    H, L = float(np.min(hi)), float(np.max(lo))
    i_lo, j_lo = np.unravel_index(int(np.argmax(lo)), lo.shape)
    i_hi, j_hi = np.unravel_index(int(np.argmin(hi)), hi.shape)
    return {
        "x_hi": H, "x_lo": L, "feasible": bool(H > 0 and H >= L and not static_bad.any()),
        "lo_binding": {"s": float(s[i_lo]), "joint": int(j_lo)},
        "hi_binding": {"s": float(s[i_hi]), "joint": int(j_hi)},
        "max_gravity_util": float(np.max(np.abs(g) / tau)),
        "gravity_overload_fraction": float(np.mean(np.any(np.abs(g) > tau, axis=1))),
    }


def diagnose_cell(cell_dir: Path, restarts: int, oracle, limits: dict, *, max_lanes: int | None = None) -> dict:
    lanes_file = cell_dir / f"native_R{restarts}_lanes.npz"
    out = {"cell": str(cell_dir), "lanes": []}
    if not lanes_file.exists():
        out["status"] = "NO_CAPTURED_LANES"
        return out
    z = dict(np.load(lanes_file))
    attempts = sorted({k.split("_")[0] for k in z})
    vlim, alim, jlim = (limits[k] for k in ("velocity", "acceleration", "jerk"))
    for att in attempts:
        controls = z[f"{att}_controls"][..., :7]
        sb = z[f"{att}_start_boundary"][..., :7]
        goal = z[f"{att}_goal"][..., :7]
        seed_flag = z.get(f"{att}_seed_feasible")
        for lane in range(controls.shape[0]):
            if max_lanes is not None and len(out["lanes"]) >= max_lanes:
                break
            rec = {"attempt": att, "lane": lane,
                   "native_seed_feasible": None if seed_flag is None else bool(seed_flag[lane])}
            try:
                support = native_support(sb[lane], controls[lane], goal[lane])
                first, last, spans = nonstationary_interval(support)
                spl = native_uniform_spline(sb[lane], controls[lane], goal[lane])
                s = np.linspace(first / spans, last / spans, 1025)
                rec["uniform_seed_replay"] = uniform_seed_interval(oracle, spl, s, vlim, alim, jlim)
                conv = native_cubic_to_clamped(sb[lane], controls[lane], goal[lane])
                res = oracle.retime(conv["controls"], degree=3, knots=conv["knots"])
                rec["k107"] = res.to_json()
                rec["conversion_max_error"] = conv["conversion_max_error"]
            except Exception as error:
                rec["error"] = f"{type(error).__name__}: {error}"[:300]
            rec["class"] = classify_lane(rec)
            out["lanes"].append(rec)
    counts: dict[str, int] = {}
    for r in out["lanes"]:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    out["class_counts"] = counts
    agree = [r for r in out["lanes"] if r.get("native_seed_feasible") is not None and "uniform_seed_replay" in r]
    out["seed_replay_agreement"] = (
        None if not agree else float(np.mean([r["native_seed_feasible"] == r["uniform_seed_replay"]["feasible"]
                                              for r in agree])))
    return out


def classify_lane(rec: dict) -> str:
    if "error" in rec:
        return "DIAGNOSIS_ERROR"
    k = rec["k107"]["status"]
    native_ok = rec.get("native_seed_feasible")
    if native_ok:
        return "NATIVE_SEED_FEASIBLE"
    if k == "FEASIBLE":
        return "NATIVE_INIT_RESTRICTION"
    if k == "TOPPRA_INFEASIBLE":
        return "GEOMETRY_TORQUE_INFEASIBLE"
    if k == "NEWTON_START_FAILED":
        return "TOPPRA_FEASIBLE_NEWTON_START_FAILED"
    return "K107_" + k


def native_limits(planner_kp) -> dict:
    jl = planner_kp.joint_limits
    return {k: getattr(jl, k).detach().cpu().numpy()[-1, :7] * LIMIT_SCALE if getattr(jl, k).ndim == 2
            else getattr(jl, k).detach().cpu().numpy()[:7] * LIMIT_SCALE
            for k in ("velocity", "acceleration", "jerk")}


def main() -> None:
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg

    from .k107_oracle import K107Oracle

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="mine_native output directory")
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--max-lanes", type=int, default=None)
    a = ap.parse_args()
    kp = KinematicsCfg.from_robot_yaml_file("franka.yml").kinematics_config
    limits = native_limits(kp)
    oracles: dict[str, K107Oracle] = {}
    summary = []
    for f in sorted(a.run.glob(f"cells/*/*/native_R{a.restarts}.json")):
        rec = read_json(f)
        if rec["success"]:
            continue
        p = Payload.from_json(rec["payload"])
        oracle = oracles.setdefault(p.label, K107Oracle(p))
        d = diagnose_cell(f.parent, a.restarts, oracle, limits, max_lanes=a.max_lanes)
        write_json(f.parent / f"seed_diagnosis_R{a.restarts}.json", d)
        summary.append({"task_id": rec["task_id"], "payload": p.label, "class_counts": d.get("class_counts"),
                        "seed_replay_agreement": d.get("seed_replay_agreement")})
        print(rec["task_id"], p.label, d.get("class_counts"), d.get("seed_replay_agreement"), flush=True)
    write_json(a.run / f"seed_diagnosis_R{a.restarts}_summary.json", summary)


if __name__ == "__main__":
    main()
