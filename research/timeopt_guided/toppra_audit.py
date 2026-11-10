# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Are the TOPP-RA laws on ``TOPPRA_FEASIBLE_NEWTON_START_FAILED`` native lanes real? (CUDA env)

PLAN P1 item 3. For every such lane (from ``seed_diagnosis_R*.json``):

1. Re-solve TOPP-RA (copp topp2_ra, SAME constraint build as K107's
   ``toppra_newton_start_v2``: velocity^2 and jerk caps as first-order bounds,
   acceleration and torque as second-order rows, margin 0.995 of Scaled95) on
   three grids: ``oracle`` (2049 samples, stride 3; what K107 used),
   ``stride1`` (2049, stride 1) and ``fine`` (8193, stride 1).
2. Rebuild the exact time law: TOPP-RA's x = sdot^2 is piecewise linear in s,
   i.e. the path acceleration u = sddot is constant per segment, so
   s(tau) = s_i + sdot_i tau + u_i tau^2 / 2 in closed form.
3. Dense audit BETWEEN grid nodes (``--sub`` points per segment) on the exact native
   cubic path with cuRobo RNEA (same payload): velocity, acceleration, torque
   utilization vs Scaled95. Jerk is reported separately: a TOPP-RA law has
   piecewise-constant u, so jerk is unbounded at nodes; a second-order law is
   NOT a jerk-limited (C3) law.
4. Export the ``fine`` trajectory for ``independent_validate`` (Pinocchio).

Verdict per lane:
  REAL_SECOND_ORDER_LAW  feasible on all grids, dense V/A/tau util <= 1 + 1e-3 on
                         ``fine`` (independent check reported separately)
  MARGINAL               feasible on all grids, dense excess <= 2% and shrinking
                         under refinement
  GRID_ARTEFACT          infeasible after refinement, or dense excess > 2%
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import Payload, read_json, write_json
from .geometry import native_cubic_to_clamped

GRIDS = (("oracle", 2049, 3), ("stride1", 2049, 1), ("fine", 8193, 1))
MARGIN = 0.995


def toppra_profile(terms, stride: int, *, jerk_cap: bool = True) -> dict:
    """Raw TOPP-RA profile x(s) on the stride-subsampled K107 grid (or failure)."""
    import copp_py as copp
    import torch
    from curobo_timeopt.retiming.bspline.pdf_toppra_seed import toppra_stage

    ends = torch.zeros(2, device=terms.grid.device, dtype=torch.float32)
    staged = toppra_stage(terms, ends, stride, asynchronous=False)
    host, small = staged["host"].numpy(), staged["small"].numpy()
    limit = small[2:].reshape(staged["limit_shape"])
    s = np.ascontiguousarray(host[0, :, 0])
    n, D = s.size, host.shape[2]
    vel_a, acc_a, tq_a, acc_b, tq_b, g, q_sss = (np.ascontiguousarray(a) for a in host[1:])
    amax = ((MARGIN**2 * limit[0])[None] / np.maximum(vel_a, 1e-300)).min(1, keepdims=True)
    if jerk_cap:
        cap = ((MARGIN * limit[3])[None] / np.maximum(np.abs(q_sss), 1e-300)) ** (2.0 / 3.0)
        amax = np.minimum(amax, cap.min(1, keepdims=True))
    acc_lim = np.ascontiguousarray(np.broadcast_to(MARGIN * limit[1], (n, D)))
    tq_lim = MARGIN * limit[2]
    c = copp.Constraints(D, capacity=n)
    c.append_s(s)
    c.add_constraint_1st(np.ascontiguousarray(amax), 0)
    c.add_constraint_2nd(acc_a, acc_b, acc_lim, 0)
    c.add_constraint_2nd(acc_a, acc_b, np.ascontiguousarray(-acc_lim), 0, is_negative=True)
    c.add_constraint_2nd(tq_a, tq_b, np.ascontiguousarray(tq_lim[None] - g), 0)
    c.add_constraint_2nd(tq_a, tq_b, np.ascontiguousarray(-tq_lim[None] - g), 0, is_negative=True)
    try:
        prob = copp.solver.topp2_ra.Problem(c, idx_s_interval=(0, n - 1), a_boundary=(0.0, 0.0))
        x = np.asarray(copp.solver.topp2_ra.solve(prob, copp.solver.topp2_ra.Options()), dtype=np.float64)
    except copp.CoppError as error:
        return {"feasible": False, "reason": str(error)[:160], "nodes": int(n)}
    if x.shape != s.shape or not np.isfinite(x).all():
        return {"feasible": False, "reason": "invalid profile", "nodes": int(n)}
    x = np.maximum(x, 0.0)
    node_tau = tq_a * x[:, None] + g  # u-term added below
    u = np.diff(x) / (2.0 * np.diff(s))
    uu = np.r_[u, u[-1]]
    node_tau = node_tau + tq_b * uu[:, None]
    return {"feasible": True, "nodes": int(n), "s": s, "x": x, "u": u,
            "duration_s": float(np.sum(2.0 * np.diff(s) / (np.sqrt(x[1:]) + np.sqrt(x[:-1]) + 1e-300))),
            "node_torque_util": float(np.max(np.abs(node_tau) / (limit[2][None]))),
            "gravity_util_max": float(np.max(np.abs(g) / limit[2][None]))}


def time_law(prof: dict, sub: int) -> dict:
    """Exact (t, s, sdot, sddot) at ``sub`` points per segment plus all nodes."""
    s, x, u = prof["s"], prof["x"], prof["u"]
    v = np.sqrt(x)
    dt = 2.0 * np.diff(s) / np.maximum(v[1:] + v[:-1], 1e-300)
    t0 = np.r_[0.0, np.cumsum(dt)]
    frac = np.arange(sub) / sub
    tau = dt[:, None] * frac[None]  # [segments, sub]
    ss = s[:-1, None] + v[:-1, None] * tau + 0.5 * u[:, None] * tau**2
    sd = v[:-1, None] + u[:, None] * tau
    T = np.r_[(t0[:-1, None] + tau).ravel(), t0[-1]]
    S = np.r_[ss.ravel(), s[-1]]
    SD = np.r_[np.maximum(sd, 0.0).ravel(), v[-1]]
    SDD = np.r_[np.repeat(u, sub), u[-1]]
    node = np.r_[(np.arange(len(T) - 1) % sub) == 0, True]
    return {"t": T, "s": np.clip(S, 0.0, 1.0), "sd": SD, "sdd": SDD, "node": node}


def dense_audit(oracle, conv: dict, law: dict, limits: dict) -> dict:
    from scipy.interpolate import BSpline

    spl = BSpline(conv["knots"], conv["controls"], 3)
    q, qs, qss, qsss = (spl(law["s"], nu=k) for k in range(4))
    sd, sdd = law["sd"][:, None], law["sdd"][:, None]
    qd = qs * sd
    qdd = qss * sd**2 + qs * sdd
    jerk = qsss * sd**3 + 3.0 * qss * sd * sdd
    tau = oracle.torque(q, qd, qdd)
    util = {
        "torque": np.max(np.abs(tau) / oracle.torque_limit[None], axis=1),
        "velocity": np.max(np.abs(qd) / limits["velocity"][None], axis=1),
        "acceleration": np.max(np.abs(qdd) / limits["acceleration"][None], axis=1),
    }
    node, out = law["node"], {}
    for k, v in util.items():
        out[k + "_node"] = float(v[node].max())
        out[k + "_between"] = float(v[~node].max()) if (~node).any() else None
    out["jerk_within_segments"] = float(np.max(np.abs(jerk) / limits["jerk"][None]))
    g = np.abs(oracle.gravity(q)) / oracle.torque_limit[None]
    over = np.any(g > 1.0, axis=1)
    out["time_in_gravity_overload_s"] = float(np.sum(np.diff(law["t"])[over[:-1]])) if over.any() else 0.0
    out["min_speed_in_overload"] = float(law["sd"][over].min()) if over.any() else None
    out["max_gravity_util"] = float(g.max())
    traj = {"time": law["t"], "q": q, "qd": qd, "qdd": qdd, "jerk": jerk}
    return out, traj


def k107_retries(oracle, conv: dict) -> dict:
    """Full K107 route (TOPP-RA start -> Newton -> C3 screen) with finer TOPP-RA and larger fits."""
    import torch

    k = oracle.k
    out = {}
    for stride, fit_k, samples in ((3, 29, 2049), (1, 29, 4097), (1, 55, 4097), (1, 107, 4097)):
        key = f"stride{stride}_fit{fit_k}_G{samples}"
        terms = oracle.polish_terms(conv["controls"], degree=3, knots=conv["knots"], samples=samples)
        ends = torch.zeros(2, device=terms.grid.device, dtype=torch.float32)
        try:
            seed, diag = k["toppra_newton_start_v2"](terms, ends, margin=0.995, fixed_ends=True, stride=stride,
                                                     jerk_cap=True, fit_k=fit_k, batched_scale=True,
                                                     scale_backend="torch")
        except ValueError as error:
            out[key] = {"status": "START_ERROR", "reason": str(error)[:80]}
            continue
        if seed is None:
            out[key] = {"status": "TOPPRA_INFEASIBLE"}
            continue
        try:
            res = k["polish_timing_law_pd_fast"](terms, seed, margin=0.995, fixed_ends=True,
                                                 max_iterations=16, max_decrease=3, linear_solver="banded",
                                                 sync_mode="fused", kernel_variant="split", restore_bisect=True)
            out[key] = {"status": "NEWTON_OK", "duration_s": float(res.duration), "scale": diag.get("scale")}
        except ValueError as error:
            out[key] = {"status": "NEWTON_START_FAILED", "scale": diag.get("scale"), "reason": str(error)[:80]}
    return out


def verdict(per_grid: dict) -> str:
    if not all(per_grid[name]["feasible"] for name, _, _ in GRIDS):
        return "GRID_ARTEFACT"
    worst = lambda r: max(r["dense"][k] or 0.0 for k in r["dense"] if k.endswith(("_node", "_between")))  # noqa: E731
    fine, coarse = worst(per_grid["fine"]), worst(per_grid["oracle"])
    if fine <= 1.0 + 1e-3:
        return "REAL_SECOND_ORDER_LAW"
    if fine <= 1.02 and fine <= coarse:
        return "MARGINAL"
    return "GRID_ARTEFACT"


def main() -> None:
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg

    from .k107_oracle import K107Oracle
    from .seed_diagnosis import native_limits

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="mine_native output with seed diagnoses")
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sub", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    kp = KinematicsCfg.from_robot_yaml_file("franka.yml").kinematics_config
    limits = native_limits(kp)  # 0.95-scaled V/A/J
    oracles, rows, seen = {}, [], {}
    for f in sorted(a.run.glob(f"cells/*/*/seed_diagnosis_R{a.restarts}.json")):
        diag = read_json(f)
        lanes = [r for r in diag["lanes"] if r["class"] == "TOPPRA_FEASIBLE_NEWTON_START_FAILED"]
        if not lanes:
            continue
        rec = read_json(f.parent / f"native_R{a.restarts}.json")
        p = Payload.from_json(rec["payload"])
        oracle = oracles.setdefault(p.label, K107Oracle(p, warmup=False))
        z = dict(np.load(f.parent / f"native_R{a.restarts}_lanes.npz"))
        for lane in lanes:
            if a.limit and len(rows) >= a.limit:
                break
            att, li = lane["attempt"], lane["lane"]
            conv = native_cubic_to_clamped(z[f"{att}_start_boundary"][li][:, :7],
                                           z[f"{att}_controls"][li][:, :7], z[f"{att}_goal"][li][:7])
            gkey = (p.label, np.round(conv["controls"], 6).tobytes())
            if gkey in seen:
                rows.append({"task_id": rec["task_id"], "payload": p.label, "attempt": att, "lane": li,
                             "duplicate_of": seen[gkey], "verdict": rows[seen[gkey]]["verdict"]})
                continue
            seen[gkey] = len(rows)
            per_grid, traj = {}, None
            for name, samples, stride in GRIDS:
                terms = oracle.polish_terms(conv["controls"], degree=3, knots=conv["knots"], samples=samples)
                prof = toppra_profile(terms, stride)
                entry = {k: v for k, v in prof.items() if not isinstance(v, np.ndarray)}
                if prof["feasible"]:
                    entry["dense"], tr = dense_audit(oracle, conv, time_law(prof, a.sub), limits)
                    if name == "fine":
                        traj = tr
                per_grid[name] = entry
            row = {"task_id": rec["task_id"], "payload": p.label, "attempt": att, "lane": li,
                   "grids": per_grid, "verdict": verdict(per_grid), "k107_retries": k107_retries(oracle, conv)}
            if traj is not None:
                stem = a.output / rec["task_id"] / p.label / f"{att}_lane{li}_toppra_fine"
                stem.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(str(stem) + ".traj.npz", **traj)
                from .run_campaign import physical_limits

                write_json(Path(str(stem) + ".traj.json"), {
                    "payload": p.to_json(), "scene": rec["scene"], "start": traj["q"][0].tolist(),
                    "goal": traj["q"][-1].tolist(), "limits": physical_limits(kp), "kind": "toppra_fine",
                    "collision_samples": 4097})
                row["export"] = str(stem)
            rows.append(row)
            d = {k: (per_grid[k].get("duration_s"), per_grid[k]["feasible"]) for k in per_grid}
            print(rec["task_id"], p.label, att, li, row["verdict"], d,
                  {k: v["status"] for k, v in row["k107_retries"].items()}, flush=True)
    write_json(a.output / "toppra_audit.json", rows)


if __name__ == "__main__":
    main()
