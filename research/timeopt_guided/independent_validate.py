# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification of exported trajectories (Pinocchio env, NO torch/cuRobo).

Run with the Pinocchio interpreter (``$PIN_PY``). For every ``*.traj.npz`` +
``*.traj.json`` pair under ``--root`` it writes ``*.validation.json``:

* dynamics: Pinocchio RNEA on the ORIGINAL URDF, model-aligned to cuRobo
  (0.01 kg default inertials as in ``cohort_retime.dynamics``), with the payload
  REPLACING the 0.01 kg attached_object placeholder at the panda_hand frame;
* limits: velocity/acceleration/jerk/torque utilization against both the
  physical limits (1.0) and Scaled95 (0.95), with ORIGINAL effort limits;
* geometry: franka.yml collision spheres + payload sphere vs scene primitives,
  and self-collision pairs with franka.yml ignore lists and buffers;
* endpoints: start/goal position error and rest (velocity) at both ends.
Sampled checks; not continuous-time certificates. Also checks payload model
parity on ``parity_*.npz`` files (cuRobo tau exported by the CUDA side).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "curobo/content/assets/robot/franka_description/franka_panda.urdf"
ROBOT_YML = ROOT / "curobo/content/configs/robot/franka.yml"
SCENE_DIR = ROOT / "curobo/content/configs/scene"
SCENES = {"free": None, "table": "collision_test.yml", "cluttered": "collision_primitives_3d.yml",
          **{f"spoke_r{r}{b}": f"research/timeopt_guided/scenes/spoke_r{r}{b}.yml" for r in (55, 65, 75, 85)
             for b in ("", "_b", "_c", "_b_c")}}
PARITY_ABS_NM, PARITY_REL = 0.15, 0.01
TOL = 1e-3  # relative tolerance on utilization (sampling/float32)


def build_model(payload: dict):
    """Pinocchio model aligned to cuRobo, with the payload on attached_object."""
    model = pin.buildModelFromUrdf(str(URDF))
    placeholder = pin.Inertia(0.01, np.zeros(3), np.eye(3) * 1e-4)
    right = model.frames[model.getFrameId("right_gripper", pin.BODY)]
    model.inertias[right.parentJoint] -= right.placement.act(placeholder)
    for name in ("panda_link8", "panda_leftfinger", "panda_rightfinger", "panda_hand"):
        f = model.frames[model.getFrameId(name, pin.BODY)]
        model.inertias[f.parentJoint] += f.placement.act(placeholder)
    mass = float(payload["mass_kg"])
    hand = model.frames[model.getFrameId("panda_hand", pin.BODY)]
    if mass > 0:
        # Replace cuRobo's default attached_object placeholder (0.01 kg at hand origin).
        model.inertias[hand.parentJoint] -= hand.placement.act(placeholder)
        r = float(payload["radius_m"])
        inertia = pin.Inertia(mass, np.asarray(payload["com_m"], float), np.eye(3) * 0.4 * mass * r * r)
        model.inertias[hand.parentJoint] += hand.placement.act(inertia)
    state = pin.neutral(model)
    for name in ("panda_finger_joint1", "panda_finger_joint2"):
        state[model.joints[model.getJointId(name)].idx_q] = 0.04
    qi = [model.joints[model.getJointId(f"panda_joint{i}")].idx_q for i in range(1, 8)]
    vi = [model.joints[model.getJointId(f"panda_joint{i}")].idx_v for i in range(1, 8)]
    return model, model.createData(), state, qi, vi


class Geometry:
    """Sphere robot model (franka.yml + payload sphere) vs scene primitives."""

    def __init__(self, model, data, state, qi, scene: str, payload: dict):
        cfg = yaml.safe_load(ROBOT_YML.read_text())["robot_cfg"]["kinematics"]
        self.model, self.data, self.state, self.qi = model, data, state.copy(), qi
        links, local, radii = [], [], []
        for name, spheres in cfg["collision_spheres"].items():
            for s in spheres:
                links.append(name)
                local.append(s["center"])
                radii.append(s["radius"])
        if float(payload["mass_kg"]) > 0:
            links.append("attached_object")
            local.append(list(payload["com_m"]))
            radii.append(float(payload["radius_m"]))
        self.links, self.local, self.radii = links, np.asarray(local, float), np.asarray(radii, float)
        frame_of = lambda n: "panda_hand" if n == "attached_object" else n  # noqa: E731
        self.groups = [(np.array([i for i, n in enumerate(links) if n == name]),
                        model.getFrameId(frame_of(name), pin.BODY)) for name in sorted(set(links))]
        ignore, pad = cfg["self_collision_ignore"], cfg["self_collision_buffer"]
        pairs = [(i, j) for i in range(len(links)) for j in range(i + 1, len(links))
                 if links[i] != links[j] and links[j] not in ignore.get(links[i], [])
                 and links[i] not in ignore.get(links[j], [])]
        self.ia, self.ib = np.asarray(pairs).T
        self.self_limit = self.radii[self.ia] + self.radii[self.ib] + np.asarray(
            [pad.get(links[i], 0) + pad.get(links[j], 0) for i, j in pairs])
        f = SCENES[scene]
        path = None if f is None else (ROOT / f if "/" in f else SCENE_DIR / f)
        sc = yaml.safe_load(path.read_text()) if path else {}
        self.cuboids = []
        for b in (sc.get("cuboid") or {}).values():
            p = b["pose"]
            self.cuboids.append((np.asarray(p[:3]), Rotation.from_quat([p[4], p[5], p[6], p[3]]),
                                 np.asarray(b["dims"]) / 2))
        self.spheres = [(np.asarray(s.get("position", s.get("pose", [0, 0, 0])[:3])), float(s["radius"]))
                        for s in (sc.get("sphere") or {}).values()]
        self.capsules = []
        for c in (sc.get("capsule") or {}).values():
            p = c["pose"]
            rot = Rotation.from_quat([p[4], p[5], p[6], p[3]])
            self.capsules.append((rot.apply(c["base"]) + p[:3], rot.apply(c["tip"]) + p[:3], float(c["radius"])))
        unsupported = set(sc) - {"cuboid", "sphere", "capsule"}
        if unsupported:
            raise ValueError(f"unsupported scene primitives {unsupported}")

    def clearance_profile(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-sample world and self clearance [N] and sphere centers [N, S, 3]."""
        n = len(q)
        world, selfc = np.full(n, np.inf), np.full(n, np.inf)
        xyzs = np.empty((n, len(self.local), 3))
        for k, item in enumerate(q):
            self.state[self.qi] = item
            pin.framesForwardKinematics(self.model, self.data, self.state)
            xyz = xyzs[k]
            for ids, fid in self.groups:
                pose = self.data.oMf[fid]
                xyz[ids] = self.local[ids] @ pose.rotation.T + pose.translation
            selfc[k] = float(np.min(np.linalg.norm(xyz[self.ia] - xyz[self.ib], axis=1) - self.self_limit))
            for center, rot, half in self.cuboids:
                d = np.abs(rot.inv().apply(xyz - center)) - half
                dist = np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(np.max(d, axis=1), 0) - self.radii
                world[k] = min(world[k], float(dist.min()))
            for center, r in self.spheres:
                world[k] = min(world[k], float(np.min(np.linalg.norm(xyz - center, axis=1) - self.radii - r)))
            for a, b, r in self.capsules:
                ax = b - a
                f = np.clip((xyz - a) @ ax / float(ax @ ax), 0, 1)
                world[k] = min(world[k], float(np.min(np.linalg.norm(xyz - (a + f[:, None] * ax), axis=1)
                                                      - self.radii - r)))
        return world, selfc, xyzs

    def clearance(self, q: np.ndarray) -> tuple[float, float]:
        world, selfc = np.inf, np.inf
        for item in q:
            self.state[self.qi] = item
            pin.framesForwardKinematics(self.model, self.data, self.state)
            xyz = np.empty_like(self.local)
            for ids, fid in self.groups:
                pose = self.data.oMf[fid]
                xyz[ids] = self.local[ids] @ pose.rotation.T + pose.translation
            selfc = min(selfc, float(np.min(np.linalg.norm(xyz[self.ia] - xyz[self.ib], axis=1) - self.self_limit)))
            for center, rot, half in self.cuboids:
                d = np.abs(rot.inv().apply(xyz - center)) - half
                dist = np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(np.max(d, axis=1), 0) - self.radii
                world = min(world, float(dist.min()))
            for center, r in self.spheres:
                world = min(world, float(np.min(np.linalg.norm(xyz - center, axis=1) - self.radii - r)))
            for a, b, r in self.capsules:
                ax = b - a
                f = np.clip((xyz - a) @ ax / float(ax @ ax), 0, 1)
                world = min(world, float(np.min(np.linalg.norm(xyz - (a + f[:, None] * ax), axis=1) - self.radii - r)))
        return world, selfc


def torques(model, data, state, qi, vi, q, qd, qdd) -> np.ndarray:
    v, a, out = np.zeros(model.nv), np.zeros(model.nv), np.empty_like(q)
    s = state.copy()
    for k in range(len(q)):
        s[qi], v[vi], a[vi] = q[k], qd[k], qdd[k]
        out[k] = pin.rnea(model, data, s, v, a)[vi]
    return out


def validate(traj: dict, meta: dict) -> dict:
    payload, lim = meta["payload"], meta["limits"]
    model, data, state, qi, vi = build_model(payload)
    q, qd, qdd, jerk = (np.asarray(traj[k], float) for k in ("q", "qd", "qdd", "jerk"))
    tau = torques(model, data, state, qi, vi, q, qd, qdd)
    effort = np.asarray(lim["effort"], float)  # [2, 7] ORIGINAL signed limits
    out = {"samples": int(len(q)), "duration_s": float(traj["time"][-1] - traj["time"][0])}
    util = {
        "velocity": float(np.max(np.abs(qd) / np.asarray(lim["velocity"]))),
        "acceleration": float(np.max(np.abs(qdd) / np.asarray(lim["acceleration"]))),
        "jerk": float(np.max(np.abs(jerk) / np.asarray(lim["jerk"]))),
        "torque": float(np.max(np.maximum(tau / effort[1], tau / effort[0]))),
    }
    pos = np.asarray(lim["position"], float)
    out["position_violation_rad"] = float(max(np.max(pos[0] - q), np.max(q - pos[1]), 0.0))
    out["utilization_physical"] = util
    out["utilization_scaled95"] = {k: v / 0.95 for k, v in util.items()}
    geo = Geometry(model, data, state, qi, meta["scene"], payload)
    if meta.get("gap_check"):
        # Inter-sample guarantee: between consecutive samples a sphere center moves along a
        # path of length ~ |p_{k+1} - p_k| (dense sampling); a sample's clearance must exceed
        # the largest displacement of any sphere to its neighbours (conservative for the
        # world; for self pairs both spheres move, hence the factor 2).
        w, sc, xyz = geo.clearance_profile(q)
        disp = np.max(np.linalg.norm(np.diff(xyz, axis=0), axis=2), axis=1)  # [N-1]
        nb = np.maximum(np.r_[disp, 0.0], np.r_[0.0, disp])
        w_ok = ~np.isfinite(w) | (w >= nb)
        s_ok = sc >= 2.0 * nb
        out["gap_check"] = {"max_step_displacement_m": float(disp.max()), "world_ok": bool(w_ok.all()),
                            "self_ok": bool(s_ok.all()),
                            "min_world_slack_m": None if not np.isfinite(w).any() else float(np.min((w - nb)[np.isfinite(w)])),
                            "min_self_slack_m": float(np.min(sc - 2.0 * nb))}
        world = float(np.min(w)) if np.isfinite(w).any() else np.inf
        selfc = float(np.min(sc))
    else:
        stride = max(1, len(q) // int(meta.get("collision_samples", 2049)))
        world, selfc = geo.clearance(q[::stride])
    out["world_clearance_m"] = None if not np.isfinite(world) else world
    out["self_clearance_m"] = selfc
    out["start_error_rad"] = float(np.max(np.abs(q[0] - np.asarray(meta["start"]))))
    out["goal_error_rad"] = float(np.max(np.abs(q[-1] - np.asarray(meta["goal"]))))
    out["end_speed"] = float(max(np.abs(qd[0]).max(), np.abs(qd[-1]).max()))
    ok_geom = (out["position_violation_rad"] <= 1e-6 and selfc >= 0 and (not np.isfinite(world) or world >= 0))
    ok_end = out["start_error_rad"] <= 1e-3 and out["goal_error_rad"] <= 1e-3 and out["end_speed"] <= 1e-2
    out["physical_pass"] = bool(ok_geom and ok_end and all(v <= 1 + TOL for v in util.values()))
    out["scaled95_pass"] = bool(out["physical_pass"] and all(v <= 1 + TOL for v in out["utilization_scaled95"].values()))
    if "gap_check" in out:
        out["certified"] = bool(out["scaled95_pass"] and out["gap_check"]["world_ok"] and out["gap_check"]["self_ok"])
    out["note"] = "sampled Pinocchio + sphere audit; not a continuous-time certificate"
    return out


def check_parity(path: Path) -> dict:
    z = np.load(path)
    payload = json.loads(str(z["payload_json"]))
    model, data, state, qi, vi = build_model(payload)
    tau = torques(model, data, state, qi, vi, z["q"], z["qd"], z["qdd"])
    ref = z["tau_curobo"]
    err = np.abs(tau - ref)
    ok = bool(np.all(err <= PARITY_ABS_NM + PARITY_REL * np.abs(ref)))
    return {"file": path.name, "payload": payload, "samples": int(len(ref)), "max_abs_nm": float(err.max()),
            "pass": ok}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    parity = [check_parity(p) for p in sorted(a.root.rglob("parity_*.npz"))] if a.shard == 0 else []
    if parity:
        (a.root / "payload_parity.json").write_text(json.dumps(parity, indent=2) + "\n")
        print(f"parity: {sum(p['pass'] for p in parity)}/{len(parity)} pass, "
              f"max {max(p['max_abs_nm'] for p in parity):.4f} Nm")
    n = 0
    for i, meta_file in enumerate(sorted(a.root.rglob("*.traj.json"))):
        if i % a.nshards != a.shard:
            continue
        out = meta_file.with_name(meta_file.name.replace(".traj.json", ".validation.json"))
        if out.exists() and not a.overwrite:
            continue
        meta = json.loads(meta_file.read_text())
        traj = dict(np.load(meta_file.with_name(meta_file.name.replace(".traj.json", ".traj.npz"))))
        try:
            res = validate(traj, meta)
        except Exception as error:
            res = {"error": f"{type(error).__name__}: {error}"[:300], "physical_pass": False, "scaled95_pass": False}
        out.write_text(json.dumps(res, indent=2, default=float) + "\n")
        n += 1
    print(f"validated {n} trajectories under {a.root}")


if __name__ == "__main__":
    main()
