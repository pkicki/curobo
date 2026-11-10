# SPDX-License-Identifier: Apache-2.0
"""Independent configuration-space geometry screening for synthetic feasibility.

Aligns with cohort_validate's Pinocchio sphere collision model. All returned
clearances are sampled; final qualification always reruns cohort_validate.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import yaml
from scipy.spatial.transform import Rotation

from .cohort_retime import ROOT, dynamics


class IndependentGeometryAuditor:
    def __init__(self, scene: str | None, *,
                 extra_attached_spheres: list[dict] | None = None):
        cfg = yaml.safe_load(
            (ROOT / "curobo/content/configs/robot/franka.yml").read_text()
        )
        collision = cfg["robot_cfg"]["kinematics"]
        if extra_attached_spheres is not None:
            if len(extra_attached_spheres)!=4:
                raise ValueError("research payload must use exact four-sphere shape")
            import copy
            collision=copy.deepcopy(collision)
            collision["collision_spheres"]["attached_object"]=extra_attached_spheres
            link=collision["extra_links"]["attached_object"]
            if (link["parent_link_name"]!="panda_hand" or
                list(link["fixed_transform"])!=[0,0,0,1,0,0,0]):
                raise ValueError("attached_object frame is not identical to panda_hand")
        self.model, self.data, self.neutral, self.qi, self.vi = dynamics()
        self.q = self.neutral.copy()
        self.zero = np.zeros(self.model.nv)
        self.rnea_calls = 0
        links = []
        local = []
        radii = []
        for name, spheres in collision["collision_spheres"].items():
            for sphere in spheres:
                links.append(name)
                local.append(sphere["center"])
                radii.append(sphere["radius"])
        self.links = links
        self.local = np.asarray(local, dtype=float)
        self.radii = np.asarray(radii, dtype=float)
        self.grouped = [
            (
                np.array([i for i, n in enumerate(links) if n == name]),
                self.model.getFrameId(
                    "panda_hand" if name=="attached_object" else name,pin.BODY
                ),
            )
            for name in sorted(set(links))
        ]
        ignore = collision["self_collision_ignore"]
        padding = collision["self_collision_buffer"]
        pairs = [
            (i, j)
            for i in range(len(links))
            for j in range(i + 1, len(links))
            if links[i] != links[j]
            and links[j] not in ignore.get(links[i], [])
            and links[i] not in ignore.get(links[j], [])
        ]
        if not pairs:
            raise ValueError("no robot self collision pairs")
        self.ia, self.ib = np.asarray(pairs, dtype=int).T
        self.self_limits = (
            self.radii[self.ia]
            + self.radii[self.ib]
            + np.asarray(
                [padding.get(links[i], 0) + padding.get(links[j], 0) for i, j in pairs]
            )
        )
        scene_path = ROOT / "curobo/content/configs/scene" / (scene or "")
        self.scene = yaml.safe_load(scene_path.read_text()) if scene else {}
        self.cuboids = []
        for name, box in self.scene.get("cuboid", {}).items():
            pose = box["pose"]
            rot = Rotation.from_quat([pose[4], pose[5], pose[6], pose[3]])
            self.cuboids.append(
                (name, np.asarray(pose[:3]), rot, np.asarray(box["dims"]) / 2)
            )
        self.spheres = [
            (
                name,
                np.asarray(s.get("position", s.get("pose", [0, 0, 0])[:3])),
                float(s["radius"]),
            )
            for name, s in self.scene.get("sphere", {}).items()
        ]
        self.capsules = []
        for name, cap in self.scene.get("capsule", {}).items():
            pose = cap["pose"]
            rot = Rotation.from_quat([pose[4], pose[5], pose[6], pose[3]])
            base = rot.apply(cap["base"]) + pose[:3]
            tip = rot.apply(cap["tip"]) + pose[:3]
            self.capsules.append((name, base, tip, float(cap["radius"])))
        baseline = (
            ROOT
            / "research/joint_time_oracle/results/native_reconstruction_20261106"
            / "baseline_validation_states.npz"
        )
        with np.load(baseline) as f:
            self.position_bounds = np.asarray(f["position_bounds"])
            self.nominal_effort = np.asarray(f["torque_bounds"])

    def gravity(self, q: np.ndarray) -> np.ndarray:
        values = np.asarray(q, dtype=float)
        if values.ndim == 1:
            values = values[None]
        if (
            values.ndim != 2
            or values.shape[1] != len(self.qi)
            or not np.isfinite(values).all()
        ):
            raise ValueError("invalid configuration array")
        tau = np.empty_like(values)
        for k, item in enumerate(values):
            self.q[self.qi] = item
            tau[k] = pin.rnea(self.model, self.data, self.q, self.zero, self.zero)[
                self.vi
            ]
            self.rnea_calls += 1
        return tau

    def check(self, waypoints: np.ndarray, *, min_clearance_m: float = 0.005) -> dict:
        q = np.asarray(waypoints, dtype=float)
        if q.ndim != 2 or q.shape[1] != 7 or not np.isfinite(q).all():
            raise ValueError("invalid robot path array")
        lo, hi = self.position_bounds
        violation = float(max(np.max(lo - q), np.max(q - hi), 0))
        if violation > 0:
            return {
                "position_violation_rad": violation,
                "world_clearance_m": None,
                "self_clearance_m": None,
                "sampled_acceptable": False,
                "gravity_torques": None,
            }
        world = float("inf")
        self_clear = float("inf")
        for item in q:
            self.q[self.qi] = item
            pin.framesForwardKinematics(self.model, self.data, self.q)
            xyz = np.empty_like(self.local)
            for ids, frame_id in self.grouped:
                pose = self.data.oMf[frame_id]
                xyz[ids] = self.local[ids] @ pose.rotation.T + pose.translation
            separation = (
                np.linalg.norm(xyz[self.ia] - xyz[self.ib], axis=1) - self.self_limits
            )
            self_clear = min(self_clear, float(np.min(separation)))
            for _, center, rotation, half in self.cuboids:
                delta = np.abs(rotation.inv().apply(xyz - center)) - half
                distance = (
                    np.linalg.norm(np.maximum(delta, 0), axis=1)
                    + np.minimum(np.max(delta, axis=1), 0)
                    - self.radii
                )
                world = min(world, float(np.min(distance)))
            for _, center, radius in self.spheres:
                world = min(
                    world,
                    float(
                        np.min(
                            np.linalg.norm(xyz - center, axis=1) - self.radii - radius
                        )
                    ),
                )
            for _, start, end, radius in self.capsules:
                axis = end - start
                length = float(axis @ axis)
                if length <= 1e-14:
                    raise ValueError("zero-length obstacle capsule")
                fraction = np.clip((xyz - start) @ axis / length, 0, 1)
                point = start + fraction[:, None] * axis
                world = min(
                    world,
                    float(
                        np.min(
                            np.linalg.norm(xyz - point, axis=1) - self.radii - radius
                        )
                    ),
                )
        collision_free = bool(
            world >= min_clearance_m and self_clear >= min_clearance_m
        )
        return {
            "position_violation_rad": violation,
            "world_clearance_m": None if np.isinf(world) else world,
            "self_clearance_m": self_clear,
            "sampled_acceptable": collision_free,
            "gravity_torques": self.gravity(q),
            "note": (
                "sampled independent Pinocchio spheres and scene primitives; "
                "NOT continuous certification"
            ),
        }
