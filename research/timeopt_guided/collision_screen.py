# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched cuRobo world/self/joint-limit screening of candidate paths (CUDA env).

Candidates are checked on a dense, uniform path-parameter grid with the payload
sphere installed on ``attached_object``. This is the SEARCH-TIME screen only;
final qualification uses ``independent_validate`` (Pinocchio spheres + scene
primitives) on the retimed trajectory.
"""

from __future__ import annotations

import numpy as np
import torch

from .common import Payload, curobo_scene
from .geometry import evaluate


class PathCollisionChecker:
    """Reusable checker for one scene; payload sphere updated in place."""

    def __init__(self, scene: str, *, margin_m: float = 0.005, samples: int = 257) -> None:
        from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg

        cfg = RobotCollisionCheckerCfg.load_from_config(
            robot_config="franka.yml", scene_model=curobo_scene(scene),
            collision_activation_distance=margin_m, self_collision_activation_distance=margin_m,
            n_cuboids=32, n_meshes=2,
        )
        self.checker = RobotCollisionChecker(cfg)
        self.kp = self.checker.kinematics.config.kinematics_config
        self.samples = samples
        self.scene = scene
        self.margin_m = margin_m
        self.calls = 0
        # Warm up kernels so the first search arm is not charged for compilation.
        q = np.zeros((16, samples, 7))
        self.valid_configs(q)

    def set_payload(self, payload: Payload) -> None:
        p = payload.validate()
        n = self.kp.get_number_of_spheres("attached_object")
        dev = self.kp.link_spheres.device
        spheres = torch.zeros((n, 4), device=dev, dtype=self.kp.link_spheres.dtype)
        spheres[:, 3] = -100.0
        if p.mass_kg > 0:
            spheres[0] = torch.as_tensor([*p.com_m, p.radius_m], device=dev)
        self.kp.update_link_spheres("attached_object", spheres)

    def valid_configs(self, q: np.ndarray, *, strict_margin: bool = False) -> np.ndarray:
        """Boolean validity for configurations [B, H, 7] -> [B, H].

        ``strict_margin``: also reject configurations within ``margin_m`` of an obstacle
        or of a self-collision pair. ``RobotCollisionChecker.validate`` only rejects
        actual penetration; the activation-distance costs are positive inside the margin.
        """
        qt = torch.as_tensor(np.asarray(q, float), device=self.kp.link_spheres.device, dtype=torch.float32)
        if qt.ndim == 2:
            qt = qt[None]
        qt = qt.contiguous()
        valid = self.checker.validate(qt)
        if strict_margin:
            dw, ds = self.checker.get_scene_self_collision_distance_from_joints(qt)
            valid = valid & (dw.reshape(*qt.shape[:2], -1).sum(-1) <= 0) & (ds.reshape(*qt.shape[:2], -1).sum(-1) <= 0)
        return valid.detach().cpu().numpy()

    def check_paths(self, controls: list[np.ndarray], *, degree: int = 5, knots=None) -> np.ndarray:
        """True for each candidate whose sampled path is collision free and in joint limits."""
        self.calls += len(controls)
        s = np.linspace(0, 1, self.samples)
        q = np.stack([evaluate(c, s, degree=degree, knots=knots) for c in controls])
        return self.valid_configs(q).all(axis=1)
