# SPDX-License-Identifier: Apache-2.0
"""Use the exact cuRobo planner RNEA, kinematics, scene and collision masks.

CUDA-only. Numerical feasibility here is NOT independent physics verification.
Do not modify planner payload or scene while an oracle solve is in progress.
"""
from __future__ import annotations

import numpy as np
import torch

from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.state.state_joint import JointState

from .oracle import JointLimits


class CuroboOracleModel:
    """Differentiable callable dynamics and collision callbacks."""

    def __init__(self, planner, *, clearance: float = 0.0):
        self.kinematics = planner.kinematics
        self.joint_names = list(planner.joint_names)
        self.device_cfg = self.kinematics.device_cfg
        self.device = self.device_cfg.device
        if self.device.type != "cuda":
            raise ValueError("native cuRobo RNEA/collision requires CUDA")
        # Reuse the SAME dynamics object, model gravity and payload as TrajOpt.
        self.dynamics = planner.trajopt_solver.transition_model.robot_dynamics
        if self.dynamics is None:
            raise RuntimeError("planner lacks RNEA dynamics; refusing model substitution")
        self.scene = planner.scene_collision_checker
        self.world_weight = torch.ones(1,device=self.device,dtype=torch.float32)
        self.activation = torch.tensor([clearance],device=self.device,dtype=torch.float32)
        self.buffer = None
        scfg = self.kinematics.config.self_collision_config
        if scfg is None:
            raise RuntimeError("self collision configuration required")
        self.self_cost = SelfCollisionCost(SelfCollisionCostCfg(
            weight=1.0,device_cfg=self.device_cfg,
            self_collision_kin_config=scfg))
        self.last_horizon = None

    def limits(self):
        src = self.kinematics.get_joint_limits()
        if list(src.joint_names) != self.joint_names:
            raise ValueError("active joint ordering differs")
        if src.effort is None:
            raise ValueError("torque limits required")
        d = {}
        for name,source in (("position","position"),("velocity","velocity"),
                            ("acceleration","acceleration"),("jerk","jerk"),
                            ("torque","effort")):
            d[name] = getattr(src,source).detach().cpu().numpy().astype(np.float64)
        return JointLimits(**d)

    def _state(self,q,qd=None,qdd=None):
        return JointState(position=q.unsqueeze(0),
                          velocity=qd.unsqueeze(0) if qd is not None else None,
                          acceleration=qdd.unsqueeze(0) if qdd is not None else None,
                          joint_names=self.joint_names)

    def torque(self,q,qd,qdd):
        if self.last_horizon != len(q):
            self.dynamics.setup_batch_size(1,len(q))
            self.last_horizon = len(q)
        return self.dynamics.compute_inverse_dynamics(self._state(q,qd,qdd))[0]

    def collision(self,q):
        state = self.kinematics.compute_kinematics(self._state(q))
        spheres = state.robot_spheres
        if spheres is None:
            raise RuntimeError("no collision spheres available")
        costs = []
        if self.scene is not None:
            if self.buffer is None:
                self.buffer = CollisionBuffer.from_shape(spheres.shape,self.device_cfg)
            else:
                self.buffer.resize(spheres.shape,self.device_cfg)
            costs.append(self.scene.get_sphere_distance(
                state,self.buffer,self.world_weight,self.activation,
                sum_distance=False).reshape(-1))
        self.self_cost.setup_batch_tensors(1,len(q))
        costs.append(self.self_cost.forward(spheres).reshape(-1))
        return torch.cat(costs)
