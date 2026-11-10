# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native cuRoboV2-Fast C-space planning with a rigid payload (CUDA env).

One ``MotionPlanner`` per scene, constructed once, warmed up, and reused for all
payloads and tasks (payload changes via ``update_link_inertial`` and in-place
sphere updates; both keep CUDA-graph buffer addresses). Production defaults
(``plan_cspace`` with its default attempt/graph policy, ``use_cuda_graph=True``)
are NOT changed.

A "restart" is one complete ``plan_cspace`` call; ``K`` restarts call it ``K``
times with ``reset_seed()`` in between and keep the first success (the
"native budget"). Every TimeOpt solve is snapshotted (all lanes), so native
failures can be analysed on the exact native cubic geometry (``seed_diagnosis``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .common import JOINT_NAMES, SCENES, Payload

DEFAULT_ATTACHED = {"mass": 0.01, "com": (0.0, 0.0, 0.0), "inertia": (1e-4, 1e-4, 1e-4, 0.0, 0.0, 0.0)}


@dataclass
class NativeAttempt:
    """Result of one ``plan_cspace`` call (one restart)."""

    success: bool
    duration_s: float | None
    wall_s: float
    trajectory: dict | None = None  # dense samples of the selected lane
    timeopt_snapshots: list = field(default_factory=list)  # per TrajOpt solve
    failure_stage: str | None = None
    native_reported_success: bool = False  # True if native said success (even if rejected here)


# Planner variants (research only; "native" = production defaults).
#   torque_aware: RobotCfg(load_dynamics=True), so TrajOpt rollouts compute inverse
#                 dynamics and TrajOpt's existing cspace torque-bound cost becomes
#                 active during geometric optimization (franka.yml defaults leave
#                 load_dynamics=False: TrajOpt is torque-blind; only TimeOpt sees torque).
#   gravity_aware_<rho>: as torque_aware, but the OPTIMIZER rollouts of TrajOpt see
#                 only gravity torque / rho (velocity and acceleration zeroed), so the
#                 existing torque-bound cost keeps |g(q)| <= rho * tau_max along the path
#                 and leaves speed to TimeOpt. Metrics rollouts (final validity) keep the
#                 true torque at the actual timing.
VARIANTS = ("native", "torque_aware", "gravity_aware_85", "gravity_aware_90", "gravity_aware_95")


class GravityOnlyDynamics:
    """Proxy for a cuRobo ``Dynamics``: compute_inverse_dynamics -> gravity torque / rho."""

    def __init__(self, dynamics, rho: float) -> None:
        self._d = dynamics
        self._rho = float(rho)

    def __getattr__(self, name):
        return getattr(self._d, name)

    def compute_inverse_dynamics(self, state):
        from curobo.types import JointState

        z = state.position * 0.0
        js = JointState(position=state.position, velocity=z, acceleration=z, jerk=z,
                        joint_names=state.joint_names)
        return self._d.compute_inverse_dynamics(js) / self._rho


def install_gravity_cost(planner, rho: float) -> int:
    """Swap the optimizer rollouts' dynamics of TrajOpt for the gravity-only proxy."""
    core = planner.trajopt_solver.core
    n = 0
    for rollout in core.optimizer_rollouts:
        tm = rollout.transition_model
        if tm.robot_dynamics is not None and not isinstance(tm.robot_dynamics, GravityOnlyDynamics):
            tm.robot_dynamics = GravityOnlyDynamics(tm.robot_dynamics, rho)
            n += 1
    if n == 0:
        raise RuntimeError("no TrajOpt optimizer rollout with dynamics; load_dynamics missing?")
    return n


def robot_config_for(variant: str):
    if variant == "native":
        return "franka.yml"
    from curobo._src.util.config_io import join_path, resolve_config
    from curobo.content import get_robot_configs_path

    data = resolve_config(join_path(get_robot_configs_path(), "franka.yml"))
    data["robot_cfg"]["load_dynamics"] = True
    return data


def _timeopt_stage(planner):
    solver = planner.trajopt_solver
    for owner in (solver, getattr(solver, "core", None)):
        stage = getattr(owner, "_timeopt", None) if owner is not None else None
        if stage is not None:
            return stage
    return None


class NativePlanner:
    """Reusable native planner for one scene."""

    def __init__(self, scene: str, *, seed: int = 0, use_cuda_graph: bool = True,
                 warmup_iterations: int = 3, capture: bool = True, variant: str = "native") -> None:
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

        if scene not in SCENES:
            raise ValueError(f"unknown scene {scene}")
        if variant not in VARIANTS:
            raise ValueError(f"unknown planner variant {variant}")
        self.scene = scene
        self.variant = variant
        self.planner = MotionPlanner(MotionPlannerCfg.create(
            robot=robot_config_for(variant), scene_model=SCENES[scene],
            collision_cache={"cuboid": 32, "mesh": 2},
            graph_planner_algorithm="rrt", use_cuda_graph=use_cuda_graph, random_seed=seed,
        ))
        self.kp = self.planner.kinematics.config.kinematics_config
        if variant.startswith("gravity_aware_"):
            self.gravity_rollouts = install_gravity_cost(self.planner, int(variant.split("_")[-1]) / 100.0)
        self.payload = Payload(0.0, (0.0, 0.0, 0.0))
        t = time.perf_counter()
        self.planner.warmup(enable_graph=True, num_warmup_iterations=warmup_iterations)
        torch.cuda.synchronize()
        self.warmup_s = time.perf_counter() - t
        effort = self.kp.joint_limits.effort.detach().cpu().numpy()
        self.original_effort = effort.copy()
        self.position_limits = self.kp.joint_limits.position.detach().cpu().numpy()
        self.capture = capture
        self._payload_dynamics = None
        self._replay = None
        self._snapshots: list = []
        if capture:
            self._wrap_solve()

    # ------------------------------------------------------------- payload
    def _update_inertial(self, **kw) -> None:
        """Update the attached_object inertial model shared by all planner stages.

        ``MotionPlanner.update_link_inertial`` raises for the default Franka planner
        (observed at 299d7a6) because some IK/TrajOpt rollouts have no dynamics
        model. All cuRobo ``Dynamics`` objects built on the same KinematicsParams
        share its cached reduced inertial model (``kp._dynamics_model``), which
        ``Dynamics.update_link_inertial`` refreshes in place (CUDA-graph safe). We
        therefore update through one Dynamics bound to the planner's KinematicsParams
        and VERIFY with ``gravity_all_models`` that TimeOpt sees the payload.
        """
        if self._payload_dynamics is None:
            from curobo._src.robot.dynamics.dynamics import Dynamics
            from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg

            self._payload_dynamics = Dynamics(DynamicsCfg(
                kinematics_config=self.kp, device_cfg=self.planner.kinematics.config.device_cfg,
                gravity=[0.0, 0.0, -9.81]))
        self._payload_dynamics.update_link_inertial("attached_object", **kw)

    def set_payload(self, payload: Payload) -> None:
        """Install mass/COM/inertia on attached_object and the payload collision sphere."""
        p = payload.validate()
        dev = self.kp.link_spheres.device
        if p.mass_kg > 0:
            inertia = np.r_[p.inertia_diag, 0.0, 0.0, 0.0]
            self._update_inertial(
                mass=float(p.mass_kg),
                com=torch.as_tensor(p.com_m, dtype=torch.float32),
                inertia=torch.as_tensor(inertia, dtype=torch.float32),
            )
            n = self.kp.get_number_of_spheres("attached_object")
            spheres = torch.zeros((n, 4), device=dev, dtype=self.kp.link_spheres.dtype)
            spheres[:, 3] = -100.0  # disabled slots
            spheres[0] = torch.as_tensor([*p.com_m, p.radius_m], device=dev)
            self.kp.update_link_spheres("attached_object", spheres)
        else:
            d = DEFAULT_ATTACHED
            self._update_inertial(
                mass=d["mass"], com=torch.as_tensor(d["com"], dtype=torch.float32),
                inertia=torch.as_tensor(d["inertia"], dtype=torch.float32),
            )
            n = self.kp.get_number_of_spheres("attached_object")
            spheres = torch.zeros((n, 4), device=dev, dtype=self.kp.link_spheres.dtype)
            spheres[:, 3] = -100.0
            self.kp.update_link_spheres("attached_object", spheres)
        torch.cuda.synchronize()
        self.planner.reset_seed()
        self.payload = p
        if not np.array_equal(self.kp.joint_limits.effort.detach().cpu().numpy(), self.original_effort):
            raise RuntimeError("original actuator effort limits were modified")

    def gravity_all_models(self, q: np.ndarray) -> dict:
        """Gravity torque from the trajopt dynamics and from the TimeOpt dynamics (parity)."""
        from curobo.types import JointState

        qt = torch.as_tensor(np.asarray(q, float), device=self.kp.link_spheres.device, dtype=torch.float32)
        z = torch.zeros_like(qt)
        out = {}
        dyn = self.planner.trajopt_solver.transition_model.robot_dynamics if hasattr(
            self.planner.trajopt_solver, "transition_model") else None
        stage = _timeopt_stage(self.planner)
        for name, d in (("trajopt", dyn), ("timeopt", None if stage is None else stage.optimizer._dynamics)):
            if d is None:
                continue
            d.setup_batch_size(qt.shape[0])
            js = JointState(position=qt, velocity=z, acceleration=z, joint_names=list(JOINT_NAMES))
            out[name] = d.compute_inverse_dynamics(js).detach().cpu().numpy()[:, :7]
        return out

    def sphere_world_distance(self, q: np.ndarray) -> np.ndarray:
        """Signed native world distance per configuration (min over all robot spheres)."""
        from curobo.types import JointState

        qt = torch.as_tensor(np.asarray(q, float), device=self.kp.link_spheres.device, dtype=torch.float32)
        js = JointState.from_position(qt, joint_names=list(JOINT_NAMES))
        state = self.planner.kinematics.compute_kinematics(js)
        spheres = state.robot_spheres  # [B, S, 4] (x, y, z, r); r < 0 disabled
        return spheres.detach().cpu().numpy()

    # ------------------------------------------------------------- capture
    def _wrap_solve(self) -> None:
        solver = self.planner.trajopt_solver
        original = solver.solve_cspace
        runner = self

        def solve_cspace(*args, **kwargs):
            result = original(*args, **kwargs)
            runner._snapshot()
            return result

        solver.solve_cspace = solve_cspace

    def _snapshot(self) -> None:
        stage = _timeopt_stage(self.planner)
        if stage is None:
            return
        opt = stage.optimizer
        solver = getattr(opt, "_solver", None)
        snap = {}
        for key, obj, attr in (
            ("controls", opt, "_controls"), ("start_boundary", opt, "_start_boundary"),
            ("goal", opt, "_goal"), ("success", opt, "_success"), ("status", opt, "_status"),
            ("times_end", opt, "_times"), ("input_valid", opt, "_input_valid"),
            ("optimization_ok", opt, "_optimization_ok"), ("stationary", opt, "_stationary"),
            ("seed_feasible", solver, "feasible"), ("seed_bounds", solver, "seed_bounds"),
            ("uniform_duration", opt, "_uniform_duration"),
        ):
            v = getattr(obj, attr, None) if obj is not None else None
            if isinstance(v, torch.Tensor):
                a = v.detach().cpu().numpy().copy()
                snap[key] = a[:, -1] if key == "times_end" else a
        masks = getattr(stage, "validation_masks", None)
        if isinstance(masks, dict):
            for k, v in masks.items():
                snap["mask_" + k] = v.detach().cpu().numpy().reshape(-1).copy()
        self._snapshots.append(snap)

    # ---------------------------------------------------------------- plan
    def plan(self, start: np.ndarray, goal: np.ndarray, *, restarts: int = 1,
             dense_samples: int = 2049, max_attempts: int | None = None) -> list[NativeAttempt]:
        """Up to ``restarts`` complete plan_cspace calls; stops at the first success."""
        from curobo.types import JointState

        dev = self.kp.link_spheres.device
        s = JointState.from_position(torch.as_tensor(start, device=dev, dtype=torch.float32)[None],
                                     joint_names=list(JOINT_NAMES))
        g = JointState.from_position(torch.as_tensor(goal, device=dev, dtype=torch.float32)[None],
                                     joint_names=list(JOINT_NAMES))
        attempts = []
        for _ in range(restarts):
            self._snapshots = []
            torch.cuda.synchronize()
            t = time.perf_counter()
            result = (self.planner.plan_cspace(g, s) if max_attempts is None
                      else self.planner.plan_cspace(g, s, max_attempts=max_attempts))
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
            ok = result is not None and bool(result.success.any())
            att = NativeAttempt(ok, None, wall, timeopt_snapshots=self._snapshots)
            if ok:
                traj = result.get_interpolated_plan()
                dt = float(traj.dt.reshape(-1)[0].item())
                pos = traj.position.reshape(-1, traj.position.shape[-1])[:, :7]
                finite = bool(np.isfinite(dt) and dt > 0 and torch.isfinite(pos).all().item())
                if finite:
                    att.duration_s = (pos.shape[0] - 1) * dt
                    att.trajectory = self._dense_selected(result, pos.detach().cpu().numpy(), dt, dense_samples)
                else:
                    # Observed at 299d7a6 with 4-5 kg payloads: success flag set, NaN trajectory.
                    att.success = ok = False
                    att.native_reported_success = True
                    att.failure_stage = "success_flag_with_nonfinite_trajectory"
            else:
                att.failure_stage = self._failure_stage()
            attempts.append(att)
            if ok:
                break
            self.planner.reset_seed()
        return attempts

    def _failure_stage(self) -> str:
        if not self._snapshots:
            return "no_trajopt_solve"
        last = self._snapshots[-1]
        if "seed_feasible" in last and not np.any(last["seed_feasible"]):
            return "timeopt_seed_infeasible_all_lanes"
        if "success" in last and not np.any(last["success"]):
            return "timeopt_failed_all_lanes"
        if "mask_collision_valid" in last and not np.any(last["mask_collision_valid"]):
            return "collision_all_lanes"
        return "other_validation"

    def _dense_selected(self, result, positions: np.ndarray, dt: float, n: int) -> dict:
        """Dense (t,q,qd,qdd,jerk) of the selected native lane by exact TimeOpt replay.

        Falls back to the interpolated trajectory if the replay lane cannot be
        identified (recorded in ``source``).
        """
        traj = result.get_interpolated_plan()
        fallback = {
            "time": np.arange(positions.shape[0]) * dt, "q": positions,
            "qd": traj.velocity.reshape(-1, traj.velocity.shape[-1])[:, :7].detach().cpu().numpy(),
            "qdd": traj.acceleration.reshape(-1, traj.acceleration.shape[-1])[:, :7].detach().cpu().numpy(),
            "jerk": traj.jerk.reshape(-1, traj.jerk.shape[-1])[:, :7].detach().cpu().numpy(),
            "source": "interpolated",
        }
        stage = _timeopt_stage(self.planner)
        if stage is None or not self._snapshots:
            return fallback
        try:
            replay = self._replica(stage.optimizer)
            lanes = replay.batch_size
            ends = replay._times[:, -1]
            T = (positions.shape[0] - 1) * dt
            grid = torch.linspace(0, T, positions.shape[0], device=ends.device)[None].repeat(lanes, 1)
            replay.reset_cuda_graph()
            probe = replay.sample(grid.contiguous()).position[..., :7].detach().cpu().numpy()
            err = np.max(np.abs(probe - positions[None]), axis=(1, 2))
            lane = int(np.argmin(err))
            if err[lane] > 1e-4:
                return fallback | {"replay_error": float(err[lane])}
            dur = float(ends[lane].item())
            tg = torch.linspace(0, dur, n, device=ends.device)[None].repeat(lanes, 1).contiguous()
            replay.reset_cuda_graph()
            st = replay.sample(tg)
            pick = lambda x: x[lane, :, :7].detach().cpu().numpy().copy()  # noqa: E731
            if not all(np.isfinite(pick(v)).all() for v in (st.position, st.velocity, st.acceleration, st.jerk)):
                return fallback | {"replay_nonfinite": True}
            return {"time": tg[lane].cpu().numpy(), "q": pick(st.position), "qd": pick(st.velocity),
                    "qdd": pick(st.acceleration), "jerk": pick(st.jerk), "source": "timeopt_replay",
                    "lane": lane, "replay_error": float(err[lane]), "duration_s": dur}
        except Exception as error:  # replay is diagnostic only
            return fallback | {"replay_exception": f"{type(error).__name__}: {error}"[:200]}

    def _replica(self, opt):
        """Eager TimeOpt copy holding the last solve's buffers (never touches the
        planner's captured graphs, so native latency is unaffected)."""
        from curobo._src.motion.motion_trajectory_timeopt import TrajectoryTimeOpt
        from curobo._src.motion.motion_trajectory_timeopt_cfg import TrajectoryTimeOptCfg

        if self._replay is None or self._replay.batch_size != opt.batch_size:
            cfg = TrajectoryTimeOptCfg(robot=self.planner.kinematics.config, batch_size=opt.batch_size,
                                       n_knots=opt._controls.shape[1], use_cuda_graph=False)
            self._replay = TrajectoryTimeOpt(cfg)
            self._replay._setup_path(reference=True)
        r = self._replay
        for attr in ("_controls", "_start_boundary", "_goal", "_timing", "_times", "_success", "_stationary"):
            getattr(r, attr).copy_(getattr(opt, attr))
        r._has_result = True
        return r

    def destroy(self) -> None:
        self.planner.destroy()


class HybridPlanner:
    """Native first; gravity-aware TrajOpt only when native TimeOpt rejects every seed.

    1. native ``plan_cspace(max_attempts=1)``; success -> return (route "native_first").
    2. failure with ``timeopt_seed_infeasible_all_lanes`` (or a NaN false success)
       -> gravity-aware planner with default attempts (route "gravity_fallback").
    3. any other failure -> native with default attempts (route "native_rest").
    Both planners are constructed, warmed and payload-updated once per scene.
    """

    TRIGGERS = ("timeopt_seed_infeasible_all_lanes", "success_flag_with_nonfinite_trajectory")

    def __init__(self, scene: str, *, seed: int = 0, fallback: str = "gravity_aware_85") -> None:
        self.native = NativePlanner(scene, seed=seed, variant="native")
        self.fallback = NativePlanner(scene, seed=seed, variant=fallback)
        self.warmup_s = self.native.warmup_s + self.fallback.warmup_s

    def set_payload(self, payload: Payload) -> None:
        self.native.set_payload(payload)
        self.fallback.set_payload(payload)

    def gravity_all_models(self, q):
        return self.native.gravity_all_models(q)

    def plan(self, start, goal, *, restarts: int = 1, dense_samples: int = 2049) -> list[NativeAttempt]:
        if restarts != 1:
            raise ValueError("hybrid planner is defined for one request")
        t = time.perf_counter()
        first = self.native.plan(start, goal, max_attempts=1, dense_samples=dense_samples)[-1]
        if first.success:
            route, out = "native_first", first
        elif first.failure_stage in self.TRIGGERS:
            route, out = "gravity_fallback", self.fallback.plan(start, goal, dense_samples=dense_samples)[-1]
        else:
            route, out = "native_rest", self.native.plan(start, goal, dense_samples=dense_samples)[-1]
        out.wall_s = time.perf_counter() - t
        out.failure_stage = None if out.success else f"{route}:{out.failure_stage}"
        out.route = route
        return [out]

    def destroy(self) -> None:
        self.native.destroy()
        self.fallback.destroy()
