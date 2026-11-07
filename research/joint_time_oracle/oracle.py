# SPDX-License-Identifier: Apache-2.0
"""Offline joint path and free-final-time trajectory optimization.

No cuRobo dependency in the mathematical core. This solves a local nonlinear
problem, NOT a certified global minimum. Dynamics/collision callbacks must
return differentiable torch tensors in original units.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
from scipy.interpolate import BSpline
from scipy.optimize import minimize


@dataclass(frozen=True)
class OracleConfig:
    controls: int = 18
    samples_per_span: int = 3
    audit_samples_per_span: int = 64
    maxiter: int = 250
    ftol: float = 2e-5
    attempts: int = 4
    seed_noise: float = 0.04
    trust_radius: float = 0.5
    constraint_tol: float = 2e-3
    min_duration: float = 0.05
    max_duration: float = 120.0
    limit_scale: float = 0.95
    rng_seed: int = 7
    solver: str = "slsqp"
    smooth_temperature: float = 0.002

    def __post_init__(self):
        if self.controls < 8 or self.samples_per_span < 2 or self.audit_samples_per_span < 2:
            raise ValueError("quintic spline requires >=8 controls and >=2 samples/span")
        if not 0 < self.limit_scale <= 1 or self.attempts < 1:
            raise ValueError("invalid safety scale or attempt count")
        if not 0 < self.min_duration < self.max_duration:
            raise ValueError("invalid duration interval")
        if self.constraint_tol < 0 or self.trust_radius <= 0:
            raise ValueError("invalid tolerance or trust region")
        if self.solver not in ("slsqp", "ipopt") or self.smooth_temperature < 0:
            raise ValueError("invalid NLP solver or smoothing parameter")


@dataclass(frozen=True)
class Boundary:
    q0: np.ndarray
    q1: np.ndarray
    v0: np.ndarray
    v1: np.ndarray
    a0: np.ndarray
    a1: np.ndarray

    @classmethod
    def rest(cls, q0, q1):
        p = np.asarray(q0, dtype=np.float64)
        f = np.asarray(q1, dtype=np.float64)
        if p.shape != f.shape or p.ndim != 1:
            raise ValueError("endpoints must be equal-size 1D vectors")
        return cls(p, f, np.zeros_like(p), np.zeros_like(f), np.zeros_like(p), np.zeros_like(f))

    def validate(self, dof):
        for name, val in vars(self).items():
            a = np.asarray(val)
            if a.shape != (dof,) or not np.isfinite(a).all():
                raise ValueError(f"invalid endpoint {name}")


@dataclass(frozen=True)
class JointLimits:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    torque: np.ndarray

    def validate(self, dof):
        for name, val in vars(self).items():
            a = np.asarray(val)
            if a.shape != (2, dof) or not np.isfinite(a).all() or np.any(a[0] >= a[1]):
                raise ValueError(f"{name} must be finite [2,dof], lower < upper")


class QuinticBasis:
    """Eliminate endpoint q, qdot, qddot exactly, leaving only interior controls."""

    def __init__(self, controls, points_per_span, *, device):
        self.controls = controls
        self.device = device
        self.nspans = controls - 5
        knots = np.r_[np.zeros(5), np.linspace(0, 1, controls - 4), np.ones(5)]
        self.spline = BSpline(knots, np.eye(controls), 5)
        points = [0.0, 1.0]
        for i in range(self.nspans):
            for j in range(points_per_span + 1):
                points.append((i + j / points_per_span) / self.nspans)
        # Include interior knots; spline has sufficient continuity for jerk.
        self.points = np.unique(np.asarray(points))
        self.B = [torch.as_tensor(self.spline(self.points, nu=d).copy(),
                                   device=device, dtype=torch.float64) for d in range(4)]
        self.boundary_indices = np.r_[np.arange(3), np.arange(controls - 3, controls)]
        self.interior_indices = np.arange(3, controls - 3)
        e = np.vstack([self.spline([0], nu=d)[0] for d in range(3)] +
                      [self.spline([1], nu=d)[0] for d in range(3)])
        self.endpoint_matrix = torch.as_tensor(
            e[:, self.boundary_indices].copy(), device=device, dtype=torch.float64
        )
        self.endpoint_int = torch.as_tensor(
            e[:, self.interior_indices].copy(), device=device, dtype=torch.float64
        )

    @property
    def variable_count(self):
        return len(self.interior_indices)

    def coefficients(self, interior, T, boundary):
        def vec(x):
            return torch.as_tensor(x, device=self.device, dtype=torch.float64)
        duration = T.to(dtype=torch.float64)
        rhs64 = torch.stack([vec(boundary.q0), duration * vec(boundary.v0),
                             duration*duration * vec(boundary.a0), vec(boundary.q1),
                             duration * vec(boundary.v1), duration*duration * vec(boundary.a1)])
        interior64 = interior.to(dtype=torch.float64)
        bc = torch.linalg.solve(self.endpoint_matrix,
                                rhs64 - self.endpoint_int @ interior64)
        return torch.cat([bc[:3], interior64, bc[3:]], dim=0)

    def evaluate(self, interior, T, boundary):
        c = self.coefficients(interior, T, boundary)
        duration = T.to(dtype=torch.float64)
        return (self.B[0]@c, (self.B[1]@c)/duration,
                (self.B[2]@c)/(duration*duration),
                (self.B[3]@c)/(duration*duration*duration), c)


def normalized_worst(values, bounds, scale, temperature=0.0, scale_about_zero=False):
    lo, hi = bounds[0], bounds[1]
    center = (hi+lo)*0.5
    lower = scale*lo if scale_about_zero else center + scale*(lo-center)
    upper = scale*hi if scale_about_zero else center + scale*(hi-center)
    width = torch.clamp((upper-lower)*0.5, min=1e-6)
    margins = torch.minimum((values-lower)/width, (upper-values)/width)
    if temperature > 0:
        # Conservative soft minimum over collocation samples.
        return -temperature * torch.logsumexp(-margins/temperature, dim=0)
    return margins.amin(dim=0)


class JointTimeOracle:
    """Local multi-start SLSQP optimizer; objective T, geometry and T both variable."""

    def __init__(self, limits: JointLimits, boundary: Boundary,
                 torque: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
                 collision: Callable[[torch.Tensor], torch.Tensor] | None = None,
                 *, config: OracleConfig = OracleConfig(), device: str = "cpu"):
        self.cfg = config
        self.limits = limits
        self.boundary = boundary
        self.torque = torque
        self.collision = collision
        self.dof = len(boundary.q0)
        boundary.validate(self.dof)
        limits.validate(self.dof)
        self.device = torch.device(device)
        self.basis = QuinticBasis(config.controls, config.samples_per_span, device=self.device)
        self.audit_basis = QuinticBasis(config.controls, config.audit_samples_per_span, device=self.device)
        self.bounds_t = {k:torch.as_tensor(v, device=self.device, dtype=torch.float32)
                         for k,v in vars(limits).items()}
        self._cache_x = self._cache_f = self._cache_j = None

    def _states(self, z, basis):
        interior = z[:basis.variable_count*self.dof].reshape(basis.variable_count, self.dof)
        T = z[-1].exp()
        return T, basis.evaluate(interior, T, self.boundary)

    def _margins(self, z, basis, *, audit=False):
        T, (q, qd, qdd, qddd, _) = self._states(z, basis)
        tau = self.torque(q.to(dtype=torch.float32), qd.to(dtype=torch.float32),
                          qdd.to(dtype=torch.float32))
        if tau.shape != q.shape:
            raise ValueError("torque callback must return [samples,dof]")
        families = (("position",q), ("velocity",qd), ("acceleration",qdd),
                    ("jerk",qddd), ("torque",tau))
        smooth = self.cfg.smooth_temperature if (self.cfg.solver=="ipopt" and not audit) else 0.0
        rows = [normalized_worst(value,self.bounds_t[name],
                                 1.0 if name=="position" else self.cfg.limit_scale,
                                 temperature=smooth, scale_about_zero=(name=="torque"))
                for name,value in families]
        if self.collision is not None:
            costs = self.collision(q.to(dtype=torch.float32)).reshape(-1)
            if not costs.numel():
                raise ValueError("collision callback must return at least one value")
            rows.append((z[-2]-costs.amax()).reshape(1))
        return torch.cat(rows)

    def _evaluate(self, x, jac):
        x = np.asarray(x, dtype=np.float64)
        key = x.tobytes()
        if key != self._cache_x or (jac and self._cache_j is None):
            z = torch.as_tensor(x, device=self.device, dtype=torch.float32).clone().requires_grad_(jac)
            m = self._margins(z, self.basis)
            f = m.detach().cpu().numpy().astype(np.float64)
            j = None
            if jac:
                grads = []
                for i in range(m.numel()):
                    g = torch.autograd.grad(m[i], z, retain_graph=i+1 < m.numel())[0]
                    grads.append(g.detach().cpu().numpy().astype(np.float64))
                j = np.stack(grads)
            self._cache_x, self._cache_f, self._cache_j = key, f, j
        return self._cache_j if jac else self._cache_f

    def _seed(self, q_samples, duration):
        q = np.asarray(q_samples, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != self.dof or q.shape[0] < 2:
            raise ValueError("seed positions must be [time,dof]")
        if not self.cfg.min_duration < duration < self.cfg.max_duration:
            raise ValueError("invalid initial duration")
        b = self.basis
        M = b.spline(np.linspace(0,1,len(q)))
        e = np.vstack([b.spline([0],nu=d)[0] for d in range(3)] +
                      [b.spline([1],nu=d)[0] for d in range(3)])
        inv = np.linalg.inv(e[:,b.boundary_indices])
        rhs = np.stack([self.boundary.q0,duration*self.boundary.v0,
                        duration**2*self.boundary.a0,self.boundary.q1,
                        duration*self.boundary.v1,duration**2*self.boundary.a1])
        H = M[:,b.interior_indices]-M[:,b.boundary_indices]@inv@e[:,b.interior_indices]
        off = M[:,b.boundary_indices]@inv@rhs
        interior = np.linalg.lstsq(H,q-off,rcond=None)[0]
        return np.r_[interior.reshape(-1), 0.0, np.log(duration)]

    def audit(self, x):
        with torch.no_grad():
            z = torch.as_tensor(x, device=self.device, dtype=torch.float32)
            T,(q,qd,qdd,qddd,c) = self._states(z, self.audit_basis)
            margins = self._margins(z, self.audit_basis, audit=True)
            minimum = float(margins.min().item())
            collision = float(self.collision(q).amax().item()) if self.collision is not None else 0.0
            slack = float(z[-2].item())
            violation = max(0.,-minimum,collision)
            return dict(duration=float(T.item()), max_violation=violation,
                        feasible=bool(torch.isfinite(margins).all() and
                                      minimum>=-self.cfg.constraint_tol and
                                      collision<=1e-5 and abs(slack)<=1e-5),
                        collision_cost=collision,collision_slack=slack,
                        min_margin=minimum, samples=len(q),
                        coefficients=c.cpu().numpy().tolist(),
                        positions=q.cpu().numpy().tolist(),
                        velocities=qd.cpu().numpy().tolist())

    def solve(self, seed_positions, duration):
        seed = self._seed(seed_positions, duration)
        rng = np.random.default_rng(self.cfg.rng_seed)
        n = self.basis.variable_count*self.dof
        bounds = [(float(v-self.cfg.trust_radius),float(v+self.cfg.trust_radius))
                  for v in seed[:-2]]
        bounds.append((0.0, 0.20))  # nonnegative collision slack (penalized in objective)
        bounds.append((np.log(self.cfg.min_duration),np.log(self.cfg.max_duration)))
        best, best_duration, trials = None, np.inf, []
        for attempt in range(self.cfg.attempts):
            self._cache_x = self._cache_f = self._cache_j = None
            x = seed.copy()
            if attempt:
                x[:n] += rng.normal(0,self.cfg.seed_noise,n)
                x[:n] = np.clip(x[:n],[v[0] for v in bounds[:n]],
                                [v[1] for v in bounds[:n]])
                x[-1] += np.log(1+0.12*attempt)
            if self.cfg.solver == "ipopt":
                from .ipopt_backend import solve_ipopt
                result = solve_ipopt(self,x,bounds)
            else:
                result = minimize(lambda z:float(np.exp(z[-1])+1000.0*z[-2]), x, method="SLSQP",
                                  jac=lambda z:np.r_[np.zeros(n),1000.0,np.exp(z[-1])],
                                  bounds=bounds,
                                  constraints=[dict(type="ineq",fun=lambda z:self._evaluate(z,False),
                                                    jac=lambda z:self._evaluate(z,True))],
                                  options=dict(maxiter=self.cfg.maxiter,ftol=self.cfg.ftol,disp=False))
            report = self.audit(result.x)
            trial = dict(attempt=attempt, solver_success=bool(result.success),
                               verified=report["feasible"],duration=report["duration"],
                               max_violation=report["max_violation"],message=str(result.message),
                               iterations=int(result.nit), nfev=int(result.nfev),
                               coefficients=report["coefficients"])
            if hasattr(result, "ipopt_info"):
                info = result.ipopt_info
                jac = self._evaluate(result.x, True)
                stationarity = (
                    np.r_[np.zeros(n), 1000.0, np.exp(result.x[-1])]
                    + jac.T @ info["mult_g"]
                    - info["mult_x_L"] + info["mult_x_U"]
                )
                trial.update({
                    "ipopt_status": int(info["status"]),
                    "ipopt_constraint_violation": float(
                        max(0.0, -np.min(self._evaluate(result.x, False)))
                    ),
                    "ipopt_dual_infeasibility": (
                        float(result.ipopt_history[-1][3])
                        if result.ipopt_history else None
                    ),
                    "ipopt_complementarity": (
                        float(info["compl_inf"]) if "compl_inf" in info else None
                    ),
                    "constraint_evaluations": int(result.constraint_evaluations),
                    "stationarity_inf": float(np.max(np.abs(stationarity))),
                    "iteration_history": result.ipopt_history,
                })
            trials.append(trial)
            if report["feasible"] and report["duration"] < best_duration:
                best,best_duration = result.x.copy(),report["duration"]
        return dict(best=self.audit(best) if best is not None else None,
                    trials=trials,config=asdict(self.cfg),
                    status="LOCAL_FEASIBLE" if best is not None else "NO_VERIFIED_CANDIDATE",
                    global_lower_bound=None)
