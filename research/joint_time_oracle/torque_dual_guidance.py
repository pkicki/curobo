# SPDX-License-Identifier: Apache-2.0
"""LP-dual geometric guidance for path-dependent torque feasibility.

For a fixed q(s) with x=sdot**2 >= 0, y=x'(s)/2 unconstrained:
    tau(s) = g(q) + m(q, q_s, q_ss) x + a(q, q_s) y.
The pointwise relaxed LP optimizes eps>=0 such that
    lo - eps*halfwidth <= tau <= hi + eps*halfwidth.

Its dual upper/lower multipliers provide the local directional derivative
of the optimal relaxation with respect to geometric controls:
    d eps / d theta = (lambda_upper-lambda_lower)^T
                       d[g+m*x+a*y]/d theta
at fixed optimal x,y (envelope theorem), assuming stable active set.
We calculate the latter Jacobian by central finite differences of RNEA at
ONE path point, rather than refitting/reoptimizing a time law per parameter.

This gradient is a heuristic near LP active-set changes; the TRUE LP
obstruction and independent collision check always decide acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .cohort_geometry import QuinticGeometry


@dataclass(frozen=True)
class PointLP:
    status: str
    expansion: float
    x: float
    y: float
    upper_dual: np.ndarray
    lower_dual: np.ndarray
    dual_bound: float
    duality_gap: float
    primal_residual: float
    dual_residual: float


def solve_point_lp(a, m, g, bounds: np.ndarray) -> PointLP:
    """Return primal and dual when numerically reliable; fail closed otherwise."""
    a, m, g = (np.asarray(v, dtype=float) for v in (a, m, g))
    lim = np.asarray(bounds, dtype=float)
    n = len(g)
    if (
        a.shape != (n,)
        or m.shape != (n,)
        or lim.shape != (2, n)
        or not np.isfinite(np.r_[a, m, g, lim.ravel()]).all()
        or not np.all(lim[0] < lim[1])
    ):
        raise ValueError("invalid torque LP coefficient or bounds")
    lo, hi = lim
    half = (hi - lo) / 2
    A = np.concatenate(
        (np.column_stack((m, a, -half)), np.column_stack((-m, -a, -half)))
    )
    b = np.r_[hi - g, g - lo]
    result = linprog(
        [0.0, 0.0, 1.0],
        A_ub=A,
        b_ub=b,
        bounds=((0.0, None), (None, None), (0.0, None)),
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
        },
    )
    zeros = np.zeros(n, dtype=float)
    if not result.success or result.x is None or result.ineqlin is None:
        return PointLP(
            "LP_FAILURE",
            np.nan,
            np.nan,
            np.nan,
            zeros,
            zeros,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
    lam = -np.asarray(result.ineqlin.marginals, dtype=float)
    reduced = np.array([0.0, 0.0, 1.0]) + A.T @ lam
    dual = float(-lam @ b)
    gap = float(result.fun - dual)
    pres = float(max(0.0, np.max(A @ result.x - b)))
    dres = float(max(0.0, -np.min(lam), -reduced[0], -reduced[2], abs(reduced[1])))
    status = (
        "SOLVED"
        if (abs(gap) <= 1e-6 and pres <= 1e-7 and dres <= 1e-7)
        else "DUAL_UNVERIFIED"
    )
    return PointLP(
        status,
        float(result.fun),
        float(result.x[0]),
        float(result.x[1]),
        lam[:n],
        lam[n:],
        dual,
        gap,
        pres,
        dres,
    )


def normalized_bottlenecks(
    coeff: np.ndarray, bounds: np.ndarray, *, top_k: int = 3
) -> list[dict]:
    """Return most obstructed interior path points; all LPs must solve."""
    coeff = np.asarray(coeff, dtype=float)
    if coeff.ndim != 3 or coeff.shape[0] != 3:
        raise ValueError("expected coefficients [a,m,g] x samples x joints")
    records = []
    for k in range(coeff.shape[1]):
        r = solve_point_lp(coeff[0, k], coeff[1, k], coeff[2, k], bounds)
        if r.status != "SOLVED":
            raise ValueError(f"torque LP invalid at progress sample {k}: {r.status}")
        records.append(
            {
                "index": k,
                "expansion": r.expansion,
                "dual_bound": r.dual_bound,
                "x": r.x,
                "y": r.y,
                "upper_dual": r.upper_dual,
                "lower_dual": r.lower_dual,
            }
        )
    indices = sorted(range(len(records)), key=lambda i: (-records[i]["expansion"], i))
    return [records[i] for i in indices[:top_k]]


class AlignedPointDynamics:
    """Independent model-aligned RNEA at one spline progress point."""

    def __init__(self):
        # Defer Pinocchio imports until an actual CPU research run. LP and
        # mathematical tests can run without Pinocchio.
        import pinocchio as pin

        from .cohort_retime import dynamics

        self.pin = pin
        self.model, self.data, self.neutral, self.qi, self.vi = dynamics()
        self.q = self.neutral.copy()
        self.v = np.zeros(self.model.nv)
        self.acc = np.zeros(self.model.nv)
        self.rnea_calls = 0

    def coefficients(self, geometry: QuinticGeometry, s: float):
        values = geometry.evaluate(np.array([s]))
        q, qs, qss = [np.asarray(z[0], dtype=float) for z in values[:3]]
        self.q[self.qi] = q
        self.v.fill(0.0)
        self.acc.fill(0.0)
        grav = self.pin.rnea(self.model, self.data, self.q, self.v, self.acc)[
            self.vi
        ].copy()
        self.acc[self.vi] = qs
        a = (
            self.pin.rnea(self.model, self.data, self.q, self.v, self.acc)[
                self.vi
            ].copy()
            - grav
        )
        self.v[self.vi] = qs
        self.acc[self.vi] = qss
        m = (
            self.pin.rnea(self.model, self.data, self.q, self.v, self.acc)[
                self.vi
            ].copy()
            - grav
        )
        self.rnea_calls += 3
        return a, m, grav

    def dual_gradient(
        self, controls: np.ndarray, s: float, lp: PointLP, *, step: float = 2e-4
    ) -> np.ndarray:
        """Finite-difference the fixed-primal, dual-weighted torque field."""
        if lp.status != "SOLVED" or lp.expansion <= 0:
            raise ValueError("gradient requires a positive certified local LP")
        controls = np.asarray(controls, dtype=float)
        n, dof = controls.shape
        if n < 8 or dof != len(lp.upper_dual):
            raise ValueError("bad quintic controls/LP joint count")
        if not 0 < s < 1 or step <= 0:
            raise ValueError("invalid interior progress or finite-difference step")
        signed = lp.upper_dual - lp.lower_dual
        grad = np.zeros_like(controls)
        # Only the n-6 interior controls may change. This preserves endpoint
        # position and derivatives and makes path proposals reproducible.
        for i in range(3, n - 3):
            for j in range(dof):
                perturbed = controls.copy()
                perturbed[i, j] += step
                a1, m1, g1 = self.coefficients(QuinticGeometry(perturbed), s)
                perturbed[i, j] -= 2 * step
                a0, m0, g0 = self.coefficients(QuinticGeometry(perturbed), s)
                difference = ((g1 - g0) + lp.x * (m1 - m0) + lp.y * (a1 - a0)) / (
                    2 * step
                )
                grad[i, j] = float(signed @ difference)
        return grad


def normalized_descent_direction(
    gradient: np.ndarray, *, max_norm: float = 1e5
) -> np.ndarray | None:
    grad = np.asarray(gradient, dtype=float)
    if not np.isfinite(grad).all():
        return None
    n = np.linalg.norm(grad)
    if n < 1e-10 or n > max_norm:
        return None
    direction = -grad / n
    direction[:3] = 0
    direction[-3:] = 0
    return direction
