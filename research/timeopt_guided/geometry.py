# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact path representations and geometric proposal operators (pure NumPy/SciPy).

* Native cuRobo TimeOpt paths are UNIFORM cubic B-splines over the support
  ``[start_boundary(4), controls[:-1], goal x3]``. ``native_cubic_to_clamped``
  converts one lane to an EXACTLY equal clamped cubic (knot insertion, no
  fitting) and trims constant end spans, so K107 can retime the very same curve.
* Search candidates are clamped-uniform quintic B-splines with 22 controls and
  rest endpoints (first/last three controls equal to start/goal).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

from .common import enforce_rest, quintic_knots


# --------------------------------------------------------------------------- native
def native_support(start_boundary: np.ndarray, controls: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Uniform cubic support points of one native lane ([n+6, dof])."""
    sb, c, g = (np.asarray(a, dtype=np.float64) for a in (start_boundary, controls, goal))
    if sb.ndim != 2 or sb.shape[0] != 4 or c.ndim != 2 or g.ndim != 1:
        raise ValueError("native lane must be start_boundary[4,d], controls[n,d], goal[d]")
    return np.concatenate([sb, c[:-1], np.tile(g, (3, 1))])


def native_uniform_spline(start_boundary, controls, goal) -> BSpline:
    """Native path as a SciPy BSpline on [0,1] (uniform knots, exact)."""
    support = native_support(start_boundary, controls, goal)
    spans = len(support) - 3
    knots = (np.arange(len(support) + 4) - 3) / spans
    return BSpline(knots, support, 3, extrapolate=False)


def nonstationary_interval(support: np.ndarray) -> tuple[int, int, int]:
    """(first, last+1, spans) of spans whose 4 support points are not all equal."""
    spans = len(support) - 3
    active = [i for i in range(spans) if not np.all(support[i : i + 4] == support[i])]
    if not active:
        raise ValueError("entire native path is stationary")
    return active[0], active[-1] + 1, spans


def native_cubic_to_clamped(start_boundary, controls, goal) -> dict:
    """Exact clamped cubic representation on the trimmed native interval.

    Returns dict(controls [m, d], knots [m+4], degree=3, s0, s1) where the native
    progress ``s_native = s0 + (s1 - s0) * u`` for the clamped parameter ``u``.
    """
    support = native_support(start_boundary, controls, goal)
    first, last, spans = nonstationary_interval(support)
    spl = native_uniform_spline(start_boundary, controls, goal)
    a, b = first / spans, last / spans
    work = BSpline(spl.t.copy(), spl.c.copy(), 3, extrapolate=False)
    for x in (a, b):
        # Raise the multiplicity of knot x to 3 (= degree), which clamps it.
        mult = int(np.sum(np.isclose(work.t, x, atol=1e-14, rtol=0)))
        if mult < 3:
            work = work.insert_knot(x, 3 - mult)
    t, c = work.t, np.asarray(work.c)
    # Keep the knots/controls supported on [a, b].
    ia = int(np.searchsorted(t, a - 1e-14))  # first index of x == a
    ib = int(np.searchsorted(t, b + 1e-14))  # one past last index of x == b
    knots = np.r_[a, t[ia:ib], b]
    n_ctrl = len(knots) - 4
    ctrl = c[ia - 1 : ia - 1 + n_ctrl]
    knots = (knots - a) / (b - a)
    knots[:4] = 0.0
    knots[-4:] = 1.0
    out = {"controls": np.ascontiguousarray(ctrl), "knots": knots, "degree": 3, "s0": a, "s1": b}
    # Exactness check against the original native spline.
    u = np.linspace(0, 1, 513)
    ref = spl(a + (b - a) * u)
    got = BSpline(knots, ctrl, 3)(u)
    err = float(np.max(np.abs(ref - got)))
    if not err < 1e-9:
        raise ValueError(f"native->clamped conversion not exact (max error {err:.3e})")
    out["conversion_max_error"] = err
    return out


# -------------------------------------------------------------------------- quintic
def quintic_spline(controls: np.ndarray) -> BSpline:
    c = np.asarray(controls, float)
    return BSpline(quintic_knots(len(c)), c, 5, extrapolate=False)


def evaluate(controls: np.ndarray, s: np.ndarray, *, degree: int = 5, knots=None, order: int = 0):
    c = np.asarray(controls, float)
    t = quintic_knots(len(c)) if knots is None else np.asarray(knots, float)
    spl = BSpline(t, c, degree, extrapolate=False)
    return spl(np.asarray(s, float), nu=order)


def fit_quintic(samples_s: np.ndarray, q: np.ndarray, n: int = 22, smooth: float = 1e-6) -> np.ndarray:
    """Least-squares rest-to-rest quintic fit to path samples (used only to SEED search).

    The fitted curve is a different geometry from the native one; results on it
    must never be reported as exact-native-path results.
    """
    s = np.asarray(samples_s, float)
    q = np.asarray(q, float)
    knots = quintic_knots(n)
    B = BSpline.design_matrix(np.clip(s, 0, 1), knots, 5).toarray()
    q0, q1 = q[0], q[-1]
    fixed = np.zeros((n, q.shape[1]))
    fixed[:3], fixed[-3:] = q0, q1
    free = slice(3, n - 3)
    A = B[:, free]
    rhs = q - B @ fixed
    reg = smooth * np.eye(A.shape[1])
    x = np.linalg.solve(A.T @ A + reg, A.T @ rhs)
    out = fixed.copy()
    out[free] = x
    return out


# ------------------------------------------------------------------------ proposals
def bump(n: int, center: float, width: float) -> np.ndarray:
    """Gaussian envelope over control index (in [0,1]) that is zero on rest controls."""
    idx = np.linspace(0, 1, n)
    env = np.exp(-0.5 * ((idx - center) / width) ** 2)
    env[:3] = 0.0
    env[-3:] = 0.0
    return env


def global_proposal(base: np.ndarray, rng: np.random.Generator, *, amplitude: float,
                    lower: np.ndarray, upper: np.ndarray, lobes: int | None = None) -> np.ndarray:
    """Broad multi-lobe random detour around ``base`` (joint-space, rest-preserving).

    A sum of 1-3 Gaussian bumps with random centers/widths and random joint-space
    directions; amplitude in radians. Clipped into the joint position limits.
    """
    c = np.asarray(base, float).copy()
    n, d = c.shape
    lobes = int(rng.integers(1, 4)) if lobes is None else lobes
    for _ in range(lobes):
        direction = rng.normal(size=d)
        direction /= max(np.linalg.norm(direction), 1e-12)
        center = float(rng.uniform(0.2, 0.8))
        width = float(rng.uniform(0.12, 0.35))
        c += amplitude * float(rng.uniform(0.5, 1.0)) * bump(n, center, width)[:, None] * direction[None]
    return enforce_rest(np.clip(c, lower, upper), base[0], base[-1])


def local_proposal(base: np.ndarray, rng: np.random.Generator, *, amplitude: float,
                   lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Small isotropic perturbation of every interior control (elite refinement)."""
    c = np.asarray(base, float).copy()
    n, d = c.shape
    noise = rng.normal(size=(n, d)) * amplitude
    smooth = np.convolve(np.ones(3) / 3, np.ones(1), mode="full")  # light smoothing kernel
    for j in range(d):
        noise[:, j] = np.convolve(noise[:, j], smooth, mode="same")
    noise[:3] = 0.0
    noise[-3:] = 0.0
    return enforce_rest(np.clip(c + noise, lower, upper), base[0], base[-1])


def directed_proposal(base: np.ndarray, rng: np.random.Generator, *, s_center: float,
                      direction: np.ndarray, amplitude: float, lower: np.ndarray,
                      upper: np.ndarray, width: float | None = None,
                      second_lobe: bool = True) -> np.ndarray:
    """Bottleneck-directed detour: move the path near ``s_center`` along ``direction``.

    ``direction`` is a joint-space descent direction (e.g. minus the gradient of the
    binding joint's gravity torque). A random orthogonal component and an optional
    second, broader lobe keep proposals diverse (local LP-gradient guidance alone
    failed in the 2026-11-08 study).
    """
    c = np.asarray(base, float).copy()
    n, d = c.shape
    v = np.asarray(direction, float)
    v = v / max(np.linalg.norm(v), 1e-12)
    noise = rng.normal(size=d)
    noise -= (noise @ v) * v
    noise /= max(np.linalg.norm(noise), 1e-12)
    mix = v + float(rng.uniform(0.0, 0.6)) * noise
    mix /= np.linalg.norm(mix)
    w = float(rng.uniform(0.08, 0.2)) if width is None else width
    # Control-index position of the path parameter is approximately s for clamped uniform knots.
    c += amplitude * float(rng.uniform(0.5, 1.2)) * bump(n, s_center, w)[:, None] * mix[None]
    if second_lobe:
        w2 = min(0.45, 2.5 * w)
        c += 0.5 * amplitude * float(rng.uniform(0.0, 1.0)) * bump(n, s_center, w2)[:, None] * mix[None]
    return enforce_rest(np.clip(c, lower, upper), base[0], base[-1])


def path_distance(a: np.ndarray, b: np.ndarray, n: int = 129) -> float:
    """Max joint-space distance between two quintic candidates at matched s."""
    s = np.linspace(0, 1, n)
    return float(np.max(np.linalg.norm(evaluate(a, s) - evaluate(b, s), axis=1)))
