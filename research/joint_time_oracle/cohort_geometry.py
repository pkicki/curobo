# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact geometric path adapters for native cubic and joint-oracle quintic splines."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from .fixed_path_control import path_at


class NativeGeometry:
    """Evaluate a captured native cubic path without fitting or changing geometry."""

    def __init__(self, buffers: dict, lane: int) -> None:
        self.buffers = buffers
        self.lane = lane
        self.spans = buffers['controls'].shape[1] + 3

    def evaluate(self, progress: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return position and first three derivatives in native path coordinates."""
        s=np.asarray(progress)
        if np.any((s<0)|(s>1)):
            raise ValueError('path progress outside [0, 1]')
        return path_at(s,self.buffers,self.lane)

    def nonstationary_interval(self, tolerance: float = 1e-10) -> tuple[float, float]:
        """Find leading and trailing exactly stationary cubic spans by derivative checks."""
        active=[]
        for i in range(self.spans):
            # First derivative is quadratic; three interior nodes identify a zero polynomial.
            samples=(i+np.array([.2113248654,.5,.7886751346]))/self.spans
            if np.max(np.abs(self.evaluate(samples)[1]))>tolerance:
                active.append(i)
        if not active:
            raise ValueError('entire path is stationary')
        return active[0]/self.spans,(active[-1]+1)/self.spans

    def evaluate_trimmed(self, progress: np.ndarray) -> tuple[np.ndarray, ...]:
        """Remove only constant end spans and apply the chain rule for the new coordinate."""
        start,end=self.nonstationary_interval(); scale=end-start
        evaluated=self.evaluate(start+scale*np.asarray(progress))
        return tuple(value*scale**derivative for derivative,value in enumerate(evaluated))


class QuinticGeometry:
    """Evaluate the joint optimizer's original quintic controls and knots exactly."""

    def __init__(self, controls: np.ndarray) -> None:
        controls=np.asarray(controls)
        if controls.ndim!=2 or len(controls)<6:
            raise ValueError('quintic controls must have shape [at least 6, dof]')
        knots=np.r_[np.zeros(5),np.linspace(0,1,len(controls)-4),np.ones(5)]
        self.spline=BSpline(knots,controls,5,extrapolate=False)

    def evaluate(self, progress: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return original geometric spline values and first three path derivatives."""
        s=np.asarray(progress)
        if np.any((s<0)|(s>1)):
            raise ValueError('path progress outside [0, 1]')
        return tuple(self.spline(s,nu=i) for i in range(4))
