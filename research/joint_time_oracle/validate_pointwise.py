# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Densely verify an IPOPT fixed-path candidate using independent dynamics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
from .fixed_path_pointwise import PointwiseTiming


def main() -> None:
    """Save dense states and violations against physical and 95% dynamic limits."""
    parser = argparse.ArgumentParser()
    parser.add_argument('attempt')
    parser.add_argument('--scale', type=float, default=1.)
    parser.add_argument('--output', default='pointwise_dense')
    args = parser.parse_args()
    out = Path(__file__).resolve().parent / 'results/native_reconstruction_20261106'
    p = PointwiseTiming(out, grid_count=16385)
    controls = np.load(out / (args.attempt + '.npz'))['controls'] * args.scale
    basis = BSpline(np.r_[np.zeros(3), np.linspace(0, 1, 30), np.ones(3)], controls, 3)
    s = p.grid; path = p.path; x, dx, ddx = (basis(s, nu=i) for i in range(3))
    speed = np.sqrt(x)[:, None]
    q = path[0]; qd = path[1]*speed
    qdd = path[2]*x[:,None]+.5*path[1]*dx[:,None]
    jerk = speed*(path[3]*x[:,None]+1.5*path[2]*dx[:,None]+.5*path[1]*ddx[:,None])
    tau = p.coeff[2]+p.coeff[1]*x[:,None]+.5*p.coeff[0]*dx[:,None]
    nodes, weights = np.polynomial.legendre.leggauss(6)
    length = np.diff(s); sample = s[:-1,None]+length[:,None]*(nodes[None,:]+1)/2
    time = np.r_[0, np.cumsum(length*np.sum(weights[None,:]/2/np.sqrt(basis(sample)),axis=1))]
    result = {'attempt': args.attempt, 'squared_speed_scale': args.scale, 'duration_s': float(time[-1]), 'samples': len(s),
              'min_squared_speed': float(x.min()), 'families': {},
              'endpoint_residuals': {'q0': float(np.max(np.abs(q[0]-p.path[0,0]))),
                  'q1': float(np.max(np.abs(q[-1]-p.path[0,-1]))),
                  'v0': float(np.max(np.abs(qd[0]))), 'v1': float(np.max(np.abs(qd[-1]))),
                  'a0': float(np.max(np.abs(qdd[0]))), 'a1': float(np.max(np.abs(qdd[-1])))}}
    raw = np.load(out / 'baseline_validation_states.npz')
    for family, val in [('position',q),('velocity',qd),('acceleration',qdd),('jerk',jerk),('torque',tau)]:
        low, high = raw[family+'_bounds']; center=(low+high)/2
        result['families'][family] = {}
        for label, scale in [('physical',1.),('scaled_95', .95 if family != 'position' else 1.)]:
            lower=center+scale*(low-center); upper=center+scale*(high-center)
            residual=np.maximum(lower-val,val-upper); i,j=np.unravel_index(np.argmax(residual),residual.shape)
            result['families'][family][label]={'signed_residual':float(residual[i,j]),
                'violation':float(max(0,residual[i,j])),'joint':int(j+1), 'time_s':float(time[i]),
                'value':float(val[i,j]),'lower':float(lower[j]),'upper':float(upper[j])}
    np.savez_compressed(out/(args.output + '.npz'),time=time,progress=s,q=q,qd=qd,qdd=qdd,jerk=jerk,tau=tau)
    (out/(args.output + '.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
