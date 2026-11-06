# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Densely sample the independently optimized squared-speed timing law."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
from scipy.interpolate import BSpline
from .fixed_path_control import path_at


def main() -> None:
    root=Path(__file__).resolve().parents[2]
    out=root/'research/joint_time_oracle/results/native_reconstruction_20261106'
    buffers=np.load(out/'native_buffers.npz'); lane=json.loads((out/'reconstruction.json').read_text())['selected_lane_by_position_match']
    attempt=np.load(out/'fixed_path_attempt.npz'); timing=attempt['solution_timing'] * 0.997
    basis=BSpline(np.r_[np.zeros(3),np.linspace(0,1,30),np.ones(3)],timing,3)
    n=buffers['controls'].shape[1]; knots=np.arange(1,n+3)/(n+3)
    s=np.unique(np.r_[np.linspace(0,1,16385),knots-1e-8,knots,knots+1e-8])
    path=path_at(s,buffers,lane); x,x1,x2=[basis(s,nu=i) for i in range(3)]
    if np.any(x<=0): raise ValueError('nonpositive squared speed')
    rootx=np.sqrt(x)[:,None]
    q=path[0]; qd=path[1]*rootx
    qdd=path[2]*x[:,None]+.5*path[1]*x1[:,None]
    jerk=rootx*(path[3]*x[:,None]+1.5*path[2]*x1[:,None]+.5*path[1]*x2[:,None])
    nodes,weights=np.polynomial.legendre.leggauss(6)
    length=np.diff(s); sample=s[:-1,None]+length[:,None]*(nodes[None,:]+1)/2
    dt=length*np.sum(weights[None,:]/2/np.sqrt(basis(sample)),axis=1)
    times=np.r_[0,np.cumsum(dt)]
    np.savez_compressed(out/'fixed_dense.npz',time=times,progress=s,q=q,qd=qd,qdd=qdd,jerk=jerk)
    (out/'fixed_restoration.json').write_text(json.dumps({'squared_speed_scale':0.997,
        'duration_s':float(times[-1]),'samples':len(s),
        'note':'Uniform speed reduction after independent nonlinear solve; geometry unchanged.'}, indent=2)+'\n')
    print('samples',len(s),'duration',times[-1])


if __name__=='__main__':
    main()
