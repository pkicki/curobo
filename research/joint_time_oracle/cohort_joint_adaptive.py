# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate exact quintic candidates near sampled physical constraint peaks."""
from __future__ import annotations
import json
import numpy as np
from .cohort_geometry import QuinticGeometry
from .cohort_retime import PILOT,RESULTS


def main() -> None:
    """Save quintic states at nearby peak times and spline knots."""
    tasks=json.loads((PILOT/'tasks.json').read_text());diagnostics=json.loads((RESULTS/'independent_original.json').read_text())
    for task in tasks:
        case=task['case_id'];path=RESULTS/case/'c_best.json';diag=diagnostics.get(case+':C')
        if not path.exists() or not diag:continue
        best=json.loads(path.read_text());geometry=QuinticGeometry(np.asarray(best['coefficients']));duration=best['duration']
        centers=list(np.unique(geometry.spline.t)*duration)
        for family in diag['families'].values():centers.extend([family['physical']['time_s'],family['scaled95']['time_s']])
        centers.extend(x for x in (diag['world_min_time_s'],diag['self_min_time_s']) if x is not None)
        step=duration/8192
        times=np.unique(np.clip(np.r_[(np.asarray(centers)[:,None]+np.array([-1.,-.25,0.,.25,1.])*step).ravel(),0.,duration],0,duration))
        states=geometry.evaluate(times/duration)
        np.savez_compressed(RESULTS/case/'c_adaptive_dense.npz',time=times,q=states[0],qd=states[1]/duration,
                            qdd=states[2]/duration**2,jerk=states[3]/duration**3)
        print(case,'joint adaptive states',flush=True)


if __name__=='__main__':main()
