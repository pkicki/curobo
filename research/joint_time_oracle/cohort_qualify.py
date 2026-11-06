# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entrywise finite-difference qualification of the trimmed native timing NLP."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .cohort_geometry import NativeGeometry
from .cohort_retime import PILOT, RESULTS, problem


def main() -> None:
    """Record all analytical entries and step-sweep absolute errors for one task."""
    case='pilot_free_00';buffers=np.load(PILOT/case/'a10_buffers.npz');geo=NativeGeometry(buffers,0)
    start,end=geo.nonstationary_interval();count=buffers['controls'].shape[1]
    knots=(np.arange(1,count+3)/(count+3)-start)/(end-start)
    initial=np.load(RESULTS/case/'b_grid257.npz')['controls'];p=problem(geo.evaluate_trimmed,knots,257,initial)
    x=initial/p.scale; analytic=p.values_jac(x)[1];grad=p.gradient(x); errors=[];objectives=[]
    for step in (1e-4,1e-5,1e-6):
        jac=np.empty_like(analytic);g=np.empty_like(x)
        for col in range(32):
            direction=np.zeros(32);direction[col]=step
            jac[:,col]=(p.values_jac(x+direction)[0]-p.values_jac(x-direction)[0])/(2*step)
            g[col]=(p.objective(x+direction)-p.objective(x-direction))/(2*step)
        errors.append(np.abs(jac-analytic));objectives.append(np.abs(g-grad))
    errors=np.asarray(errors);objectives=np.asarray(objectives);best=errors.min(axis=0)
    out=RESULTS/case;np.savez_compressed(out/'b_jacobian_sweep.npz',analytical=analytic,errors=errors,objective_errors=objectives)
    result={'shape':list(analytic.shape),'steps':[1e-4,1e-5,1e-6],
            'max_best_step_absolute_error':float(best.max()),'entries_over_1e-5':int(np.sum(best>1e-5)),
            'max_objective_error':float(objectives.min(axis=0).max()),'scope':'smooth fixed-path timing NLP; not CUDA collision derivatives'}
    (out/'b_jacobian_sweep.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
