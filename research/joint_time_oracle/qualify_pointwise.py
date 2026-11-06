# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Central-difference sweeps of every pointwise timing Jacobian entry."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .fixed_path_pointwise import PointwiseTiming


def main() -> None:
    """Persist entrywise errors for native and optimized controls over three step sizes."""
    out = Path(__file__).resolve().parent/'results/native_reconstruction_20261106'
    p = PointwiseTiming(out)
    candidates = {'native': p.original, 'optimized': np.load(out/'pointwise_ipopt_restored_stride1.npz')['controls']}
    summary = {}
    for name, controls in candidates.items():
        x=controls/p.scale; analytic=p.values_jac(x)[1]; grad=p.gradient(x)
        errors=[]; objective_errors=[]
        for h in (1e-4,1e-5,1e-6):
            jac=np.empty_like(analytic); obj=np.empty_like(grad)
            for i in range(32):
                step=np.zeros(32); step[i]=h
                jac[:,i]=(p.values_jac(x+step)[0]-p.values_jac(x-step)[0])/(2*h)
                obj[i]=(p.objective(x+step)-p.objective(x-step))/(2*h)
            errors.append(np.abs(jac-analytic)); objective_errors.append(np.abs(obj-grad))
        errors=np.array(errors); objective_errors=np.array(objective_errors)
        best=np.min(errors,axis=0)
        summary[name]={'shape':list(analytic.shape),'steps':[1e-4,1e-5,1e-6],
                       'max_best_absolute_error':float(best.max()),
                       'max_objective_error_by_step':objective_errors.max(axis=1).tolist(),
                       'entries_over_1e-5_absolute':int(np.sum(best>1e-5)),
                       'note':'Errors are for smooth explicit pointwise timing constraints and analytical Pinocchio-derived torque coefficients; not CUDA collision Jacobians.'}
        np.savez_compressed(out/f'pointwise_jacobian_{name}.npz', analytic=analytic, absolute_errors=errors,
                            objective_analytic=grad, objective_absolute_errors=objective_errors)
    (out/'pointwise_jacobian.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
