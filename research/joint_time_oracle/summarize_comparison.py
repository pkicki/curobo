# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize saved reconstruction and sampled checks without claiming qualification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np


def main() -> None:
    root=Path(__file__).resolve().parents[2]
    out=root/'research/joint_time_oracle/results/native_reconstruction_20261106'
    dense=json.loads((out/'dense_validation.json').read_text())
    native=json.loads((out/'reconstruction.json').read_text())
    fixed=json.loads((out/'fixed_path_attempt.json').read_text())
    restore=json.loads((out/'fixed_restoration.json').read_text())
    pin=json.loads((out/'pinocchio_dense.json').read_text())
    jac=json.loads((out/'oracle_jacobian_sweep.json').read_text())['entries']
    cuda=json.loads((out/'cuda_derivative_sweep.json').read_text())
    sweep={}
    for step in (.01,.003,.001):
        entries=[e for e in jac if e['step']==step]
        unresolved=[e for e in entries if not e['possible_active_set_change'] and
                    e['abs_error']>.01+.02*abs(e['analytic'])]
        sweep[str(step)]={'entries':len(entries),'possible_active_set_switches':sum(e['possible_active_set_change'] for e in entries),
                          'unflagged_outside_tolerance':len(unresolved),'examples':unresolved[:5],
                          'max_absolute_error':max(e['abs_error'] for e in entries)}
    cases={}
    for name,key in [('baseline','baseline_16385'),('fixed_path','fixed_path_8193'),('candidate','candidate_16385')]:
        d=dense[key]; state=np.load(out/f'{name}_validation_states.npz')
        cases[name]={'duration_s':float(state['time'][-1]),'dense_samples':d['samples'],
            'path_length_rad':float(np.linalg.norm(np.diff(state['q'],axis=0),axis=1).sum()),
            'physical_max_violation':{f:d[f]['physical']['max_violation'] for f in ('position','velocity','acceleration','jerk','torque')},
            'scaled_95_max_violation':{f:d[f]['scaled_95']['max_violation'] for f in ('position','velocity','acceleration','jerk','torque')},
            'endpoint_residuals':d['endpoint_residuals'],'world_swept_max_cost':d['world_swept']['max_cost'],
            'self_discrete_max_cost':d['self_discrete']['max_cost'],
            'pinocchio_max_torque_difference_nm':pin[name]['max_difference_nm'],
            'pinocchio_physical_torque_violation_nm':pin[name]['physical']['max_violation_nm']}
    summary={'status':'BLOCKED', 'pilot':'BLOCKED', 'cases':cases,
             'reconstruction':native,'fixed_path_solver':fixed,'fixed_path_restoration':restore,
             'endpoint_acceptance':{'position_rad':1e-5,'velocity_rad_s':1e-4,'acceleration_rad_s2':1e-3,
                 'justification':'Conservative float32 evaluation tolerance; exact saved quintic coefficients give endpoint acceleration residual near 1.4e-12 rad/s². Physical trajectory limits are evaluated separately.'},
             'oracle_jacobian_sweep':sweep,'cuda_collision_cases':cuda['collision_cases'],
             'active_sample_diagnosis': json.loads((out/'active_sample_diagnosis.json').read_text()),
             'limitations':['No independently implemented world/self collision check; native swept-world and sampled self collision only.',
               'SLSQP fixed-path solver terminated at its iteration budget; result was uniformly slowed to restore the 95% sampled limits.',
               'Nonsmooth active-sample switches require careful handling; the representative step 0.003 discrepancy was traced to a switch from sample 8 to sample 7. Other flagged cases are not individually classified.',
               'Native baseline and joint candidate have nonzero violations of the 95% limits despite satisfying physical limits on sampled grids.',
               'No globally optimality or continuous-time feasibility certificate; B has not retimed the original quintic geometry.'],
             'note':'Durations are observed candidate durations, not verified optimums or a qualified improvement.'}
    (out/'qualification_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    paths=sorted(p for p in out.iterdir() if p.is_file() and p.name!='checksums.json')
    (out/'checksums.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    print(json.dumps({'status':summary['status'],'durations':{k:v['duration_s'] for k,v in cases.items()},
                      'jacobian_sweep':sweep},indent=2))


if __name__=='__main__':
    main()
