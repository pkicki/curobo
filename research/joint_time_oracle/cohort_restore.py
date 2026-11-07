# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep originals and produce separately sampled 95%-scaled timing restorations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pinocchio as pin
from .cohort_retime import PILOT, RESULTS, dynamics


def main() -> None:
    """Uniformly slow only candidates exceeding sampled 95% dynamic limits."""
    parser=argparse.ArgumentParser();parser.add_argument('--case')
    selected=parser.parse_args().case
    reference=np.load(Path(__file__).resolve().parent/'results/native_reconstruction_20261106/baseline_validation_states.npz')
    bounds={n:reference[n+'_bounds'] for n in ('velocity','acceleration','jerk','torque')}
    tasks=json.loads((PILOT/'tasks.json').read_text());model,data,q,qi,vi=dynamics()
    records={}
    for task in tasks:
        case=task['case_id']
        if selected and selected!=case:continue
        dest=RESULTS/case;dest.mkdir(parents=True,exist_ok=True)
        for method in ('A10','A40','B','C','D'):
            source=(PILOT if method in ('A10','A40') else RESULTS)/case/f'{method.lower()}_dense.npz'
            if not source.exists():continue
            archive=np.load(source);s={key:archive[key] for key in archive.files};factors=[1.]
            adaptive_path=dest/f'{method.lower()}_adaptive_dense.npz'
            adaptive=None
            if adaptive_path.exists():
                archive_adaptive=np.load(adaptive_path)
                adaptive={key:archive_adaptive[key] for key in archive_adaptive.files}
            for family,key,order in [('velocity','qd',1),('acceleration','qdd',2),('jerk','jerk',3)]:
                lo,hi=bounds[family];center=(lo+hi)/2;width=(hi-lo)*.475
                peak=float(np.max(np.abs((s[key]-center)/width)))
                if adaptive is not None:peak=max(peak,float(np.max(np.abs((adaptive[key]-center)/width))))
                factors.append(peak**(1/order))
            factor=max(factors)
            if factor>1:factor*=1.0001
            for attempt in range(12):
                qd=s['qd']/factor;qdd=s['qdd']/factor**2;jerk=s['jerk']/factor**3
                vel=np.zeros(model.nv);acc=np.zeros(model.nv);tau=np.empty_like(s['q'])
                for i in range(len(tau)):
                    q[qi]=s['q'][i];vel[vi]=qd[i];acc[vi]=qdd[i];tau[i]=pin.rnea(model,data,q,vel,acc)[vi]
                max_residual=-float('inf')
                for family,values in [('velocity',qd),('acceleration',qdd),('jerk',jerk),('torque',tau)]:
                    lo,hi=bounds[family];center=(lo+hi)/2;width=(hi-lo)*.475
                    max_residual=max(max_residual,float(np.max(np.abs((values-center)/width)-1)))
                if adaptive is not None:
                    av=adaptive['qd']/factor;aa=adaptive['qdd']/factor**2;aj=adaptive['jerk']/factor**3
                    at=np.empty_like(adaptive['q'])
                    for i in range(len(at)):
                        q[qi]=adaptive['q'][i];vel[vi]=av[i];acc[vi]=aa[i];at[i]=pin.rnea(model,data,q,vel,acc)[vi]
                    for family,values in [('velocity',av),('acceleration',aa),('jerk',aj),('torque',at)]:
                        lo,hi=bounds[family];center=(lo+hi)/2;width=(hi-lo)*.475
                        max_residual=max(max_residual,float(np.max(np.abs((values-center)/width)-1)))
                if max_residual<=1e-8:break
                factor*=1.005
            row={'time_factor':factor,'duration_s':float(s['time'][-1]*factor),'max_normalized_scaled_residual':max_residual,
                 'scaled_sampled_feasible':bool(max_residual<=1e-8),'iterations':attempt+1}
            records[case+':'+method]=row
            if factor>1.:
                np.savez_compressed(dest/f'{method.lower()}_scaled_dense.npz',time=s['time']*factor,q=s['q'],qd=qd,qdd=qdd,jerk=jerk)
            (RESULTS/'scaled_restoration.json').write_text(json.dumps(records,indent=2)+'\n')
        print(case,'restored',flush=True)


if __name__=='__main__':main()
