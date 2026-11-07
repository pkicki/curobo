# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Add exact-geometry samples around independent B/D torque extrema."""
from __future__ import annotations
import json
import numpy as np
import pinocchio as pin
from .cohort_retime import PILOT, RESULTS, dynamics, states
from .cohort_geometry import NativeGeometry, QuinticGeometry


def torque_samples(q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
    """Evaluate aligned Pinocchio actuator torques at seven-joint states."""
    model,data,state,qi,vi=dynamics();out=np.empty_like(q)
    velocity=np.zeros(model.nv);acceleration=np.zeros(model.nv)
    for i in range(len(q)):
        state[qi]=q[i];velocity[vi]=qd[i];acceleration[vi]=qdd[i]
        out[i]=pin.rnea(model,data,state,velocity,acceleration)[vi]
    return out


def main() -> None:
    """Save B/D trajectories augmented near torque maxima/minima for audit."""
    tasks=json.loads((PILOT/'tasks.json').read_text())
    for task in tasks:
        case=task['case_id'];folder=RESULTS/case
        for method in ('B','D'):
            source=folder/f'{method.lower()}_dense.npz'
            if not source.exists():continue
            saved=np.load(source);controls=saved['controls'];knots=saved['knots']
            if method=='B':
                row=next(json.loads(line) for line in (PILOT/'pilot.jsonl').read_text().splitlines()
                         if json.loads(line)['case_id']==case)
                geo=NativeGeometry(np.load(PILOT/case/'a10_buffers.npz'),row['methods']['A10']['lane'])
                fn=lambda s:geo.evaluate_trimmed(s)
            else:
                coeff=np.asarray(json.loads((folder/'c_best.json').read_text())['coefficients'])
                geo=QuinticGeometry(coeff);fn=geo.evaluate
            base=states(fn,controls,knots=knots)
            tau=torque_samples(base['q'],base['qd'],base['qdd'])
            # Add progressively smaller neighborhoods around both signed extrema.
            progress=base['progress'];extra=[]
            for joint in range(tau.shape[1]):
                for index in (int(np.argmin(tau[:,joint])),int(np.argmax(tau[:,joint]))):
                    for delta in (-2.,-1.,-.5,.5,1.,2.):
                        extra.append(np.clip(progress[index]+delta/8192,0,1))
            audited=states(fn,controls,knots=knots,extra=np.asarray(extra))
            np.savez_compressed(folder/f'{method.lower()}_adaptive_dense.npz',**audited,
                                controls=controls,knots=knots)
            print(case,method,'samples',len(audited['time']),flush=True)


if __name__=='__main__':main()
