# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reconstruct uniformly slowed adaptive samples without changing their geometry."""
from __future__ import annotations
import json
import numpy as np
from .cohort_retime import PILOT,RESULTS


def main() -> None:
    """Scale exact native and quintic adaptive states using stored restoration factors."""
    tasks=json.loads((PILOT/'tasks.json').read_text());factors=json.loads((RESULTS/'scaled_restoration.json').read_text())
    for task in tasks:
        case=task['case_id']
        for name in ('A10','A40','B','C','D'):
            src=RESULTS/case/f'{name.lower()}_adaptive_dense.npz';meta=factors.get(case+':'+name)
            if not src.exists() or not meta or meta['time_factor']<=1.:continue
            z=np.load(src);factor=meta['time_factor']
            np.savez_compressed(RESULTS/case/f'{name.lower()}_scaled_adaptive_dense.npz',time=z['time']*factor,q=z['q'],
                                qd=z['qd']/factor,qdd=z['qdd']/factor**2,jerk=z['jerk']/factor**3)


if __name__=='__main__':main()
