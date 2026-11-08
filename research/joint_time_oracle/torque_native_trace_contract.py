# SPDX-License-Identifier: Apache-2.0
"""Pure-CPU contract shared by instrumented native replay and its independent audit."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

CASES={
    "trap_cluttered_08":("geometry_trap_source_20261108",
                        "geometry_feasibility_discovery_20261108"),
    "confirm_geom_table_07":("geometry_trap_confirm_20261109",
                             "geometry_feasibility_confirmation_20261109"),
    "confirm_geom_cluttered_02":("geometry_trap_confirm_20261109",
                                "geometry_feasibility_confirmation_20261109"),
}
SCENARIO="j1234_f0300"
OUTPUT=ROOT/"research/joint_time_oracle/results/torque_lp_native_stage_trace_20261108"


def cubic_geometry_sha(buffers:dict,lane:int)->str:
    """Hash exact cubic path controls and boundary fields with native convention."""
    digest=hashlib.sha256()
    for key in ("start_boundary","controls","goal"):
        value=np.ascontiguousarray(buffers[key][lane])
        digest.update(key.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()
