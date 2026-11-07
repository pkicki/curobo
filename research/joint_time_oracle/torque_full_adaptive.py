# SPDX-License-Identifier: Apache-2.0
"""Replay exact full-native candidate time laws at dense+adaptive physical times.

This script runs in the cuRobo GPU environment; independent Pinocchio/sphere
audit and receipt creation must be performed afterward in the other env.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from curobo._src.motion.motion_trajectory_timeopt import TrajectoryTimeOpt
from curobo._src.motion.motion_trajectory_timeopt_cfg import TrajectoryTimeOptCfg

from .cohort_explore import planner_for
from .cohort_torque_adaptive import _near_times
from .torque_active import read_profile

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/"research/joint_time_oracle/results/torque_stress_final_20261107"


def replay(arm:Path,*,case_filter:str|None=None) -> dict:
    native=arm/"native_full"
    diagnostics=json.loads((native/"independent_sampled.json").read_text())
    tasks=json.loads((arm/"tasks.json").read_text())
    cache={}
    outcomes={}
    for task in tasks:
        case=task["case_id"]
        if case_filter and case!=case_filter:
            continue
        folder=native/case
        candidate_file=folder/"native_candidates.json"
        candidates=json.loads(candidate_file.read_text()) if candidate_file.exists() else {}
        profile=read_profile(arm/case/"torque_profile.json")
        outcomes[case]={}
        for name,entry in candidates.items():
            if entry.get("status")!="SUCCESS":
                continue
            source=folder/(name.lower()+"_dense.npz")
            diag=diagnostics.get(case+":"+name)
            if not source.exists() or not isinstance(diag,dict):
                outcomes[case][name]={"status":"MISSING_SOURCE_OR_DIAGNOSTICS"}
                continue
            try:
                with np.load(source) as a:
                    original={k:np.asarray(a[k]) for k in ("time","q","qd","qdd","jerk")}
                with np.load(folder/(name.lower()+"_buffers.npz")) as a:
                    frozen={k:np.asarray(a[k]) for k in a.files}
                with np.load(folder/(name.lower()+"_native_input.npz")) as a:
                    native_input={k:np.asarray(a[k]) for k in a.files}
                duration=float(original["time"][-1])
                query=_near_times(diag,duration,np.asarray(original["time"],dtype=np.float64))
                query=np.unique(query.astype(np.float32))
                if task["scene"] not in cache:
                    cache[task["scene"]]=planner_for(task["scene"])
                planner=cache[task["scene"]]
                device=planner.default_joint_state.position.device
                cast=lambda x:torch.as_tensor(x,device=device,dtype=torch.float32).contiguous()
                config=TrajectoryTimeOptCfg(
                    robot=planner.kinematics.config,
                    batch_size=int(frozen["controls"].shape[0]),
                    n_knots=int(native_input["knot"].shape[1]),
                    use_cuda_graph=False,
                    torque_lower=cast(profile["effort_lower"]),
                    torque_upper=cast(profile["effort_upper"]),
                    limit_scale=.95,jerk_limit_scale=.95)
                solver=TrajectoryTimeOpt(config)
                solver._setup_path(reference=True)
                for key,attr in (("controls","_controls"),
                                 ("start_boundary","_start_boundary"),
                                 ("goal","_goal"),("timing","_timing"),
                                 ("integrated_times","_times"),
                                 ("success","_success"),("stationary","_stationary")):
                    dest=getattr(solver,attr)
                    dest.copy_(cast(frozen[key]))
                solver._has_result=True
                lane=int(entry["lane"])
                times=cast(query).repeat(config.batch_size,1).contiguous()
                state=solver.sample(times)
                arrays={"time":query,
                        "q":state.position[lane].detach().cpu().numpy().copy(),
                        "qd":state.velocity[lane].detach().cpu().numpy().copy(),
                        "qdd":state.acceleration[lane].detach().cpu().numpy().copy(),
                        "jerk":state.jerk[lane].detach().cpu().numpy().copy()}
                lookup=np.searchsorted(query,original["time"].astype(np.float32))
                if not np.array_equal(query[lookup],original["time"].astype(np.float32)):
                    raise ValueError("original sample grid missing from adaptive query")
                discrepancy=max(float(np.max(np.abs(arrays[k][lookup]-original[k])))
                                for k in ("q","qd","qdd","jerk"))
                # Float32 reconstruction should be identical or near-exact at shared points.
                # Do not silently accept a rerun with a different TimeOpt law.
                if discrepancy>2e-3:
                    raise ValueError(f"native adaptive replay mismatch {discrepancy}")
                np.savez_compressed(folder/(name.lower()+"_adaptive_dense.npz"),**arrays)
                outcomes[case][name]={"status":"REPLAYED","samples":len(query),
                                      "added_samples":len(query)-len(original["time"]),
                                      "max_shared_state_error":discrepancy}
            except Exception as exc:
                outcomes[case][name]={"status":"ERROR","reason":repr(exc)}
        (folder/"native_adaptive_replay.json").write_text(json.dumps(outcomes[case],indent=2)+"\n")
        print(case,outcomes[case],flush=True)
    for planner in cache.values():
        planner.destroy()
    (native/"adaptive_replay_summary.json").write_text(json.dumps(outcomes,indent=2)+"\n")
    return outcomes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=DEFAULT)
    parser.add_argument("--regime",required=True,choices=("moderate","severe"))
    parser.add_argument("--case")
    a=parser.parse_args()
    replay(a.root/a.regime,case_filter=a.case)


if __name__=="__main__":
    main()
