# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retime geometrically distinct stored cuRobo candidate lanes on stress cases."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear

from .cohort_geometry import NativeGeometry
from .cohort_retime import PILOT, problem, solve, states
from .torque_active import read_profile

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/"research/joint_time_oracle/results/torque_stress_final_20261107"


def run(regime: str, root: Path, cases: list[str], *, grid: int=513,
        maxiter: int=500) -> dict:
    """Solve exact lane retimings using the same pointwise IPOPT backend as B/D."""
    arm=root/regime;out={"regime":regime,"grid":grid,"maxiter":maxiter,"cases":{}}
    basis=BSpline(np.r_[np.zeros(3),np.linspace(0,1,30),np.ones(3)],np.eye(32),3)
    for case in cases:
        source=PILOT/case;folder=arm/case
        with np.load(source/"a10_buffers.npz") as z:buffers={k:np.asarray(z[k]) for k in z.files}
        with np.load(source/"native_input.npz") as z:native={k:np.asarray(z[k]) for k in z.files}
        profile=read_profile(folder/"torque_profile.json")
        if profile["source_case_id"]!=case or profile["regime"]!=regime:
            raise ValueError(f"profile mismatch for {case}/{regime}")
        lanes=[]
        for lane in range(len(buffers["controls"])):
            if not bool(buffers["success"][lane]):
                lanes.append({"lane":lane,"status":"SOURCE_LANE_FAILED"});continue
            geo=NativeGeometry(buffers,lane);start,end=geo.nonstationary_interval();fn=geo.evaluate_trimmed
            count=buffers["controls"].shape[1];native_knots=np.arange(1,count+3)/(count+3)
            knots=(native_knots-start)/(end-start)
            sgrid=np.linspace(0,1,1001);target=basis(start+(end-start)*sgrid)@buffers["timing"][lane]/(end-start)**2
            init=lsq_linear(basis(sgrid),target,bounds=(1e-6,np.inf)).x
            profile_path=(folder/"torque_profile.json").resolve()
            os.environ["ORACLE_TORQUE_PROFILE"]=str(profile_path)
            extras=np.array([]);passes=[];best_controls=init.copy();result=None
            for refinement in range(3):
                p=problem(fn,knots,grid,best_controls,extras)
                stamp=time.perf_counter();candidate,info=solve(p,best_controls,maxiter)
                info["wall_including_problem_s"]=time.perf_counter()-stamp
                reconstructed=states(fn,candidate,knots=knots)
                dense_p=np.asarray(fn(reconstructed["progress"]))
                x=BSpline(np.r_[np.zeros(3),np.linspace(0,1,30),np.ones(3)],candidate,3)(reconstructed["progress"])
                dx=BSpline(np.r_[np.zeros(3),np.linspace(0,1,30),np.ones(3)],candidate,3)(reconstructed["progress"],nu=1)
                from .cohort_retime import path_coefficients
                coeff=path_coefficients(dense_p)
                tau=coeff[2]+coeff[1]*x[:,None]+.5*coeff[0]*dx[:,None]
                limits=np.load(ROOT/"research/joint_time_oracle/results/native_reconstruction_20261106/baseline_validation_states.npz")
                raw=np.stack([limits[k+"_bounds"] for k in ("velocity","acceleration","jerk","torque")])
                raw[3]=np.stack((profile["effort_lower"],profile["effort_upper"]))
                values=np.stack((reconstructed["qd"],reconstructed["qdd"],reconstructed["jerk"],tau))
                ratios=[]
                for family in range(4):
                    lo,hi=raw[family];center=(lo+hi)/2
                    ratios.append(np.where(values[family]>=center,(values[family]-center)/(hi-center),(center-values[family])/(center-lo)))
                worst=float(max(np.max(x) for x in ratios))
                info.update({"dense_duration_s":float(reconstructed["time"][-1]),
                             "dense_max_utilization":worst,"refinement":refinement})
                passes.append({k:v for k,v in info.items() if k!="history"})
                result={"controls":candidate,"trajectory":reconstructed,"info":info}
                if worst<=1+1e-6:
                    best_controls=candidate;break
                worst_idx=np.unravel_index(np.argmax(np.stack(ratios)),np.stack(ratios).shape)
                new=np.unique(reconstructed["progress"][worst_idx[1]]) if len(worst_idx)>1 else np.array([])
                extras=np.unique(np.r_[extras,new,np.clip(new-1e-8,0,1),np.clip(new+1e-8,0,1)])
                best_controls=candidate
            assert result is not None
            name=f"X{lane}"
            np.savez_compressed(folder/f"x{lane}_dense.npz",**result["trajectory"],
                                controls=result["controls"],knots=knots)
            item={"lane":lane,"status":info["message"],"solver_converged":info["solver_converged"],
                  "solver_status":info["status"],"duration_s":info["dense_duration_s"],
                  "dense_max_utilization":info["dense_max_utilization"],"solver_history":passes,
                  "source_success":bool(buffers["success"][lane]),
                  "source_geometry_hash":__import__('hashlib').sha256(np.ascontiguousarray(buffers['controls'][lane]).tobytes()).hexdigest(),
                  "trajectory_file":name.lower()+"_dense.npz"}
            lanes.append(item)
            print(regime,case,name,item["duration_s"],item["solver_converged"],item["dense_max_utilization"],flush=True)
        out["cases"][case]={"source_candidate_count":len(lanes),"lanes":lanes}
        (folder/"geometric_lane_search.json").write_text(json.dumps(out["cases"][case],indent=2)+"\n")
    target=root/regime/"geometric_lane_search.json"
    target.write_text(json.dumps(out,indent=2)+"\n")
    return out


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime",choices=("moderate","severe"),required=True)
    parser.add_argument("--root",type=Path,default=DEFAULT)
    parser.add_argument("--cases",default="pilot_free_00,pilot_table_00,pilot_cluttered_00")
    parser.add_argument("--grid",type=int,default=513);parser.add_argument("--maxiter",type=int,default=500)
    args=parser.parse_args()
    print(json.dumps(run(args.regime,args.root,args.cases.split(','),grid=args.grid,maxiter=args.maxiter),indent=2))


if __name__=="__main__":main()
