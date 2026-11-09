# SPDX-License-Identifier: Apache-2.0
"""Exact cuRoboV2-Fast *uniform-squared-speed* seed interval diagnosis.

Source audit (pinned native kernel timeopt_io_kernel.cuh):
  fused_coefficients_and_seed_bounds gathers per-sample constraints on ONE
  scalar x=sdot² and writes tile triples [upper,lower,valid].
  fused_seed intersects all tile intervals and requires upper>0,
  upper>=lower and every tile valid, unless fixed-start semantics differ.
An empty intersection prevents native TimeOpt from starting. That is
NOT a proof of fixed-path infeasibility under a NONUNIFORM x(s).

Use side-by-side K107/independent Pinocchio on the SAME geometric
SHA to distinguish 'uniform seed fails' from 'all timing fails'.
Do not infer geometry-wide infeasibility from native seed flags.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize_seed_bounds(bounds:np.ndarray)->list[dict]:
    arr=np.asarray(bounds,dtype=float)
    if arr.ndim!=3 or arr.shape[2]!=3 or arr.shape[1]<1:
        raise ValueError("native seed bounds must be [lanes,tiles,3]")
    result=[]
    for lane,tiles in enumerate(arr):
        upper=float(np.min(tiles[:,0]))
        lower=float(np.max(tiles[:,1]))
        valid=bool(np.all(tiles[:,2]>0))
        finite=bool(np.isfinite(tiles).all())
        gap=upper-lower
        if not finite:status="NONFINITE_SEED_COEFFICIENTS"
        elif not valid:status="INVALID_SEED_COEFFICIENTS"
        elif upper<=0 or gap<0:status="NO_COMMON_UNIFORM_SPEED"
        else:status="UNIFORM_SPEED_INTERVAL_EXISTS"
        # These are *observed* necessary uniform-speed seed bounds, not
        # a global path-feasibility LP with variable x(s).
        result.append({"lane":lane,"status":status,
             "uniform_speed_lower":lower,"uniform_speed_upper":upper,
             "gap_upper_minus_lower":gap,"all_tiles_valid":valid,
             "bounds_finite":finite,
             "tightest_upper_tile":int(np.argmin(tiles[:,0])),
             "tightest_lower_tile":int(np.argmax(tiles[:,1])),
             "interpretation":"uniform x(s)=constant seed only, not arbitrary retiming"})
    return result


def compare_with_k107(native:dict,k107:dict,*,same_exact_geometry:bool)->dict:
    if not same_exact_geometry:
        return {"status":"GEOMETRY_NOT_MATCHED","source":"cannot attribute different path comparison to native seed gate"}
    if native["status"]=="NO_COMMON_UNIFORM_SPEED" and (
         k107.get("status")=="ONLINE_SCREEN_PASS" and k107.get("online_feasible") is True):
        return {"status":"K107_NONUNIFORM_RESCUE_CANDIDATE",
                "source":"K107 online screen only; NEED independent physical+Scaled95 certification"}
    if native["status"]=="NO_COMMON_UNIFORM_SPEED" and (
         k107.get("status") in ("TOPPRA_SEED_FAILED","ONLINE_SCREEN_REJECTED")):
        return {"status":"BOTH_FAILED_TO_FIND_TIMING",
                "source":"does NOT prove all nonuniform retiming laws are infeasible"}
    return {"status":"INCONCLUSIVE_COMPARISON"}


def analyze(native_folder:Path,*,output:Path)->dict:
    reports=[]
    files=sorted(native_folder.glob("*_timeopt_phase_buffers.npz"))
    for file in files:
        with np.load(file) as z:
            if "solver_seed_bounds" not in z:
                reports.append({"file":str(file),"status":"SEED_BOUNDS_NOT_CAPTURED"})
                continue
            rows=summarize_seed_bounds(z["solver_seed_bounds"])
        reports.append({"file":str(file),"lanes":rows,
                        "no_common_uniform":sum(x["status"]=="NO_COMMON_UNIFORM_SPEED"
                                                 for x in rows)})
    report={"schema_version":1,"snapshots":reports,
            "note":"Source-native CUDA seed kernel demands uniform-squared-speed interval; comparison to arbitrary x(s) requires K107 on same exact path and independent audit"}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-folder",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    out=analyze(a.native_folder,output=a.output)
    print(json.dumps({"snapshots":len(out["snapshots"])},indent=2))


if __name__=="__main__":main()
