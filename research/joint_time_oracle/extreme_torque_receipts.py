# SPDX-License-Identifier: Apache-2.0
"""Issue fail-closed adaptive sampled witness receipts.

A receipt says a *specific time-dilated curve* passed independent Pinocchio
dynamics and analytic sphere/scene checks on its dense+adaptive sample grid.
It does not prove continuous-time feasibility.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cohort_torque_report import TOLERANCES, ENDPOINT_TOL
from .extreme_torque_core import sha256_file
from .torque_active import read_profile

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"research/joint_time_oracle/results/torque_stress_final_20261107/severe"
OUTPUT=ROOT/"research/joint_time_oracle/results/extreme_torque_witness_20261108"


def family_verification(diag:dict,mode:str) -> dict:
    checks={}
    for name,tolerance in TOLERANCES.items():
        value=diag.get("families",{}).get(name,{}).get(mode,{}).get("violation")
        checks[name]=bool(isinstance(value,(float,int)) and np.isfinite(value) and
                           0<=value<=tolerance)
    return checks


def verify_witness(source:Path,arm:Path,case:str,base:dict,adaptive:dict) -> dict:
    folder=arm/case
    profile=read_profile(folder/"torque_profile.json")
    w=folder/"w_dense.npz";wa=folder/"w_adaptive_dense.npz"
    meta_file=folder/"w_source.json"
    result={"case_id":case,"physical":False,"scaled95":False,
            "collision":False,"endpoint":False,
            "torque_profile_sha256":profile["profile_sha256"],
            "base_dense_sha256":None,"adaptive_dense_sha256":None,
            "failure_reasons":[],
            "qualification_scope":"independent dense/adaptive sampled; NOT continuous-time certification"}
    problems=result["failure_reasons"]
    if not w.exists() or not wa.exists() or not meta_file.exists():
        problems.append("missing base, adaptive or source file")
        return result
    metadata=json.loads(meta_file.read_text())
    result["base_dense_sha256"]=sha256_file(w)
    result["adaptive_dense_sha256"]=sha256_file(wa)
    if metadata.get("witness_dense_sha256")!=result["base_dense_sha256"]:
        problems.append("witness base SHA changed after prescreen")
    original=source/case/"d_dense.npz"
    if not original.exists() or metadata.get("source_D_dense_sha256")!=sha256_file(original):
        problems.append("source D original trajectory SHA mismatch")
    if metadata.get("derived_profile_sha256")!=profile["profile_sha256"]:
        problems.append("derived profile SHA mismatch")
    base_diag=base.get(case+":W")
    diag=adaptive.get(case+":W_adaptive")
    if not isinstance(base_diag,dict) or not isinstance(diag,dict):
        problems.append("missing independent base or adaptive diagnostics")
        return result
    for name,entry in (("base",base_diag),("adaptive",diag)):
        if entry.get("torque_profile_sha256")!=profile["profile_sha256"]:
            problems.append(name+" diagnostic profile mismatch")
    replay_file=folder/"w_adaptive_replay.json"
    replay=json.loads(replay_file.read_text()) if replay_file.exists() else {}
    if replay.get("status")!="REPLAYED":
        problems.append("exact adaptive quintic replay failed or missing")
    with np.load(w) as data:
        base_time=np.asarray(data["time"])
    with np.load(wa) as data:
        adaptive_time=np.asarray(data["time"])
    if len(base_time)<8193 or len(adaptive_time)<len(base_time) or not np.isfinite(adaptive_time).all():
        problems.append("insufficient or nonfinite adaptive samples")
    if len(adaptive_time)>1 and np.any(np.diff(adaptive_time)<=0):
        problems.append("non-monotone adaptive timestamps")
    if diag.get("samples")!=len(adaptive_time) or base_diag.get("samples")!=len(base_time):
        problems.append("independent audit sample-count mismatch")
    endpoints={}
    for key,tolerance in ENDPOINT_TOL.items():
        value=diag.get("endpoints",{}).get(key)
        endpoints[key]=bool(isinstance(value,(int,float)) and
                            np.isfinite(value) and abs(value)<=tolerance)
    result["endpoint_checks"]=endpoints
    result["endpoint"]=bool(all(endpoints.values()) and not problems)
    world=diag.get("world_min_clearance_m")
    self_clear=diag.get("self_min_clearance_m")
    result["collision"]=bool((world is None or
                             (np.isfinite(world) and world>=-1e-7)) and
                             self_clear is not None and
                             np.isfinite(self_clear) and self_clear>=-1e-7 and
                             not problems)
    result["adaptive_family_checks"]={}
    result["base_family_checks"]={}
    for mode in ("physical","scaled95"):
        result["adaptive_family_checks"][mode]=family_verification(diag,mode)
        result["base_family_checks"][mode]=family_verification(base_diag,mode)
    all_phys=all(result["base_family_checks"]["physical"].values()) and all(
        result["adaptive_family_checks"]["physical"].values())
    all_scaled=all(result["base_family_checks"]["scaled95"].values()) and all(
        result["adaptive_family_checks"]["scaled95"].values())
    result["physical"]=bool(result["endpoint"] and result["collision"] and all_phys)
    result["scaled95"]=bool(result["physical"] and all_scaled)
    result["adaptive_samples"]=int(len(adaptive_time))
    result["base_samples"]=int(len(base_time))
    result["added_samples"]=int(len(adaptive_time)-len(base_time))
    return result


def issue(source:Path,arm:Path) -> dict:
    base=json.loads((arm/"independent_sampled.json").read_text())
    adaptive=json.loads((arm/"independent_adaptive.json").read_text())
    tasks=json.loads((arm/"tasks.json").read_text())
    receipts={}
    for task in tasks:
        case=task["case_id"]
        try:
            receipts[case+":W"]=verify_witness(source,arm,case,base,adaptive)
        except Exception as exc:
            receipts[case+":W"]={"case_id":case,"physical":False,"scaled95":False,
                                 "collision":False,"endpoint":False,
                                 "failure_reasons":[repr(exc)]}
    (arm/"w_adaptive_receipts.json").write_text(json.dumps(receipts,indent=2)+"\n")
    return receipts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=SOURCE)
    p.add_argument("--root",type=Path,default=OUTPUT)
    p.add_argument("--regime",required=True)
    a=p.parse_args()
    result=issue(a.source,a.root/a.regime)
    print(json.dumps({"total":len(result),
                      "physical":sum(x.get("physical") is True for x in result.values()),
                      "scaled95":sum(x.get("scaled95") is True for x in result.values())},indent=2))


if __name__=="__main__":
    main()
