# SPDX-License-Identifier: Apache-2.0
"""Re-evaluate exact squared-speed P/R spline near independent constraint peaks.

Unlike smoothstep W, R follows a general cubic squared-speed spline.
Adaptive progress samples must be mapped from measured physical time
through the monotonically increasing R time-progress mapping, then
reevaluated from frozen controls (NOT interpolating qd/qdd/jerk).
Dense/adaptive physical and Scaled95 audit is independent and follows
this replay. Source W and source P are never changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cohort_geometry import QuinticGeometry
from .cohort_retime import states
from .geometry_feasibility_core import sha256_file
from .torque_margin_prepare import OUTPUT


def adaptive_progress(diag:dict,source:dict,*,n:int=8193)->np.ndarray:
    time=np.asarray(source["time"],dtype=float)
    progress=np.asarray(source["progress"],dtype=float)
    if (len(time)!=len(progress) or np.any(np.diff(time)<=0) or
        np.any(np.diff(progress)<=0)):
        raise ValueError("nonmonotone source R time/progress")
    add=[]
    for family in diag.get("families",{}).values():
        for mode in ("physical","scaled95"):
            value=family.get(mode,{}).get("time_s")
            if value is not None:
                s=float(np.interp(float(value),time,progress))
                add.extend(np.clip(s+np.array([-4,-2,-1,-.25,0,.25,1,2,4])/8192,0,1))
    for name in ("world_min_time_s","self_min_time_s"):
        t=diag.get(name)
        if t is not None:
            s=float(np.interp(float(t),time,progress))
            add.extend(np.clip(s+np.array([-4,-2,-1,-.25,0,.25,1,2,4])/8192,0,1))
    return np.unique(np.r_[np.linspace(0,1,n),add])


def replay(arm:Path,*,case_filter:str|None=None)->dict:
    diag_file=arm/"independent_sampled.json"
    if not diag_file.exists():
        raise ValueError("missing independent 8193-sample R diagnostics")
    diagnostics=json.loads(diag_file.read_text())
    tasks=json.loads((arm/"tasks.json").read_text())
    results={}
    for task in tasks:
        case=task["case_id"]
        if case_filter and case!=case_filter:continue
        folder=arm/case
        row={"case_id":case,"status":"FAILED"}
        try:
            frozen=json.loads((folder/"r_registration.json").read_text())
            geometrydoc=json.loads((folder/"r_geometry.json").read_text())
            if frozen["r_geometry_sha256"]!=sha256_file(folder/"r_geometry.json"):
                raise ValueError("modified R geometry after pilot")
            if frozen["r_sha256"]!=sha256_file(folder/"r_dense.npz"):
                raise ValueError("modified R witness after pilot")
            controls_file=folder/"r_speed_controls.npz"
            control_receipt=json.loads((folder/"r_controls_receipt.json").read_text())
            if sha256_file(controls_file)!=control_receipt.get("r_speed_controls_sha256"):
                raise ValueError("R speed controls changed before adaptive replay")
            if control_receipt.get("r_dense_sha256")!=frozen["r_sha256"]:
                raise ValueError("R controls not linked to source retimed state")
            if control_receipt.get("source_P_controls_sha256")!=frozen["source_files_sha256"]["p_timing_controls.npz"]:
                raise ValueError("R controls source P SHA mismatch")
            with np.load(controls_file) as z:
                speeds=np.asarray(z["squared_speed_controls"],dtype=float)
            with np.load(folder/"r_dense.npz") as z:
                original={k:np.asarray(z[k]) for k in z.files}
            diag=diagnostics.get(case+":R")
            if not isinstance(diag,dict) or diag.get("torque_profile_sha256")!=frozen["profile_sha256"]:
                raise ValueError("missing/mismatched independently sampled R diag")
            progress=adaptive_progress(diag,original)
            geom=QuinticGeometry(np.asarray(geometrydoc["controls"],dtype=float))
            reconstructed=states(geom.evaluate,speeds,count=8193,
                                 knots=geom.spline.t,extra=progress)
            # The exact source knots/progress must remain present in the
            # reconstructed samples. Quadrature timing may differ slightly
            # due to subdivision; quantify it rather than silently ignore.
            source_s=np.asarray(original["progress"])
            s_new=np.asarray(reconstructed["progress"])
            idx=np.searchsorted(s_new,source_s)
            if idx.max()>=len(s_new) or np.max(abs(s_new[idx]-source_s))>1e-12:
                raise ValueError("exact source progress points absent")
            errors={name:float(np.max(abs(reconstructed[name][idx]-original[name])))
                    for name in ("q","qd","qdd","jerk","time")}
            if any(errors[key]>1e-8 for key in ("q","qd","qdd","jerk")):
                raise ValueError("exact squared-speed spline reconstruction mismatch")
            if errors["time"]>1e-3:
                raise ValueError("adaptive quadrature changed time >1ms")
            dest=folder/"r_adaptive_dense.npz"
            if dest.exists():
                raise ValueError("frozen adaptive R already exists")
            np.savez_compressed(dest,**reconstructed)
            row={"case_id":case,"status":"REPLAYED",
                 "source_r_sha256":frozen["r_sha256"],
                 "r_adaptive_sha256":sha256_file(dest),
                 "r_speed_controls_sha256":control_receipt["r_speed_controls_sha256"],
                 "source_samples":len(source_s),
                 "adaptive_samples":len(s_new),
                 "max_replay_errors":errors,
                 "scope":"exact analytic squared-speed spline replay; independent audit pending"}
        except Exception as exc:
            row["reason"]=repr(exc)
        results[case]=row
        (folder/"r_adaptive_replay.json").write_text(json.dumps(row,indent=2)+"\n")
        print(case,row["status"],flush=True)
    (arm/"r_adaptive_replay_summary.json").write_text(json.dumps(results,indent=2)+"\n")
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUTPUT)
    p.add_argument("--arm",required=True)
    p.add_argument("--case")
    a=p.parse_args()
    print(json.dumps(replay(a.root/a.arm,case_filter=a.case),indent=2))


if __name__=="__main__":
    main()
