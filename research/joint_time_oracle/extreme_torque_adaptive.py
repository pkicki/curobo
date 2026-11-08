# SPDX-License-Identifier: Apache-2.0
"""Exact-quintic adaptive replay of time-dilated alternative witnesses.

Run in the independent Pinocchio/cyipopt environment after the first
cohort_validate --methods W pass. Import the source *quintic* geometry and
the existing D speed spline, scale its squared-speed controls by rho**-2,
then reconstruct exact derivatives on a refined progress grid.

The resulting arrays require a SECOND independent RNEA/collision audit;
they are not certificates merely because this replay succeeded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cohort_geometry import QuinticGeometry
from .cohort_retime import states
from .torque_active import read_profile

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"research/joint_time_oracle/results/torque_stress_final_20261107/severe"
OUTPUT=ROOT/"research/joint_time_oracle/results/extreme_torque_witness_20261108"


def extra_progress(diagnostic:dict, witness:dict) -> np.ndarray:
    t=np.asarray(witness["time"],dtype=float)
    s=np.asarray(witness["progress"],dtype=float)
    if len(t)!=len(s) or np.any(np.diff(t)<=0) or np.any(np.diff(s)<=0):
        raise ValueError("nonmonotonic witness time/path-progress grid")
    queries=[]
    for entry in diagnostic.get("families",{}).values():
        for mode in ("physical","scaled95"):
            at=entry.get(mode,{}).get("time_s")
            if at is not None:
                queries.append(float(at))
    for key in ("world_min_time_s","self_min_time_s"):
        if diagnostic.get(key) is not None:
            queries.append(float(diagnostic[key]))
    if not queries:
        return s
    ds=1./8192
    extras=[]
    for at in queries:
        progress=float(np.interp(at,t,s))
        extras.extend(np.clip(progress+ds*np.array([-2.,-1.,-.2,0.,.2,1.,2.]),0,1))
    return np.unique(np.r_[s,extras])


def replay(source:Path,arm:Path,*,case_filter:str|None=None) -> dict:
    diagnostics_path=arm/"independent_sampled.json"
    if not diagnostics_path.exists():
        raise ValueError("independent base witness diagnostic missing")
    diagnostics=json.loads(diagnostics_path.read_text())
    tasks=json.loads((arm/"tasks.json").read_text())
    results={}
    for task in tasks:
        case=task["case_id"]
        if case_filter is not None and case!=case_filter:
            continue
        folder=arm/case
        witness=folder/"w_dense.npz"
        metadata_file=folder/"w_source.json"
        status={"case_id":case,"status":"ERROR"}
        try:
            profile=read_profile(folder/"torque_profile.json")
            diagnostic=diagnostics.get(case+":W")
            if diagnostic is None or not witness.exists():
                raise ValueError("missing preliminary witness or independent sampled audit")
            metadata=json.loads(metadata_file.read_text())
            if not metadata.get("witness_screen_pass"):
                raise ValueError("witness did not pass prescreen")
            if diagnostic.get("torque_profile_sha256")!=profile["profile_sha256"]:
                raise ValueError("independent diagnostic profile SHA mismatch")
            with np.load(witness) as data:
                ref={k:np.asarray(data[k]) for k in data.files}
            with np.load(source/case/"d_dense.npz") as data:
                old={k:np.asarray(data[k]) for k in data.files}
            if not all(key in old for key in ("controls","knots","progress")):
                raise ValueError("frozen D spline controls/knots/progress missing")
            coefs=json.loads((source/case/"c_best.json").read_text())["coefficients"]
            geometry=QuinticGeometry(np.asarray(coefs,dtype=float))
            rho=float(metadata["witness_slowdown"])
            progress=extra_progress(diagnostic,ref)
            dense=states(geometry.evaluate,old["controls"]/rho**2,
                         knots=old["knots"],extra=progress)
            # Every originally frozen witness sample must be present and agree
            # in position, velocity, acceleration, jerk AND execution time.
            idx=np.searchsorted(dense["progress"],ref["progress"])
            if np.any(idx>=len(dense["progress"])):
                raise ValueError("missing endpoint in exact-spline replay")
            position_error=float(np.max(np.abs(dense["progress"][idx]-ref["progress"])))
            if position_error>1e-11:
                raise ValueError(f"source progress grid not preserved {position_error}")
            errors={key:float(np.max(abs(dense[key][idx]-ref[key])))
                    for key in ("time","q","qd","qdd","jerk")}
            tolerances={"time":1e-5,"q":1e-6,"qd":2e-5,"qdd":5e-4,"jerk":5e-3}
            if any(errors[key]>tolerance for key,tolerance in tolerances.items()):
                raise ValueError(f"exact retimed spline reconstruction mismatch: {errors}")
            output=folder/"w_adaptive_dense.npz"
            np.savez_compressed(output,**dense)
            status={"case_id":case,"status":"REPLAYED",
                    "samples":int(len(dense["time"])),
                    "added_samples":int(len(dense["time"])-len(ref["time"])),
                    "max_replay_errors":errors,
                    "note":"exact-spline replay, not yet independently qualified"}
        except Exception as exc:
            status["reason"]=repr(exc)
        results[case]=status
        (folder/"w_adaptive_replay.json").write_text(json.dumps(status,indent=2)+"\n")
        print(case,status["status"],status.get("reason",""),flush=True)
    (arm/"w_adaptive_replay_summary.json").write_text(json.dumps(results,indent=2)+"\n")
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=SOURCE)
    p.add_argument("--root",type=Path,default=OUTPUT)
    p.add_argument("--regime",required=True)
    p.add_argument("--case")
    a=p.parse_args()
    replay(a.source,a.root/a.regime,case_filter=a.case)


if __name__=="__main__":
    main()
