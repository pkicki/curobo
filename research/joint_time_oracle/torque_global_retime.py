# SPDX-License-Identifier: Apache-2.0
"""Optional IPOPT time-optimal retiming AFTER independent W qualification.

This is a distinct offline postprocessing arm, not a search/feasibility
oracle. Its solve time and failure status cannot be hidden inside the
global screening search latency. A locally converged IPOPT solution is
NOT independently physically certified until cohort_validate is rerun
on the produced P trajectory (and adaptive sampled if required).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear

from .cohort_geometry import QuinticGeometry
from .cohort_retime import problem, solve, states, ROOT
from .geometry_feasibility_core import sha256_file, smooth_time
from .torque_global_bank import OUTPUT
from .torque_global_screen import ARMS
from .torque_active import read_profile


def smoothstep_initial_speed_coefficients(duration:float)->np.ndarray:
    if not np.isfinite(duration) or duration<=0:
        raise ValueError("invalid source witness duration")
    # C1 smoothstep path phase x(s)=sdot**2, approximated with a positive
    # cubic B-spline. This is only an IPOPT initial guess, not a certificate.
    u=np.linspace(.0001,.9999,1000)
    s,su,_,_=smooth_time(u)
    target=(su/duration)**2
    basis=BSpline(np.r_[np.zeros(3),np.linspace(0,1,30),np.ones(3)],
                  np.eye(32),3)(s)
    ans=lsq_linear(basis,target,bounds=(1e-8,np.inf),
                   max_iter=200,tol=1e-10)
    if not ans.success:
        raise ValueError("speed spline NNLS initial fit failed")
    return np.maximum(ans.x,1e-8)


def retime(arm:Path,case:str,*,maxiter:int=200,grid:int=513)->dict:
    if maxiter<1 or grid<129:
        raise ValueError("invalid registered IPOPT budget")
    folder=arm/case
    receipt_file=arm/"w_adaptive_receipts.json"
    receipt=(json.loads(receipt_file.read_text()).get(case+":W")
             if receipt_file.exists() else None)
    if not isinstance(receipt,dict) or not (
        receipt.get("physical") is True and receipt.get("scaled95") is True):
        raise ValueError("only independently qualified W may be polished")
    profile_path=folder/"torque_profile.json"
    profile=read_profile(profile_path)
    geometry_doc=json.loads((folder/"w_geometry.json").read_text())
    if geometry_doc.get("profile_sha256")!=profile["profile_sha256"]:
        raise ValueError("geometry/profile mismatch")
    with np.load(folder/"w_dense.npz") as source:
        duration=float(source["time"][-1])
    if receipt.get("base_sha256")!=sha256_file(folder/"w_dense.npz"):
        raise ValueError("source W changed after qualification")
    geometry=QuinticGeometry(np.asarray(geometry_doc["controls"],dtype=float))
    initial=smoothstep_initial_speed_coefficients(duration)
    original_profile=os.environ.get("ORACLE_TORQUE_PROFILE")
    if original_profile is not None:
        raise ValueError("ambient ORACLE_TORQUE_PROFILE would change independent IPOPT problem")
    os.environ["ORACLE_TORQUE_PROFILE"]=str(profile_path.resolve())
    try:
        numerical_problem=problem(geometry.evaluate,geometry.spline.t,grid,initial)
        solution,information=solve(numerical_problem,initial,maxiter=maxiter)
        output={"schema_version":1,
                "case_id":case,"source_w_sha256":receipt["base_sha256"],
                "source_geometry_sha256":sha256_file(folder/"w_geometry.json"),
                "profile_sha256":profile["profile_sha256"],
                "source_W_duration_s":duration,"solver":information,
                "qualification":"NOT_QUALIFIED; independently sample/adaptively audit P",
                "schema":"full physical feasibility cannot be inferred from IPOPT status alone"}
        if information.get("solver_converged") and information.get("primal_min_margin",-1)<-1e-6:
            output["qualification"]="IPOPT_CONSTRAINT_VIOLATION"
        elif information.get("solver_converged"):
            timeline=states(geometry.evaluate,solution,count=8193,knots=geometry.spline.t)
            np.savez_compressed(folder/"p_dense.npz",**timeline)
            output["p_sha256"]=sha256_file(folder/"p_dense.npz")
            output["preliminary_P_duration_s"]=float(timeline["time"][-1])
        (folder/"ipopt_polish.json").write_text(json.dumps(output,indent=2)+"\n")
        return output
    finally:
        os.environ.pop("ORACLE_TORQUE_PROFILE",None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUTPUT)
    p.add_argument("--arm",choices=ARMS,required=True)
    p.add_argument("--case",required=True)
    p.add_argument("--maxiter",type=int,default=200)
    p.add_argument("--grid",type=int,default=513)
    a=p.parse_args()
    print(json.dumps(retime(a.root/a.arm,a.case,maxiter=a.maxiter,grid=a.grid),indent=2))


if __name__=="__main__":
    main()
