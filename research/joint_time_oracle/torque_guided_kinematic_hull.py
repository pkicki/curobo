# SPDX-License-Identifier: Apache-2.0
"""Conservative continuous-time joint kinematic bounds for the W time law.

For QuinticGeometry q(s), BSpline derivative control coefficients enclose
q, q_s, q_ss, q_sss globally by the convex-hull property (for clamped
nonnegative B-spline basis). For s(t/T)=10u³-15u⁴+6u⁵, exact maxima are:

 |s_dot|<=15/(8T), |s_ddot|<=10/(sqrt(3) T²), |s_jerk|<=60/T³.

Triangle-inequality chain-rule bounds then give conservative global
limits for qdot, qddot and joint jerk at ALL physical times. They are
usually loose. Do NOT interpret this as a torque, world/self collision,
or complete continuous-time trajectory certificate. Read physical
limits from model-aligned independent source arrays; freeze their SHA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cohort_geometry import QuinticGeometry
from .cohort_retime import ROOT
from .geometry_feasibility_core import sha256_file

BASELINE=ROOT/"research/joint_time_oracle/results/native_reconstruction_20261106/baseline_validation_states.npz"


def certificate(controls:np.ndarray,duration_s:float,limits:dict[str,np.ndarray])->dict:
    if not np.isfinite(duration_s) or duration_s<=0:
        raise ValueError("invalid timing duration")
    geometry=QuinticGeometry(np.asarray(controls,dtype=float))
    spline=geometry.spline
    dof=spline.c.shape[1]
    ctrl=np.asarray(spline.c[:len(controls)],dtype=float)
    lo,hi=np.asarray(limits["position"],dtype=float)
    if lo.shape!=(dof,) or hi.shape!=(dof,) or not np.all(lo<hi):
        raise ValueError("joint position bound mismatch")
    min_q=np.min(ctrl,axis=0);max_q=np.max(ctrl,axis=0)
    abs_derivatives=[]
    for r in (1,2,3):
        derivative=spline.derivative(r)
        coeff=np.asarray(derivative.c,dtype=float)
        if not np.isfinite(coeff).all():
            raise ValueError("nonfinite derivative spline coefficients")
        abs_derivatives.append(np.max(np.abs(coeff),axis=0))
    A,B,C=abs_derivatives
    T=float(duration_s)
    S1=15./(8*T)
    S2=10./(np.sqrt(3)*T*T)
    S3=60./T**3
    envelopes={
        "position_min":min_q,
        "position_max":max_q,
        "velocity_abs_max":A*S1,
        "acceleration_abs_max":B*S1**2+A*S2,
        "jerk_abs_max":C*S1**3+3*B*S1*S2+A*S3,
    }
    passes={}
    passes["position"]=bool(np.all(min_q>=lo) and np.all(max_q<=hi))
    for name,key in (("velocity","velocity_abs_max"),
                     ("acceleration","acceleration_abs_max"),
                     ("jerk","jerk_abs_max")):
        bound=np.asarray(limits[name],dtype=float)
        if bound.shape!=(2,dof) or not np.all(bound[0]<bound[1]):
            raise ValueError("invalid "+name+" bounds")
        magnitude=envelopes[key]
        # Valid even when the physical interval is asymmetric about 0.
        passes[name]=bool(np.all(-magnitude>=bound[0]) and
                          np.all(magnitude<=bound[1]))
    return {"scope":"conservative continuous-time POSITION/VELOCITY/ACCELERATION/JERK ONLY; no torque or collision certificate",
            "duration_s":T,"envelopes":{k:v.tolist() for k,v in envelopes.items()},
            "per_family_pass":passes,
            "all_kinematic_families_conservatively_verified":bool(all(passes.values())),
            "conservative_basis":"quintic BSpline derivative-control convex hull + analytic smoothstep time-chain maxima"}


def run(arm:Path,*,baseline:Path=BASELINE)->dict:
    with np.load(baseline) as z:
        limits={key:np.asarray(z[key+"_bounds"]) for key in
                ("position","velocity","acceleration","jerk")}
    cases=json.loads((arm/"tasks.json").read_text())
    outcomes={}
    for task in cases:
        case=task["case_id"]
        folder=arm/case
        try:
            geom=json.loads((folder/"w_geometry.json").read_text())
            with np.load(folder/"w_dense.npz") as z:
                T=float(z["time"][-1])
            result=certificate(np.asarray(geom["controls"]),T,limits)
            result["baseline_limits_sha256"]=sha256_file(baseline)
            result["candidate_geometry_sha256"]=sha256_file(folder/"w_geometry.json")
            result["witness_dense_sha256"]=sha256_file(folder/"w_dense.npz")
        except Exception as exc:
            result={"status":"ERROR","reason":repr(exc),
                    "all_kinematic_families_conservatively_verified":False}
        outcomes[case]=result
        (folder/"kinematic_hull_certificate.json").write_text(
            json.dumps(result,indent=2)+"\n")
    (arm/"kinematic_hull_summary.json").write_text(
        json.dumps(outcomes,indent=2)+"\n")
    return outcomes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",required=True,type=Path)
    p.add_argument("--baseline",type=Path,default=BASELINE)
    a=p.parse_args()
    print(json.dumps(run(a.arm,baseline=a.baseline),indent=2))


if __name__=="__main__":
    main()
