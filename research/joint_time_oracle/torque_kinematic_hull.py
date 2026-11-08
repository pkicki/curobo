# SPDX-License-Identifier: Apache-2.0
"""Conservative whole-trajectory *kinematic* bounds for quintic-spline witnesses.

For q(s) a clamped quintic B-spline, q^(k)(s) lies in the convex hull of
the k-th derivative spline coefficients. For the fixed rest-to-rest phase
s(u)=10u^3-15u^4+6u^5 we have the rigorous polynomial derivative maxima

    max |s_u| = 15/8
    max |s_uu| = 10/sqrt(3) ~=5.7735
    max |s_uuu| = 60.

Combine these with chain-rule triangle inequalities to obtain conservative
global joint position, velocity, acceleration and jerk bounds. These
whole-domain bounds may be too loose and therefore return UNKNOWN; they
must never be used to reject a trajectory as physically infeasible.
They DO NOT certify torque, collision or the complete trajectory.
Floating-point rounding is padded but not an interval-formal proof.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cohort_geometry import QuinticGeometry
from .cohort_retime import ROOT


def derivative_hull_max(geometry:QuinticGeometry,order:int)->np.ndarray:
    if order not in (0,1,2,3):
        raise ValueError("expected spline derivative order 0..3")
    bs=geometry.spline.derivative(order)
    count=len(bs.t)-bs.k-1
    data=np.asarray(bs.c[:count],dtype=np.float64)
    if data.ndim!=2 or not np.isfinite(data).all():
        raise ValueError("invalid derivative spline control hull")
    return np.max(np.abs(data),axis=0)


def global_kinematic_hull(controls:np.ndarray,T:float,
                          bounds:dict[str,np.ndarray])->dict:
    """Sufficient global constraints using analytic B-spline convex hulls."""
    if not np.isfinite(T) or T<=0:
        raise ValueError("invalid duration")
    geometry=QuinticGeometry(np.asarray(controls,dtype=np.float64))
    q0,q1,q2,q3=[derivative_hull_max(geometry,k) for k in range(4)]
    # Polynomial absolute-maximum of each derivative of the phase law.
    k1=15./8.
    k2=10./np.sqrt(3.)
    k3=60.
    maxima={
        "velocity":q1*k1/T,
        "acceleration":(q2*k1**2+q1*k2)/T**2,
        "jerk":(q3*k1**3+3*q2*k1*k2+q1*k3)/T**3,
    }
    result={"duration_s":float(T),"families":{},
            "scope":"sufficient conservative whole-domain kinematic B-spline hull, not torque/collision certification",
            "floating_point":"analytical inequalities in binary64, not outward-rounded formal interval arithmetic"}
    for family in ("position","velocity","acceleration","jerk"):
        limits=np.asarray(bounds[family],dtype=float)
        if limits.shape!=(2,len(q0)) or not np.all(np.isfinite(limits)):
            raise ValueError("invalid "+family+" bounds")
        lo,hi=limits
        if np.any(lo>=hi):
            raise ValueError("inverted "+family+" bounds")
        if family=="position":
            # Direct B-spline convex hull applies to the control ordinates,
            # not to their absolute values.
            coeff=geometry.spline.c[:len(geometry.spline.t)-geometry.spline.k-1]
            lower=np.min(coeff,axis=0)
            upper=np.max(coeff,axis=0)
        else:
            magnitude=maxima[family]
            lower=-magnitude
            upper=magnitude
        # A small additional roundoff allowance errs toward UNKNOWN,
        # not toward falsely passing a numerically borderline trajectory.
        pad=256*np.finfo(float).eps*(np.maximum(1,abs(lower))+np.maximum(1,abs(upper)))
        passed=bool(np.all(lower-pad>=lo) and np.all(upper+pad<=hi))
        result["families"][family]={
            "sufficient_whole_domain_pass":passed,
            "whole_domain_lower_bound":(lower-pad).tolist(),
            "whole_domain_upper_bound":(upper+pad).tolist(),
            "margin_to_lower":float(np.min(lower-pad-lo)),
            "margin_to_upper":float(np.min(hi-upper-pad)),
            "not_pass_interpretation":"UNKNOWN, not demonstrated violation"}
    result["all_four_kinematic_families_pass"]=all(
        f["sufficient_whole_domain_pass"] for f in result["families"].values())
    return result


def verify_arm(arm:Path)->dict:
    baseline=(ROOT/"research/joint_time_oracle/results/native_reconstruction_20261106"
              /"baseline_validation_states.npz")
    with np.load(baseline) as f:
        limits={name:np.asarray(f[name+"_bounds"])
                for name in ("position","velocity","acceleration","jerk")}
    tasks=json.loads((arm/"tasks.json").read_text())
    output={}
    for task in tasks:
        case=task["case_id"]
        folder=arm/case
        try:
            geom=json.loads((folder/"w_geometry.json").read_text())
            with np.load(folder/"w_dense.npz") as f:
                T=float(np.asarray(f["time"])[-1])
                q0=np.asarray(f["q"])[0]
                qend=np.asarray(f["q"])[-1]
            controls=np.asarray(geom["controls"])
            if np.max(abs(controls[0]-q0))>1e-5 or (
                np.max(abs(controls[-1]-qend))>1e-5):
                raise ValueError("witness endpoints diverge from geometric controls")
            output[case]={"status":"COMPLETE",**global_kinematic_hull(controls,T,limits)}
        except Exception as exc:
            output[case]={"status":"ERROR","reason":repr(exc),
                          "all_four_kinematic_families_pass":False}
        (folder/"kinematic_hull.json").write_text(json.dumps(output[case],indent=2)+"\n")
    (arm/"kinematic_hull_summary.json").write_text(json.dumps(output,indent=2)+"\n")
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True,
                   help="One controlled-search arm directory")
    args=p.parse_args()
    result=verify_arm(args.root)
    print(json.dumps({"cases":len(result),
                      "whole_domain_kinematic_pass":sum(x.get(
                          "all_four_kinematic_families_pass") is True
                          for x in result.values())},indent=2))


if __name__=="__main__":
    main()
