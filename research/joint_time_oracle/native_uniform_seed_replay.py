# SPDX-License-Identifier: Apache-2.0
"""Independent replay of native fused_seed uniform-squared-speed admission.

Source of definition:
cuRoboV2 timeopt_io_kernel.cuh fused_coefficients_and_seed_bounds and
fused_seed. This is a GLOBAL constant x=sdot² seed interval formed by
intersecting constraints at every sampled path point, not a general
(nonuniform) TOPP reachability/certification test.

An empty interval means NO uniform-squared-speed seed satisfies the
sampled necessary inequalities. It does NOT imply absence of a feasible
nonuniform x(s) nor an inherent TimeOpt kernel bug. Use K107 or a
nonuniform offline oracle on the EXACT same geometric path to test this.

Signed lower/upper torque coefficients are from independent Pinocchio
RNEA with q_s and q_ss. This script never changes native solver code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from .payload_contract import Payload, model_aligned_payload, sha256_file

FAMILY=("velocity","acceleration","jerk","torque")
EPS=1e-10


def uniform_seed_bounds(q:np.ndarray,qs:np.ndarray,qss:np.ndarray,
                        qsss:np.ndarray,gravity:np.ndarray,dynamic:np.ndarray,
                        limits:dict)->dict:
    """Return extremal lower/upper uniform x bounds and exact witnesses."""
    arrays=(q,qs,qss,qsss,gravity,dynamic)
    shape=np.asarray(q).shape
    if shape[1:]!=(7,) or len(shape)!=2 or any(
        np.asarray(z).shape!=shape or not np.isfinite(z).all() for z in arrays):
        raise ValueError("invalid path/derivative/dynamics shapes")
    v=np.asarray(limits["velocity"],dtype=float)
    a=np.asarray(limits["acceleration"],dtype=float)
    j=np.asarray(limits["jerk"],dtype=float)
    effort=np.asarray(limits["torque"],dtype=float)
    if (v.shape!=(7,) or a.shape!=(7,) or j.shape!=(7,) or
        effort.shape!=(2,7) or not np.all(v>0) or
        not np.all(a>0) or not np.all(j>0) or
        np.any(effort[0]>=effort[1])):
        raise ValueError("invalid native limit convention")
    upper=[];lower=[]; invalid=[]
    for k in range(shape[0]):
        for joint in range(7):
            for name,derivative,limit,power in (
                ("velocity",abs(qs[k,joint]),v[joint],2.),
                ("acceleration",abs(qss[k,joint]),a[joint],1.),
                ("jerk",abs(qsss[k,joint]),j[joint],2./3.)):
                if derivative>0:
                    cap=float((limit/derivative)**power)
                    upper.append((cap,name,k,joint))
            B=float(dynamic[k,joint]);g=float(gravity[k,joint])
            if abs(B)>EPS:
                bounds=sorted(((float(effort[0,joint])-g)/B,
                               (float(effort[1,joint])-g)/B))
                lower.append((bounds[0],"torque_lower",k,joint))
                upper.append((bounds[1],"torque_upper",k,joint))
            elif g<effort[0,joint]-1e-9 or g>effort[1,joint]+1e-9:
                invalid.append({"sample":k,"joint":joint+1,
                                "gravity":g,"effort_lower":float(effort[0,joint]),
                                "effort_upper":float(effort[1,joint])})
    hi=min(upper,key=lambda v:v[0]) if upper else (float("inf"),"none",-1,-1)
    lo=max(lower,key=lambda v:v[0]) if lower else (0.,"x_nonnegative",-1,-1)
    lower_value=max(0.,lo[0])
    possible=not invalid and hi[0]>0 and hi[0]>=lower_value
    def witness(x):
        return {"value":float(x[0]),"family":x[1],"sample":int(x[2]),
                "joint":int(x[3]+1) if x[3]>=0 else None}
    return {"status":"UNIFORM_SEED_POSSIBLE" if possible else "UNIFORM_SEED_EMPTY",
            "uniform_x_lower":float(lower_value),
            "uniform_x_upper":float(hi[0]),
            "interval_gap":float(hi[0]-lower_value),
            "lower_witness":witness(lo),
            "upper_witness":witness(hi),
            "zero_dynamic_gravity_rejects":invalid,
            "samples":shape[0],
            "assumption":"constant x=sdot², hence x'=0; nonuniform timing may remain feasible",
            "proof_scope":"independent replica of sampled necessary native seed bound, not full TOPP feasibility"}


def sampled_dynamics(geometry,payload:Payload,progress:np.ndarray)->dict:
    """Build independent path coefficients with the matched inertial payload."""
    model,data,state,qi,vi=model_aligned_payload(payload)
    s=np.asarray(progress,dtype=float)
    q,qs,qss,qsss=geometry.evaluate(s)
    q=np.asarray(q);qs=np.asarray(qs);qss=np.asarray(qss);qsss=np.asarray(qsss)
    grav=np.empty_like(q);dynamic=np.empty_like(q)
    v=np.zeros(model.nv);a=np.zeros(model.nv)
    for k,x in enumerate(q):
        state[qi]=x
        v.fill(0.);a.fill(0.)
        grav[k]=pin.rnea(model,data,state,v,a)[vi]
        v[vi]=qs[k];a[vi]=qss[k]
        dynamic[k]=pin.rnea(model,data,state,v,a)[vi]-grav[k]
    return {"q":q,"qs":qs,"qss":qss,"qsss":qsss,
            "gravity":grav,"dynamic":dynamic}


def study(geometry,payload:Payload,limits:dict,*,samples:int=1025)->dict:
    if samples<129:
        raise ValueError("insufficient registered uniform seed samples")
    s=np.linspace(0,1,samples)
    d=sampled_dynamics(geometry,payload,s)
    result=uniform_seed_bounds(**d,limits=limits)
    result["payload"]=payload.serialize()
    result["sampling_progress"]=s.tolist()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls",required=True,type=Path,
                        help="NPZ with exact quintic 'controls' array")
    parser.add_argument("--payload",required=True,type=Path)
    parser.add_argument("--limits",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--samples",type=int,default=1025)
    args=parser.parse_args()
    from .cohort_geometry import QuinticGeometry
    with np.load(args.controls) as z:
        controls=np.asarray(z["controls"])
    result=study(QuinticGeometry(controls),
                 Payload(**json.loads(args.payload.read_text())),
                 json.loads(args.limits.read_text()),samples=args.samples)
    result["source_path_sha256"]=sha256_file(args.controls)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in
        ("status","uniform_x_lower","uniform_x_upper","interval_gap")},indent=2))


if __name__=="__main__":
    main()
