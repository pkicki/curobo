# SPDX-License-Identifier: Apache-2.0
"""Independent CPU reconstruction of a returned TOPP-RA+Newton K107 timing.

No cuRobo K107 runtime is imported. Uses scipy BSpline derivatives and
Gaussian quadrature of dt/ds=1/sqrt(x(s)), and re-evaluates the exact
unchanged quintic path. This is only a trajectory reconstruction:
independent payload-aware Pinocchio and sphere checking must validate
every output on S8193 and event-aware/adaptive grids.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from numpy.polynomial.legendre import leggauss

from .geometry_feasibility_core import sha256_file


def reconstruct(controls:np.ndarray,coeff:np.ndarray,knots:np.ndarray,
                *,samples:int=8193,quadrature:int=32)->dict[str,np.ndarray]:
    c=np.asarray(controls,dtype=float)
    xcoeff=np.asarray(coeff,dtype=float)
    xknot=np.asarray(knots,dtype=float)
    if c.shape!=(22,7) or xcoeff.shape!=(107,) or xknot.shape!=(111,):
        raise ValueError("unmatched K107/quintic controls or knots")
    if samples<8193 or quadrature<12:
        raise ValueError("insufficient independent grid/quadrature")
    if not np.isfinite(c).all() or not np.isfinite(xcoeff).all() or (
        not np.isfinite(xknot).all()):
        raise ValueError("nonfinite K107 response")
    path=BSpline(np.r_[np.zeros(6),np.linspace(0,1,18)[1:-1],np.ones(6)],c,5,
                 extrapolate=False)
    law=BSpline(xknot,xcoeff,3,extrapolate=False)
    s=np.linspace(0.,1.,samples,dtype=float)
    qs=path(s);q1=path(s,nu=1);q2=path(s,nu=2);q3=path(s,nu=3)
    x=law(s)
    if not np.all(np.isfinite(x)) or np.min(x)<-1e-8:
        raise ValueError("nonfinite or negative squared-speed profile")
    if np.min(x[1:-1])<=1e-12:
        raise ValueError("squared speed has an interior zero; time may be infinite")
    x1=law(s,nu=1);x2=law(s,nu=2)
    root=np.sqrt(np.maximum(x,0.))
    v=q1*root[:,None]
    a=q2*x[:,None]+.5*q1*x1[:,None]
    jerk=(q3*x[:,None]+1.5*q2*x1[:,None]+.5*q1*x2[:,None])*root[:,None]
    # Gauss integration on each progress interval, including close to
    # endpoints where x can vanish. No clipped endpoint duration estimate.
    nodes,weights=leggauss(quadrature)
    lo=s[:-1,None];hi=s[1:,None]
    points=(lo+hi)/2.+(hi-lo)/2.*nodes[None]
    xq=law(points)
    if not np.isfinite(xq).all() or np.min(xq)<=0:
        raise ValueError("nonpositive squared speed inside physical integration quadrature")
    dt=np.sum((hi-lo)/2.*weights[None]/np.sqrt(xq),axis=1)
    if not np.isfinite(dt).all() or np.min(dt)<=0:
        raise ValueError("nonfinite/unphysical timing quadrature")
    t=np.r_[0.,np.cumsum(dt)]
    if not all(np.isfinite(arr).all() for arr in (qs,v,a,jerk,t)):
        raise ValueError("nonfinite physical K107 sample reconstruction")
    return {"time":t,"progress":s,"q":qs,"qd":v,"qdd":a,"jerk":jerk}


def replay(folder:Path)->dict:
    req=json.loads((folder/"k107_request.json").read_text())
    ans=json.loads((folder/"k107_response.json").read_text())
    if ans.get("request_sha256")!=sha256_file(folder/"k107_request.json"):
        raise ValueError("retimer response/request SHA mismatch")
    if ans.get("status")!="ONLINE_SCREEN_PASS" or ans.get("online_feasible") is not True:
        raise ValueError("unqualified K107 response cannot generate claimed candidate")
    if ans.get("constraint_mode")!="C3_NO_TR" or ans.get("torque_rate_enforced") is not False:
        raise ValueError("K107/native constraint family mismatch")
    with np.load(folder/"geometry.npz") as f:
        controls=np.asarray(f["controls"],dtype=float)
    if sha256_file(folder/"geometry.npz")!=req["geometry_sha256"]:
        raise ValueError("K107 input path changed")
    result=reconstruct(controls,np.asarray(ans["coefficients"]),np.asarray(ans["knots"]))
    if not np.allclose(result["q"][[0,-1]],controls[[0,-1]],atol=1e-8):
        raise ValueError("retimed path endpoints changed")
    # Compare independent elapsed-time integral with the returned K107
    # duration. A discrepancy is a numerical issue, not feasibility.
    claimed=ans.get("duration_s")
    if claimed is None or not np.isfinite(claimed) or abs(result["time"][-1]-claimed)>.005:
        raise ValueError(f"K107 duration mismatch: {result['time'][-1]} vs {claimed}")
    output=folder/"k107_dense.npz"
    if output.exists():
        raise ValueError("refusing to overwrite independent K107 replay")
    np.savez_compressed(output,**result)
    return {"case_sha256":req["source_case_sha256"],
            "response_sha256":sha256_file(folder/"k107_response.json"),
            "dense_sha256":sha256_file(output),
            "elapsed_s":float(result["time"][-1]),
            "sample_count":len(result["time"]),
            "state":"RECONSTRUCTED_NOT_INDEPENDENTLY_PHYSICALLY_QUALIFIED"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request-dir",required=True,type=Path)
    a=p.parse_args()
    doc=replay(a.request_dir)
    (a.request_dir/"k107_replay.json").write_text(json.dumps(doc,indent=2)+"\n")
    print(json.dumps(doc,indent=2))


if __name__=="__main__":main()
