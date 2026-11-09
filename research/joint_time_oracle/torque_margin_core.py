# SPDX-License-Identifier: Apache-2.0
"""Pure mathematical contracts for frozen-geometry margin-aware retiming.

Given P with geometric path q(s) and squared speed x(s), uniform
slowdown rho>=1 is EXACTLY x_rho(s)=x(s)/rho^2. Consequently q is unchanged,
qd scales 1/rho, qdd 1/rho^2, jerk 1/rho^3, and:
  tau_rho = g(q) + (tau_P - g(q))/rho^2.
This gravity term means torque feasibility is NOT generally monotonic
with slowdown. Therefore use a REGISTERED discrete grid; do not
bisection-assume monotonicity or treat a successful approximate screen as
independent physical certification. IPOPT reoptimization is separate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .cohort_torque_report import TOLERANCES

DILATION_FACTORS = (1.0, 1.005, 1.01, 1.025, 1.05, 1.075,
                    1.10, 1.15, 1.20, 1.30, 1.50, 2.0, 2.5, 3.0)
TIGHTENING_FACTORS = (0.98, 0.96, 0.94, 0.92)
CASE_IDS = ("trap_cluttered_08", "confirm_geom_table_07",
            "confirm_geom_cluttered_02")
DEFAULT_SOURCE_ARM = "no_lp"
MIN_DURATION_IMPROVEMENT = .10


def sha256_file(path:Path)->str:
    result=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            result.update(block)
    return result.hexdigest()


def slowdown_states(source:dict[str,np.ndarray],rho:float)->dict[str,np.ndarray]:
    """Reconstruct analytical time dilation, preserving geometry/progress."""
    if not np.isfinite(rho) or rho<1.:
        raise ValueError("rho must be finite and >= 1")
    names=("time","q","qd","qdd","jerk")
    if any(k not in source for k in names):
        raise ValueError("missing trajectory derivatives")
    n=len(source["time"])
    if n<2 or any(len(source[k])!=n for k in names):
        raise ValueError("trajectory sample count mismatch")
    if not all(np.isfinite(source[k]).all() for k in names):
        raise ValueError("source trajectory is nonfinite")
    if np.any(np.diff(source["time"])<=0):
        raise ValueError("nonmonotonic timing")
    result={k:np.asarray(v).copy() for k,v in source.items()}
    result["time"]=np.asarray(source["time"],dtype=np.float64)*rho
    result["qd"]=np.asarray(source["qd"],dtype=np.float64)/rho
    result["qdd"]=np.asarray(source["qdd"],dtype=np.float64)/rho**2
    result["jerk"]=np.asarray(source["jerk"],dtype=np.float64)/rho**3
    return result


def scale_torque(tau:np.ndarray,gravity:np.ndarray,rho:float)->np.ndarray:
    if rho<1. or not np.isfinite(rho):
        raise ValueError("invalid dilation")
    tau=np.asarray(tau,dtype=np.float64)
    gravity=np.asarray(gravity,dtype=np.float64)
    if tau.shape!=gravity.shape or not np.isfinite(tau).all() or not np.isfinite(gravity).all():
        raise ValueError("invalid torque/gravity arrays")
    return gravity+(tau-gravity)/rho**2


def family_bounds(bounds:np.ndarray, family:str, *,scaled95:bool)->np.ndarray:
    a=np.asarray(bounds,dtype=np.float64)
    if a.ndim!=2 or a.shape[0]!=2 or not np.isfinite(a).all() or not np.all(a[0]<a[1]):
        raise ValueError("invalid family limits")
    if not scaled95 or family=="position":
        return a
    if family=="torque":
        # Exact cuRobo native signed actuator scaling toward zero.
        return a*.95
    center=(a[0]+a[1])/2.
    return np.stack((center+.95*(a[0]-center),
                     center+.95*(a[1]-center)))


def sampled_max_violation(values:np.ndarray,bounds:np.ndarray)->dict:
    x=np.asarray(values,dtype=np.float64)
    lo,hi=np.asarray(bounds,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=len(lo) or not np.isfinite(x).all():
        raise ValueError("invalid sampled values")
    lower=lo-x;upper=x-hi
    a=np.maximum(lower,upper)
    i,j=np.unravel_index(int(np.argmax(a)),a.shape)
    return {"violation":float(max(0.,a[i,j])),
            "signed_max":float(a[i,j]),
            "joint":int(j+1),"sample":int(i),
            "side":"lower" if lower[i,j]>upper[i,j] else "upper"}


def pilot_limits(states:dict[str,np.ndarray],tau:np.ndarray,
                 physical_bounds:dict[str,np.ndarray],rho:float)->dict:
    """Computationally cheap sampled gate; NEVER a qualification receipt."""
    scaled=slowdown_states(states,rho)
    gravity=np.asarray(states["gravity"],dtype=np.float64)
    torques=scale_torque(tau,gravity,rho)
    arrays={"position":scaled["q"],"velocity":scaled["qd"],
            "acceleration":scaled["qdd"],"jerk":scaled["jerk"],
            "torque":torques}
    families={}
    for family,values in arrays.items():
        limits=np.asarray(physical_bounds[family])
        families[family]={mode:sampled_max_violation(
            values,family_bounds(limits,family,scaled95=(mode=="scaled95")))
            for mode in ("physical","scaled95")}
    # Deliberately require a margin inside established validator tolerances;
    # independent dense/adaptive checks still decide any final GO.
    pass_screen=all(
        families[f]["scaled95"]["violation"]<=.25*tol
        for f,tol in TOLERANCES.items())
    return {"rho":rho,"duration_s":float(scaled["time"][-1]),
            "family_residuals":families,"conservative_screen_pass":pass_screen,
            "scope":"sampled dynamics only; no collision or independent certification"}


def sampled_dilation_interval(states:dict[str,np.ndarray],tau:np.ndarray,
                              physical_bounds:dict[str,np.ndarray])->dict:
    """Exact sampled Scaled95 admissible z=1/rho^2 interval, 0<z<=1.

    Velocity, acceleration, jerk scale with z**(.5,1,1.5). Torque is
    affine in z with gravity intercept. Intersect ALL sampled joint
    inequalities; no monotonicity assumption. This is diagnostic only:
    neither time-independent collision nor the continuous path between
    samples is checked, and no candidate is selected by this result.
    """
    if "gravity" not in states:
        raise ValueError("gravity required for sampled torque intervals")
    tau=np.asarray(tau,dtype=np.float64)
    g=np.asarray(states["gravity"],dtype=np.float64)
    if tau.ndim!=2 or tau.shape!=g.shape or not np.isfinite(tau).all() or not np.isfinite(g).all():
        raise ValueError("torque/gravity sample mismatch")
    low_z,high_z=0.,1.
    sources={}
    if sampled_max_violation(np.asarray(states["q"]),
                             physical_bounds["position"])["violation"]>0:
        return {"status":"EMPTY_POSITION_INTERVAL","z_interval":None,
                "scope":"sampled Scaled95 necessary conditions only"}
    def update(kind:str,lower:np.ndarray,upper:np.ndarray):
        nonlocal low_z,high_z
        low=float(np.max(lower))
        high=float(np.min(upper))
        sources[kind]={"z_lower":low,"z_upper":high}
        low_z=max(low_z,low);high_z=min(high_z,high)
    for name,key,power in (("velocity","qd",.5),
                           ("acceleration","qdd",1.),
                           ("jerk","jerk",1.5)):
        values=np.asarray(states[key],dtype=np.float64)
        lim=family_bounds(physical_bounds[name],name,scaled95=True)
        if values.shape!=tau.shape or not np.isfinite(values).all():
            raise ValueError("invalid "+name+" sample array")
        lower=np.broadcast_to(lim[0],values.shape)
        upper=np.broadcast_to(lim[1],values.shape)
        nz=np.abs(values)>1e-12
        if np.any(~nz & ((lower>0)|(upper<0))):
            return {"status":"EMPTY_ZERO_"+name.upper(),
                    "z_interval":None,"scope":"sampled only"}
        lo=np.zeros_like(values);hi=np.ones_like(values)
        first=np.divide(lower,values,out=np.full_like(values,-np.inf),where=nz)
        second=np.divide(upper,values,out=np.full_like(values,np.inf),where=nz)
        lo[nz]=np.maximum(0,np.minimum(first[nz],second[nz]))**(1/power)
        hi[nz]=np.maximum(0,np.minimum(1,np.maximum(first[nz],second[nz])))**(1/power)
        update(name,lo,hi)
    effort=family_bounds(physical_bounds["torque"],"torque",scaled95=True)
    d=tau-g
    lower=np.broadcast_to(effort[0],tau.shape)
    upper=np.broadcast_to(effort[1],tau.shape)
    nz=np.abs(d)>1e-12
    if np.any(~nz & ((g<lower)|(g>upper))):
        return {"status":"EMPTY_STATIC_TORQUE_INTERVAL",
                "z_interval":None,"scope":"sampled only"}
    t1=np.divide(lower-g,d,out=np.zeros_like(tau),where=nz)
    t2=np.divide(upper-g,d,out=np.zeros_like(tau),where=nz)
    lo=np.zeros_like(tau);hi=np.ones_like(tau)
    lo[nz]=np.minimum(t1[nz],t2[nz])
    hi[nz]=np.maximum(t1[nz],t2[nz])
    update("torque",lo,hi)
    if low_z>high_z+1e-12 or high_z<=0:
        return {"status":"EMPTY_INTERSECTION","z_interval":[low_z,high_z],
                "by_family":sources,"scope":"sampled necessary only"}
    z0=max(0.,low_z);z1=min(1.,high_z)
    return {"status":"SAMPLED_ADMISSIBLE_INTERVAL",
            "z_interval":[z0,z1],
            "rho_min":float(1/np.sqrt(z1)),
            "rho_max":float(1/np.sqrt(z0)) if z0>0 else None,
            "by_family":sources,
            "scope":"fixed-shape sampled necessary AND sufficient for these sampled algebraic bounds only; no independent/adaptive/continuous-time qualification"}


def tighten_pointwise_problem(p, beta:float):
    """Tighten the existing 95%-scaled IPOPT envelope by beta < 1.

    For asymmetric torque bounds the original center is 0.95*(lo+hi)/2
    and the width is 0.475*(hi-lo). Additional beta multiplies BOTH,
    so torque limits are scaled about zero rather than their midpoint.
    Other kinematic families retain their center and shrink widths.
    """
    if not np.isfinite(beta) or not 0<beta<=1.:
        raise ValueError("invalid additional tightening factor")
    if np.asarray(p.center).shape!=(4,7) or np.asarray(p.width).shape!=(4,7):
        raise ValueError("unexpected PointwiseTiming bounds")
    p.center[3]*=beta
    p.width[:]*=beta
    return p


def immutable_json(path:Path,document:dict):
    path=Path(path)
    payload=json.dumps(document,sort_keys=True,indent=2)+"\n"
    if path.exists() and json.loads(path.read_text())!=document:
        raise ValueError("frozen artifact changed: "+str(path))
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(payload)
