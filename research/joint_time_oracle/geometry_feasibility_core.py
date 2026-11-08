# SPDX-License-Identifier: Apache-2.0
"""Preregistered joint-selective synthetic actuator scenarios and constructive witnesses.

For a geometric curve q(s), use s(u)=10u^3-15u^4+6u^5, u=t/T.
This time law has zero endpoint velocity and acceleration. Increasing T
suppresses inertial and velocity-dependent terms by T^-2 while gravity stays.
This permits a constructive feasibility test *if* the entire path can avoid
gravity-overloaded configurations, subject to independent physical validation.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .torque_active import canonical_hash, checked_bounds, torque_utilization

# Fixed before task outcomes; alternate hypotheses must go in a NEW protocol.
JOINT_GROUPS = (
    (2, 3, 4), (1, 2, 4), (2, 4, 6), (3, 4, 5), (1, 3, 5),
    (2, 3, 5, 6), (1, 2, 3, 4), (3, 4, 6, 7), (2, 4, 5, 7),
)
JOINT_FACTORS = (0.20, 0.30, 0.40, 0.50, 0.65, 0.80)
WITNESS_TIMES_S = (0.3, 0.5, 0.75, 1., 1.5, 2., 3., 4., 6., 8., 12., 20.)
# Additional margin below 95%-scaled limits in physical admissibility.
GRAVITY_SCALED95_TARGET = 0.88
ENDPOINT_SCALED95_TARGET = 0.80
MIN_ORIGINAL_LP_GAP = 0.01
MIN_GEOMETRY_DEVIATION_RAD = 0.10
CANDIDATE_CONTROLS = 14
CANDIDATE_PERTURBATION_RAD = (0.25, 0.50, 0.80, 1.20)
CANDIDATE_SEEDS = 32


def scenarios():
    """Stable scenario inventory, independent of native-planner failure."""
    return [
        {"scenario_id":f"j{''.join(map(str, group))}_f{int(factor*1000):04d}",
         "selected_joints_1based":list(group), "derate_on_selected":factor}
        for group in JOINT_GROUPS for factor in JOINT_FACTORS
    ]


def sha256_file(path:Path):
    hasher=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def profile_from_scenario(*, source_task:dict, original_limits:np.ndarray,
                          endpoint_gravity:np.ndarray, scenario:dict,
                          model_sha256:str, source_geometry_sha256:str) -> dict:
    bounds=checked_bounds(original_limits)
    g=np.asarray(endpoint_gravity,dtype=np.float64)
    if g.shape!=(2,bounds.shape[1]):
        raise ValueError("endpoint gravity must be [2,njoints]")
    joints=scenario["selected_joints_1based"]
    factor=float(scenario["derate_on_selected"])
    if not joints or not 0<factor<=1 or any(not 1<=j<=bounds.shape[1] for j in joints):
        raise ValueError("invalid synthetic actuator scenario")
    scales=np.ones(bounds.shape[1])
    scales[np.asarray(joints,dtype=int)-1]=factor
    limits=bounds*scales
    endpoint_max=float(np.max(torque_utilization(g,limits*.95)))
    admitted=bool(endpoint_max<=ENDPOINT_SCALED95_TARGET)
    label=str(scenario["scenario_id"])
    p={
        "case_id":label+"::"+source_task["case_id"],
        "source_case_id":source_task["case_id"],
        "regime":label,"actuator_scale_by_joint":scales.tolist(),
        "derate":1.,  # legacy scalar field ignored when vector is provided
        "nominal_effort_lower":bounds[0].tolist(),
        "nominal_effort_upper":bounds[1].tolist(),
        "effort_lower":limits[0].tolist(),
        "effort_upper":limits[1].tolist(),
        "admitted":admitted,
        "rejection_reason":None if admitted else "endpoint Scaled95 gravity reserve violated",
        "endpoint_gravity_scaled95_utilization":endpoint_max,
        "endpoint_scaled95_max":ENDPOINT_SCALED95_TARGET,
        "source_task_sha256":source_task["task_sha256"],
        "source_geometry_sha256":source_geometry_sha256,
        "model_sha256":model_sha256,
        "scenario":scenario,
        "provenance":"frozen joint-selective synthetic actuator weakening; not Panda ratings",
    }
    p["profile_sha256"]=canonical_hash(p)
    return p


def smooth_time(u:np.ndarray):
    """Return s, ds/du, d²s/du², d³s/du³ for quintic rest-to-rest time law."""
    u=np.asarray(u,dtype=np.float64)
    if not np.isfinite(u).all() or np.any(u<0) or np.any(u>1):
        raise ValueError("time phase outside [0,1]")
    return (
        u**3*(10.-15.*u+6.*u*u),
        30*u*u*(1-u)**2,
        60*u-180*u*u+120*u**3,
        60-360*u+360*u*u,
    )


def timed_states(geometry,T:float,*,n:int=1025,extra_u=None)->dict:
    """Exact path derivatives + analytic chain rule; does NOT retime via IPOPT."""
    if not np.isfinite(T) or T<=0 or n<3:
        raise ValueError("invalid duration or sample count")
    u=np.linspace(0,1,n)
    if extra_u is not None:
        u=np.unique(np.r_[u,np.asarray(extra_u,dtype=np.float64)])
    s,su,suu,suuu=smooth_time(u)
    q,qs,qss,qsss=geometry.evaluate(s)
    qs,qss,qsss=(np.asarray(z,dtype=float) for z in (qs,qss,qsss))
    v=su/T
    a=suu/T**2
    j=suuu/T**3
    return {"time":u*T,"phase":u,"progress":s,
            "q":np.asarray(q,dtype=float),
            "qd":qs*v[:,None],
            "qdd":qss*v[:,None]**2+qs*a[:,None],
            "jerk":qsss*v[:,None]**3+3*qss*v[:,None]*a[:,None]+qs*j[:,None]}


def smoothstep_controls(qstart:np.ndarray,qgoal:np.ndarray,
                        *,count:int=CANDIDATE_CONTROLS)->np.ndarray:
    if count<8:
        raise ValueError("need >=8 quintic controls")
    q0,q1=(np.asarray(q,dtype=float) for q in (qstart,qgoal))
    if q0.shape!=q1.shape or q0.ndim!=1:
        raise ValueError("joint endpoint mismatch")
    z=np.linspace(0,1,count)
    smooth=z**3*(10-15*z+6*z*z)
    controls=q0[None]+smooth[:,None]*(q1-q0)[None]
    controls[:3]=q0
    controls[-3:]=q1
    return controls


def perturb_controls(base:np.ndarray, rng:np.random.Generator,
                     *,amplitude:float, target_joint: int | None=None)->np.ndarray:
    """Coordinated waypoint detours; preserve first and last three controls."""
    out=np.asarray(base,dtype=float).copy()
    if out.ndim!=2 or out.shape[0]<8:
        raise ValueError("bad controls")
    direction=rng.normal(size=out.shape[1])
    if target_joint is not None:
        direction[target_joint]+=np.sign(direction[target_joint] or 1.)*2.5
    direction/=max(np.linalg.norm(direction),1e-12)
    n=len(out)
    ctr=float(rng.choice((.35,.50,.65)))
    spread=float(rng.choice((.16,.25,.32)))
    index=np.linspace(0,1,n)
    envelope=np.exp(-.5*((index-ctr)/spread)**2)
    envelope[:3]=0;envelope[-3:]=0
    out+=amplitude*envelope[:,None]*direction[None]
    return out


def path_distance(a:np.ndarray,b:np.ndarray) -> float:
    """Max joint-space L2 gap at matched normalized *joint-space arc length*."""
    x,y=np.asarray(a),np.asarray(b)
    if x.ndim!=2 or y.ndim!=2 or x.shape[1]!=y.shape[1]:
        raise ValueError("path samples mismatch")
    def resample(z):
        distances=np.r_[0,np.cumsum(np.linalg.norm(np.diff(z,axis=0),axis=1))]
        if distances[-1]<=1e-12:
            raise ValueError("zero-length path")
        normalized=distances/distances[-1]
        unique,idx=np.unique(normalized,return_index=True)
        grid=np.linspace(0,1,257)
        return np.column_stack([np.interp(grid,unique,z[idx,k]) for k in range(z.shape[1])])
    return float(np.max(np.linalg.norm(resample(x)-resample(y),axis=1)))


def gate_l1(profile:dict, origin:dict, alternative:dict, *,min_lp_gap=MIN_ORIGINAL_LP_GAP)->dict:
    """Pure numerical research gate; never substitute for independent audits."""
    gap=origin.get("max_required_torque_bound_expansion")
    strong=isinstance(gap,(int,float)) and np.isfinite(gap) and gap>=min_lp_gap
    altlp=alternative.get("max_required_torque_bound_expansion")
    altok=isinstance(altlp,(int,float)) and np.isfinite(altlp) and altlp<1e-7
    return {"endpoint_admitted":profile.get("admitted") is True,
            "original_lp_gap":gap,
            "original_lp_positive":bool(strong),
            "alternative_pointwise_lp_feasible":bool(altok),
            "discovery_pass":bool(profile.get("admitted") is True and strong and altok),
            "scope":"screen only; candidate must be independently validated and source-disjoint"}
