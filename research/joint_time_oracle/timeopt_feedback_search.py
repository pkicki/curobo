# SPDX-License-Identifier: Apache-2.0
"""TimeOpt-in-the-loop geometric search; pure proposal/selection policy.

Arms:
  random_global   independent global geometry proposals, TimeOpt only scores
  duration_rank   best TimeOpt-scored elite becomes proposal center
  timeopt_feedback elite + joint/progress-limited targeted modification
                  selected from a real TimeOpt bottleneck diagnostic

All arms pay for the same number of K107 queries per scored task/budget
and record COMPLETE request wall time. No LP-dual-gradient optimization
is reused. A retimer failure, especially an incompatible K107 contract,
is NOT evidence that a geometric path is intrinsically infeasible.

This module has no GPU import; injected evaluator executes the actual
K107 adapter in the pinned environment (or a synthetic mock in tests).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable

import numpy as np
from scipy.interpolate import BSpline

from .cohort_geometry import QuinticGeometry
from .timeopt_oracle_contract import path_record

ARMS=("random_global","duration_rank","timeopt_feedback")
CONTROL_COUNT=22
INIT_BATCH=12
RADII=(0.18,0.35,0.60,1.0,1.5)
BUDGETS=(32,128,256)


def rng_seed(case_id:str,seed:int,arm:str)->int:
    # Identical initial random stream for all arms. Feedback is the ONLY
    # registered reason proposals may diverge after INIT_BATCH.
    if arm not in ARMS:
        raise ValueError("unknown arm")
    h=hashlib.sha256(f"{case_id}|{seed}|common".encode()).digest()
    return int.from_bytes(h[:8],"little")


def nominal_path(start:np.ndarray,goal:np.ndarray,*,count:int=CONTROL_COUNT)->np.ndarray:
    """Exact rest-flat quintic smoothstep in a 22-control B-spline space."""
    q0=np.asarray(start,dtype=float)
    q1=np.asarray(goal,dtype=float)
    if q0.shape!=(7,) or q1.shape!=(7,) or not all(
        np.isfinite(v).all() for v in (q0,q1)
    ):
        raise ValueError("seven-joint finite task endpoints required")
    knots=np.r_[np.zeros(5),np.linspace(0,1,count-4),np.ones(5)]
    points=np.linspace(0,1,513)
    basis=BSpline(knots,np.eye(count),5,extrapolate=False)(points)
    s=points
    smooth=10*s**3-15*s**4+6*s**5
    target=q0[None]+smooth[:,None]*(q1-q0)[None]
    controls=np.zeros((count,7))
    controls[:3]=q0
    controls[-3:]=q1
    rhs=target-basis[:,:3]@controls[:3]-basis[:,-3:]@controls[-3:]
    controls[3:-3]=np.linalg.lstsq(basis[:,3:-3],rhs,rcond=1e-13)[0]
    error=np.max(np.abs(QuinticGeometry(controls).evaluate(points)[0]-target))
    if error>1e-7:
        raise ValueError("nominal smoothstep B-spline projection not exact")
    return controls


def normalize_perturbation(d:np.ndarray)->np.ndarray:
    d=np.asarray(d,dtype=float).copy()
    d[:3]=0;d[-3:]=0
    scale=np.max(np.linalg.norm(d,axis=1))
    if not np.isfinite(scale) or scale<1e-12:
        raise ValueError("zero or nonfinite geometric proposal")
    return d/scale


def bump_direction(rng:np.random.Generator,*,s:float|None=None,
                   joint:int|None=None,count:int=CONTROL_COUNT,
                   focused:bool=False)->np.ndarray:
    grid=np.linspace(0,1,count)
    center=float(rng.uniform(.17,.83) if s is None else np.clip(s,.15,.85))
    spread=float(rng.uniform(.11,.42) if not focused else rng.uniform(.12,.25))
    mask=np.exp(-.5*((grid-center)/spread)**2)
    direction=rng.normal(size=7)
    if focused and joint is not None and 0<=joint<7:
        direction*=.25
        direction[joint]+=rng.choice([-1.,1.])*2.
    return normalize_perturbation(mask[:,None]*direction[None,:])


@dataclass
class Candidate:
    controls:np.ndarray
    status:str
    response:dict
    proposal:int
    novelty:float
    cost:float


def _score(result:dict)->float:
    """Research proposal quality: finite feasible duration, then soft violation."""
    status=result.get("status")
    if status=="FEASIBLE_SCREENED":
        duration=result.get("duration_s")
        return float(duration) if duration is not None and np.isfinite(duration) else 1e9
    diag=result.get("diagnostics") or {}
    penalty=diag.get("max_normalized_violation")
    if status=="INFEASIBLE_SCREENED" and penalty is not None and np.isfinite(float(penalty)):
        return 1e3+max(0.,float(penalty))
    return 1e6


def _distinct(x:np.ndarray,prev:list[Candidate])->float:
    if not prev:return float("inf")
    return min(float(np.linalg.norm(x-c.controls)/np.sqrt(x.size)) for c in prev)


def _elite(archive:list[Candidate],*,min_novelty:float=.01)->Candidate|None:
    valid=sorted((c for c in archive if c.cost<1e6),key=lambda c:c.cost)
    for candidate in valid:
        if candidate.novelty>=min_novelty:return candidate
    return valid[0] if valid else None


def propose(base:np.ndarray,archive:list[Candidate],*,arm:str,
            step:int,rng:np.random.Generator)->tuple[np.ndarray,dict]:
    if arm not in ARMS:
        raise ValueError("unknown feedback policy")
    amplitude=RADII[(step//3)%len(RADII)]
    elite=_elite(archive) if arm!="random_global" and step>=INIT_BATCH else None
    parent=base if elite is None else elite.controls
    feedback=None
    use_feedback=arm=="timeopt_feedback" and elite is not None
    if use_feedback:
        d=elite.response.get("diagnostics") or {}
        # Only a measured physical bottleneck, never an invented one.
        if "bottleneck_s" in d and "bottleneck_joint" in d and d.get(
            "diagnostic_valid",False
        ):
            s=float(d["bottleneck_s"])
            joint=int(d["bottleneck_joint"])
            if 0<=s<=1 and 0<=joint<7:
                feedback={"s":s,"joint":joint,"family":d.get("bottleneck_family"),
                          "parent_proposal":elite.proposal}
    # During initial exploration all arms use identically distributed
    # perturbations. The adaptive arms diverge only from observed
    # retimer feedback/selection after INIT_BATCH.
    move=bump_direction(rng,s=feedback["s"] if feedback else None,
                        joint=feedback["joint"] if feedback else None,
                        focused=feedback is not None)
    target=parent+amplitude*move
    target[:3]=base[:3]
    target[-3:]=base[-3:]
    return target,{"radius_rad":amplitude,
                   "parent_proposal":None if elite is None else elite.proposal,
                   "feedback":feedback,"direction_type":
                       "measured_timeopt_bottleneck" if feedback else
                       "global_random",
                   "used_timeopt_for_parent_selection":elite is not None}


def search(base:np.ndarray,oracle:Callable[[np.ndarray,int],dict],
           *,arm:str,task_id:str,seed:int=20261109,
           budget:int=128,max_wall_s:float|None=None)->dict:
    import time
    if arm not in ARMS or budget not in BUDGETS:
        raise ValueError("unregistered arm or oracle evaluation budget")
    base=np.asarray(base,dtype=float)
    if base.shape!=(CONTROL_COUNT,7):
        raise ValueError("expected clamped 22-control base geometry")
    rng=np.random.default_rng(rng_seed(task_id,seed,arm))
    archive:list[Candidate]=[]
    ledger=[]
    start=time.perf_counter()
    statuses={}
    for index in range(budget):
        if max_wall_s is not None and time.perf_counter()-start>=max_wall_s:
            break
        proposal,reason=propose(base,archive,arm=arm,step=index,rng=rng)
        novelty=_distinct(proposal,archive)
        if novelty<1e-6:
            # This remains a counted proposal with no external oracle
            # call, not a free extra chance for the adaptive method.
            reply={"status":"DUPLICATE","reason":"identical frozen path geometry"}
        else:
            reply=oracle(proposal,index+1)
        score=_score(reply)
        cand=Candidate(proposal,reply.get("status","ERROR"),reply,
                       index+1,novelty,score)
        archive.append(cand)
        row={"proposal":index+1,"path_sha256":path_record(proposal)["path_sha256"],
             "status":cand.status,"score":cand.cost,"novelty":novelty,
             "duration_s":reply.get("duration_s"),
             "diagnostics":reply.get("diagnostics"),
             "worker_wall_ms":reply.get("worker_wall_ms"),
             "subprocess_wall_ms":reply.get("subprocess_wall_ms"),
             "reason":reply.get("reason"),"elapsed_wall_s":time.perf_counter()-start,
             **reason}
        ledger.append(row)
        statuses[cand.status]=statuses.get(cand.status,0)+1
    screened=[v for v in archive if v.status=="FEASIBLE_SCREENED"]
    best=min(screened,key=lambda v:v.cost) if screened else None
    result={"schema_version":1,"task_id":task_id,"arm":arm,
            "seed":seed,"budget":budget,"proposals_attempted":len(ledger),
            "oracle_requests":sum(r["status"]!="DUPLICATE" for r in ledger),
            "total_wall_s":time.perf_counter()-start,
            "first_preliminary_feasible_proposal":next((
                c.proposal for c in archive if c.status=="FEASIBLE_SCREENED"
            ),None),
            "best_screened_duration_s":best.cost if best is not None else None,
            "best_path_sha256":path_record(best.controls)["path_sha256"] if best else None,
            "best_controls":best.controls.tolist() if best else None,
            "status_counts":statuses,"ledger":ledger,
            "independent_qualified_success":False,
            "qualification":"NOT_EXECUTED; worker online screen is not independent physical/Scaled95 qualification",
            "claim":"Retiming-guided proposal policy only; no global optimality or continuous-time guarantee"}
    return result
