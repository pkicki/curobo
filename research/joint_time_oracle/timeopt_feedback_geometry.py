# SPDX-License-Identifier: Apache-2.0
"""TimeOpt-feedback-guided *geometry* search with matched unguided controls.

This is the user-requested coupling: the NEXT geometric perturbation is
conditioned on a PREVIOUS K107 feasibility/duration result, rather than
using TimeOpt only to time the final path. Two controls:
- random_global: identical common proposal prefix + independent proposals;
- oracle_rank_only: same unguided proposals and retiming calls, but picks
  the best TimeOpt-scored incumbent at the end;
- timeopt_feedback: same common prefix, then a diversity-aware elite
  population selected from ACTUAL K107 outcomes, with periodic global
  exploration. No local LP gradients or LP gate are used.

All arms use the SAME numerical budgets, initial candidates, K107 backend
and cheap geometry screen. Report both oracle count and elapsed compute.
K107 online screen is NOT independent physical qualification.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np

from .geometry_feasibility_core import perturb_controls, smoothstep_controls
from .torque_global_bank import proposal_controls

ARMS=("random_global","oracle_rank_only","timeopt_feedback")
BUDGET=128
PREFIX=16
ELITES=6
GLOBAL_REFRESH=4
SEED=20261109


def base_seed(task:dict)->int:
    return int(hashlib.sha256((str(SEED)+":"+task["case_id"]).encode()).hexdigest()[:16],16)


def outcome_quality(result:dict)->float:
    """Rank using ONLY online K107 information, never holdout witness data."""
    if result.get("status")=="ONLINE_SCREEN_PASS" and result.get("online_feasible") is True:
        t=result.get("duration_s")
        if isinstance(t,(float,int)) and math.isfinite(t) and t>0:
            return 10. - math.log1p(t)
    if result.get("status")=="ONLINE_SCREEN_REJECTED":
        ratios=result.get("online_utilizations",{})
        values=[float(v) for v in ratios.values()
                if isinstance(v,(float,int)) and math.isfinite(v) and v>0]
        if values:
            return 1.-math.log1p(max(values))
    # Numerical failure is NOT path infeasibility: use weak rank only,
    # preserving periodic global exploration.
    if result.get("status") in ("TOPPRA_SEED_FAILED","ERROR","EVALUATION_ERROR"):
        return -10.
    return -20.


@dataclass(frozen=True)
class Scored:
    index:int
    controls:np.ndarray
    quality:float
    status:str
    wall_s:float


def _parent_score(candidate:Scored,selected:list[Scored])->float:
    if not selected:return candidate.quality
    distance=min(float(np.max(np.linalg.norm(
        candidate.controls[3:-3]-other.controls[3:-3],axis=1)))
        for other in selected)
    return candidate.quality+.10*min(distance,2.)


def elite_pool(history:list[Scored],count:int=ELITES)->list[Scored]:
    """Greedy score+diversity, deterministic tie order and finite controls."""
    eligible=[x for x in history if np.isfinite(x.quality) and
              np.isfinite(x.controls).all() and x.status!="CHEAP_GEOMETRY_REJECT"]
    chosen=[]
    while eligible and len(chosen)<count:
        x=max(eligible,key=lambda p:(_parent_score(p,chosen),-p.index))
        chosen.append(x)
        eligible.remove(x)
    return chosen


def propose_feedback(history:list[Scored],rng:np.random.Generator,
                     global_controls:np.ndarray,index:int)->tuple[np.ndarray,dict]:
    """70–75% exploitation, at least every fourth proposal is fresh global."""
    if index<PREFIX or index%GLOBAL_REFRESH==0:
        return global_controls[index].copy(),{"proposal_mode":"FRESH_GLOBAL","parent":None}
    elites=elite_pool(history)
    if not elites:
        return global_controls[index].copy(),{"proposal_mode":"NO_ORACLE_ELITE_GLOBAL","parent":None}
    parent=elites[int(rng.integers(0,len(elites)))]
    radius=(.10,.20,.35,.55,.85)[index%5]
    proposal=perturb_controls(parent.controls,rng,amplitude=radius)
    proposal[:3]=global_controls[index,:3]
    proposal[-3:]=global_controls[index,-3:]
    return proposal,{"proposal_mode":"TIMEOPT_ELITE_MUTATION","parent":parent.index+1,
                     "radius_rad":radius,"parent_oracle_quality":parent.quality}


def search(task:dict,*,arm:str,evaluate:Callable[[np.ndarray,int],dict],
           cheap_screen:Callable[[np.ndarray],dict]|None=None,
           budget:int=BUDGET,prefix:int=PREFIX)->dict:
    if arm not in ARMS or budget!=BUDGET or prefix!=PREFIX:
        raise ValueError("unregistered arm, budget or prefix")
    source=np.asarray(smoothstep_controls(np.asarray(task["start"]),
                                          np.asarray(task["goal"]),count=22))
    rng=np.random.default_rng(base_seed(task))
    global_controls,_=proposal_controls(source,count=budget,seed=base_seed(task))
    history=[]
    entries=[]
    started=time.perf_counter()
    for i in range(budget):
        if arm=="timeopt_feedback":
            candidate,meta=propose_feedback(history,rng,global_controls,i)
        else:
            candidate=global_controls[i].copy()
            meta={"proposal_mode":"FROZEN_GLOBAL","parent":None}
        if not np.array_equal(candidate[:3],source[:3]) or not np.array_equal(
                candidate[-3:],source[-3:]):
            raise ValueError("geometric endpoint contract changed")
        stage=time.perf_counter()
        screening=cheap_screen(candidate) if cheap_screen else {"sampled_acceptable":True}
        if not screening.get("sampled_acceptable",False):
            result={"status":"CHEAP_GEOMETRY_REJECT"}
        else:
            try:
                result=evaluate(candidate,i)
            except Exception as error:
                result={"status":"EVALUATION_ERROR","type":type(error).__name__,
                        "reason":str(error)[:400]}
        wall=time.perf_counter()-stage
        score=outcome_quality(result)
        state=Scored(i,candidate.copy(),score,str(result["status"]),wall)
        history.append(state)
        entries.append({"proposal":i+1,**meta,"screen":screening,
                        "retimer_status":result["status"],
                        "oracle_duration_s":result.get("duration_s"),
                        "oracle_score":score,"oracle_online_feasible":
                           result.get("online_feasible") is True,
                        "candidate_wall_s":wall,"cumulative_wall_s":time.perf_counter()-started,
                        "oracle_response":{k:v for k,v in result.items()
                                if k not in ("coefficients","knots")},
                        "controls_sha256":hashlib.sha256(
                           np.ascontiguousarray(candidate).tobytes()).hexdigest()})
    best=elite_pool(history,1)
    return {"schema_version":1,"task_sha256":task["task_sha256"],
            "arm":arm,"budget":budget,"prefix":prefix,
            "full_search_wall_s":time.perf_counter()-started,
            "oracle_calls":sum(x["retimer_status"]!="CHEAP_GEOMETRY_REJECT" for x in entries),
            "online_feasible_candidates":sum(x["oracle_online_feasible"] for x in entries),
            "best_candidate":best[0].index+1 if best else None,
            "ledger":entries,
            "claim":"Online retimer scoring only. Independent payload-model Pinocchio and collision audit still required.",
            "policy":"feedback uses K107 duration/feasibility to select parents; random controls never use K107 for proposal choices"}
