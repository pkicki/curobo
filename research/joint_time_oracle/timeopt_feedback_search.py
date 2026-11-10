# SPDX-License-Identifier: Apache-2.0
"""Global TimeOpt-feedback geometric exploration (research-only).

An identical deterministic first-generation candidate set is used for
all arms. In later generations, blind samples parents without reference
to TimeOpt results; rank_only selects elites by retimed duration or
structured infeasibility cost; feedback also directs broad geometric
perturbations towards reported limiting path regions. A fixed fraction
of proposals always comes from a global independent exploration arm.

A positive result requires independent dynamic/collision qualification,
matched complete planning wall-time, a native cuRoboV2-Fast baseline,
and a source-disjoint holdout. An oracle may return uncertified scores;
they guide research proposals but DO NOT establish feasibility. Errors,
timeouts and unsupported models are NOT evidence of infeasibility.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .retimer_oracle import RESPONSE_STATUS, digest

ARMS=("blind","rank_only","feedback")
BUDGET=128
INITIAL=16
GLOBAL_FRACTION=4
SEED=20261110
RADII=(.25,.5,.8,1.2,1.6)
BEAM=12


@dataclass(frozen=True)
class Record:
    index:int
    controls:np.ndarray
    cost:float
    status:str
    bottleneck_s:float|None
    duration_s:float|None
    novelty:float


def seed_for(task_id:str,iteration:int,seed:int=SEED)->int:
    payload=f"{task_id}|{iteration}|{seed}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:16],16)


def novelty(controls:np.ndarray,archive:list[Record])->float:
    if not archive:
        return 1.
    return float(min(np.max(np.linalg.norm(controls-r.controls,axis=1))
                     for r in archive))


def oracle_cost(result:dict)->float:
    status=result.get("status")
    if status=="FEASIBLE_PRELIMINARY":
        duration=float(result.get("duration_s",float("nan")))
        return duration if np.isfinite(duration) and duration>0 else float("inf")
    if status in ("INFEASIBLE_CERTIFIED","INFEASIBLE_HEURISTIC"):
        gap=result.get("infeasibility_score")
        if gap is not None and np.isfinite(float(gap)) and float(gap)>=0:
            return 100.+min(float(gap),100.)
    return float("inf")


def perturb(base:np.ndarray,*,radius:float,center:float,rng:np.random.Generator,
            two_lobes:bool=True)->np.ndarray:
    """Broad, coordinated changes, not the previously failed local LP gradient."""
    base=np.asarray(base,dtype=float)
    if base.shape!=(22,7) or not 0<=center<=1 or radius<=0:
        raise ValueError("invalid research spline/perturbation")
    t=np.linspace(0,1,22)
    delta=np.zeros_like(base)
    vec=rng.normal(size=7);vec/=max(np.linalg.norm(vec),1e-12)
    envelope=np.exp(-.5*((t-center)/.30)**2)
    delta+=radius*envelope[:,None]*vec
    if two_lobes:
        second=rng.normal(size=7)
        second/=max(np.linalg.norm(second),1e-12)
        center2=float(np.clip(center+rng.uniform(-.36,.36),.15,.85))
        delta+=.5*radius*np.exp(-.5*((t-center2)/.24)**2)[:,None]*second
    delta[:3]=0.;delta[-3:]=0.
    return base+delta


def parent_index(arm:str,archive:list[Record],task:str,iteration:int)->int:
    if not archive:
        raise ValueError("cannot select without archive")
    rng=np.random.default_rng(seed_for(task,iteration))
    if arm=="blind":
        return int(rng.integers(len(archive)))
    eligible=[r for r in archive if np.isfinite(r.cost)]
    if not eligible:
        return int(rng.integers(len(archive)))
    # Penalize redundant solutions; novelty is measured from the
    # actual evaluated geometries. Preserve a diverse elite set.
    order=sorted(eligible,key=lambda r:(r.cost-.15*r.novelty,r.index))[:BEAM]
    return int(order[int(rng.integers(len(order)))].index)


def proposal(base:np.ndarray,archive:list[Record],*,task:str,
             iteration:int,arm:str)->tuple[np.ndarray,dict]:
    if arm not in ARMS or iteration<0:
        raise ValueError("unregistered arm or index")
    rng=np.random.default_rng(seed_for(task,iteration))
    radius=RADII[(iteration//3)%len(RADII)]
    if iteration<INITIAL or iteration%GLOBAL_FRACTION==0:
        parent=base
        choice=None
        mode="GLOBAL_INDEPENDENT"
        center=float(rng.uniform(.18,.82))
    else:
        choice=parent_index(arm,archive,task,iteration)
        parent=archive[choice].controls
        selected=archive[choice]
        if arm=="feedback" and selected.bottleneck_s is not None:
            center=float(np.clip(selected.bottleneck_s,.12,.88))
            mode="RETIME_BOTTLENECK_GUIDED"
        else:
            center=float(rng.uniform(.18,.82))
            mode="RANKED_GLOBAL_PERTURB" if arm=="rank_only" else "BLIND_GLOBAL_PERTURB"
    candidate=perturb(parent,radius=radius,center=center,rng=rng)
    np.testing.assert_array_equal(candidate[:3],base[:3])
    np.testing.assert_array_equal(candidate[-3:],base[-3:])
    return candidate,{"mode":mode,"center_s":center,"radius_rad":radius,
                      "parent_index":choice}


def run(task_id:str,base_controls:np.ndarray,
        evaluate:Callable[[np.ndarray,int],dict],*,
        arm:str,budget:int=BUDGET)->dict:
    """Run with a REVISION-PINNED real callback or mock in tests.

    Caller's evaluation must return schema-versioned retimer outcome and
    independently verify candidate geometry. This controller never
    fabricates a certified trajectory or a native comparison.
    """
    if arm not in ARMS or budget<INITIAL or budget>BUDGET:
        raise ValueError("unregistered arm or proposal budget")
    base=np.asarray(base_controls,dtype=float)
    if base.shape!=(22,7) or not np.isfinite(base).all():
        raise ValueError("expected finite 22x7 baseline")
    archive:list[Record]=[]
    ledger=[]
    for iteration in range(budget):
        controls,meta=proposal(base,archive,task=task_id,iteration=iteration,arm=arm)
        result=evaluate(controls,iteration)
        if result.get("status") not in RESPONSE_STATUS:
            raise ValueError("callback returned unsupported engine status")
        cost=oracle_cost(result)
        changed=float(novelty(controls,archive))
        bottleneck=result.get("bottleneck_s")
        if bottleneck is not None:
            bottleneck=float(bottleneck)
            if not np.isfinite(bottleneck) or not 0<=bottleneck<=1:
                raise ValueError("invalid retimer bottleneck")
        time_law=result.get("duration_s") if result["status"]=="FEASIBLE_PRELIMINARY" else None
        point=Record(iteration,controls.copy(),cost,result["status"],
                     bottleneck,time_law,changed)
        archive.append(point)
        ledger.append({"index":iteration,"arm":arm,"status":point.status,
                       "duration_s":point.duration_s,
                       "oracle_cost":point.cost if np.isfinite(point.cost) else None,
                       "novelty_rad":changed,
                       "bottleneck_s":bottleneck,
                       "path_sha256":hashlib.sha256(np.ascontiguousarray(
                           controls).tobytes()).hexdigest(),
                       **meta,
                       "independent_physical_certified":False})
    return {"schema_version":1,"arm":arm,"task_id":task_id,
            "budget":budget,"initial":INITIAL,
            "oracle_evaluations":len(ledger),
            "ledger":ledger,"claim":"candidate search only; independent physical+Scaled95/S32768 qualification pending"}


def main():
    """Smoke generated proposals without claiming a live K107 backend."""
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    if not a.smoke:
        p.error("real scored runs require a model-parity-qualified retimer callback")
    base=np.zeros((22,7))
    result=run("smoke",base,lambda *_:{"status":"UNSUPPORTED_MODEL"},
               arm="feedback",budget=INITIAL)
    print(json.dumps({"oracle_evaluations":result["oracle_evaluations"],
                      "status":"RESEARCH_SMOKE_NOT_SCORED"}))


if __name__=="__main__":
    main()
