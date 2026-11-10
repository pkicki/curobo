# SPDX-License-Identifier: Apache-2.0
"""Retiming-in-the-loop GLOBAL geometric exploration, opt-in research only.

A scored comparison with matched oracle-call and complete wall-time budgets:
  global_unguided  : frozen LP-blind global path proposals; first oracle W.
  timeopt_ranked   : SAME frozen proposals, choose shortest oracle W.
  timeopt_guided   : SAME initial proposals, then global mutations of a
                     diverse archive selected by TimeOpt feasibility/duration.

No local LP-gradient (negative previous result) and no historical
successful W controls. No oracle-candidate is independently qualified
until a subsequent exact-geometry, same-payload sampled physical audit.
All three arms keep the original actuator bounds. Timing calls include
the foreign-process communication cost; captured production K107
latency is NOT measured by this prototype.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np

from .cohort_geometry import QuinticGeometry
from .geometry_feasibility_scene import IndependentGeometryAuditor
from .payload_model_parity import independent_payload_model
from .payload_timeopt_contract import (
    ARMS,OUT,Payload,PROPOSAL_BUDGETS,SEED,canonical_sha,immutable_json,source_sha,
)
from .torque_global_bank import proposal_controls


@dataclass
class ArchiveItem:
    controls:np.ndarray
    duration_s:float
    proposal:int
    digest:str
    world_clearance_m:float
    self_clearance_m:float


def fixed_endpoint_base(start,goal,*,control_count:int=22)->np.ndarray:
    """Shared zero-tangent/curvature initialization, no source W knowledge."""
    a=np.asarray(start,dtype=float);b=np.asarray(goal,dtype=float)
    if a.shape!=(7,) or b.shape!=(7,) or control_count<8:
        raise ValueError("invalid source endpoints")
    s=np.linspace(0,1,control_count)
    # Smoothstep has zero derivative and curvature at ends; exact
    # clamped end controls are repeated separately.
    smooth=10*s**3-15*s**4+6*s**5
    c=a[None]+smooth[:,None]*(b-a)[None]
    c[:3]=a;c[-3:]=b
    return c


def guided_candidate(base:np.ndarray,archive:list[ArchiveItem],k:int,
                     *,seed:int)->tuple[np.ndarray,dict]:
    """Global, not local sensitivity: mutate feasible elite and retain diversity."""
    rng=np.random.default_rng(int(hashlib.sha256(
        f"guided|{seed}|{k}".encode()).hexdigest()[:15],16))
    if not archive:
        raise ValueError("cannot guide without preliminary TimeOpt survivor")
    selected=archive[k%min(len(archive),5)]
    parent=selected.controls
    c=parent.copy()
    grid=np.linspace(0,1,len(c))
    rad=(.18,.32,.55,.85,1.20)[k%5]
    direction=rng.normal(size=7)
    direction/=max(float(np.linalg.norm(direction)),1e-12)
    center=float(rng.uniform(.25,.75))
    sigma=float(rng.uniform(.18,.42))
    envelope=np.exp(-.5*((grid-center)/sigma)**2)
    c+=rad*envelope[:,None]*direction[None]
    # 25% of mutations are long jumps towards another archive member
    # rather than a small neighborhood search.
    if k%4==0 and len(archive)>1:
        peer=archive[(k+1)%len(archive)]
        c=.5*c+.5*peer.controls
    c[:3]=base[:3];c[-3:]=base[-3:]
    return c,{"proposal_source":"TIMEOPT_DURATION_SELECTED_GLOBAL_MUTATION",
              "parent_proposal":selected.proposal,
              "parent_duration_s":selected.duration_s,
              "mutation_radius_rad":rad}


def independent_geometry_screen(controls,scene:str|None,payload:Payload,
                                *,clearance:float=.005)->dict:
    """Robot/scene sphere audit; payload collision shape NOT included."""
    from .payload_model_parity import independent_payload_auditor
    auditor=independent_payload_auditor(payload,scene)
    values=QuinticGeometry(controls).evaluate(np.linspace(0,1,129))[0]
    result=auditor.check(values,min_clearance_m=clearance)
    gravity=auditor.gravity(values[[0,-1]])
    bounds=auditor.nominal_effort*.95
    lo,hi=bounds
    static_admissible=bool(np.all(gravity>=lo[None]) and np.all(gravity<=hi[None]))
    return {"robot_geometry_pass":bool(result["sampled_acceptable"]),
            "endpoint_payload_gravity_pass":static_admissible,
            "world_clearance_m":float(result["world_clearance_m"]),
            "self_clearance_m":float(result["self_clearance_m"]),
            "max_payload_endpoint_gravity_nm":float(np.max(np.abs(gravity))),
            "payload_shape_collision_qualified":False,
            "qualified_geometry_screen":bool(result["sampled_acceptable"] and static_admissible)}


def update_diverse_archive(archive:list[ArchiveItem],item:ArchiveItem,
                           *,max_size:int=8,novelty_rad:float=.15)->list[ArchiveItem]:
    # Diversity is evaluated in geometric control space only. This is a
    # proposal policy, NOT a proof of distinct collision homotopies.
    new=[x for x in archive if np.linalg.norm(x.controls-item.controls)>novelty_rad]
    new.append(item)
    return sorted(new,key=lambda x:x.duration_s)[:max_size]


def search(*,task:dict,base:np.ndarray,arm:str,
           oracle:Callable[[np.ndarray,int],dict],seed:int=SEED,
           budget:int=128,time_budget_s:float=30.0,
           precomputed_bank:np.ndarray|None=None,
           screen:Callable[[np.ndarray],dict]|None=None)->dict:
    if arm not in ARMS or budget not in PROPOSAL_BUDGETS or time_budget_s<=0:
        raise ValueError("unregistered policy/budget")
    base=np.asarray(base,dtype=float)
    if base.shape!=(22,7):
        raise ValueError("invalid 22x7 starting geometry")
    if precomputed_bank is None:
        precomputed_bank,_=proposal_controls(base,count=budget,
               seed=int(hashlib.sha256(f"shared|{task['source_task_id']}|{seed}".encode()).hexdigest()[:15],16))
    bank=np.asarray(precomputed_bank,dtype=float)
    if bank.shape!=(budget,22,7):
        raise ValueError("common bank shape mismatch")
    bank_sha=canonical_sha(bank.tolist())
    archive=[];records=[];start=time.perf_counter()
    best=None
    feedback_proposals=0
    for i in range(budget):
        if time.perf_counter()-start>=time_budget_s:
            break
        proposal=i+1
        # All arms use exactly the same blind proposal stream until
        # guided has a scored TimeOpt candidate. Without a survivor,
        # guided cannot manufacture feedback and remains global random.
        use_feedback=(arm=="timeopt_guided" and bool(archive) and i>=16 and i%2==0)
        if use_feedback:
            controls,detail=guided_candidate(base,archive,i,seed=seed)
            feedback_proposals+=1
        else:
            controls=bank[i].copy()
            detail={"proposal_source":"SHARED_LP_BLIND_GLOBAL_BANK"}
        np.testing.assert_array_equal(controls[:3],base[:3])
        np.testing.assert_array_equal(controls[-3:],base[-3:])
        digest=canonical_sha(controls.tolist())
        event={"proposal":proposal,"geometry_sha256":digest,**detail,
               "status":"PENDING"}
        stamp=time.perf_counter()
        try:
            geometry_screen=(screen(controls) if screen is not None
                             else independent_geometry_screen(
                                controls,task["scene"],
                                Payload(**task["payload"])))
            event["geometry_screen"]=geometry_screen
            if not geometry_screen.get("qualified_geometry_screen",False):
                event["status"]="GEOMETRY_SCREEN_REJECTED"
            else:
                scored=oracle(controls,proposal)
                event["oracle"]={k:v for k,v in scored.items()
                    if k!="squared_speed_coefficients"}
                if scored.get("status")=="CANDIDATE":
                    duration=float(scored["duration_s"])
                    if not np.isfinite(duration) or duration<=0:
                        raise ValueError("nonpositive TimeOpt candidate duration")
                    item=ArchiveItem(controls.copy(),duration,proposal,digest,
                        geometry_screen["world_clearance_m"],
                        geometry_screen["self_clearance_m"])
                    archive=update_diverse_archive(archive,item)
                    if best is None or item.duration_s<best.duration_s or (
                        arm=="global_unguided" and best is None):
                        if arm!="global_unguided" or best is None:
                            best=item
                    event["status"]="TIMEOPT_PRELIMINARY"
                else:
                    event["status"]="TIMEOPT_REJECTED_OR_NUMERICAL_ERROR"
        except Exception as exc:
            event["status"]="EVALUATION_ERROR"
            event["error"]=repr(exc)
        event["candidate_wall_s"]=time.perf_counter()-stamp
        event["cumulative_wall_s"]=time.perf_counter()-start
        records.append(event)
    winner=(None if best is None else
            {"proposal":best.proposal,"duration_s":best.duration_s,
             "geometry_sha256":best.digest,
             "world_clearance_m":best.world_clearance_m,
             "self_clearance_m":best.self_clearance_m,
             "controls":best.controls.tolist()})
    return {"schema":1,"source_case":task["case_id"],
            "source_task_id":task["source_task_id"],"arm":arm,"seed":seed,
            "registered_budget":budget,"time_budget_s":time_budget_s,
            "shared_bank_sha256":bank_sha,"proposals_attempted":len(records),
            "timeopt_feedback_proposals":feedback_proposals,
            "total_wall_s":time.perf_counter()-start,
            "preliminary_best":winner,"ledger":records,
            "qualification":"NONE until exact independent five-family+payload collision audit",
            "conclusion":"feedback gain requires comparison at MATCHED wall time and oracle calls"}


def run_case(*,task:dict,arm:str,output:Path,top_repo:Path,
             top_python:Path,top_sha:str,original_effort:np.ndarray,
             budget:int=128,time_budget_s:float=30.,study_root:Path=OUT):
    from .foreign_toppra_newton import solve_foreign
    output=Path(output)
    from .payload_parity_qualify import verify_task_payload_receipt
    verify_task_payload_receipt(task)
    if task.get("split") not in ("development","confirmation","holdout"):
        raise ValueError("unknown source-level split")
    if task["split"]!="development":
        from .payload_joint_freeze import verify
        verify(study_root)
    if task.get("admission")!="ENDPOINT_ADMITTED":
        raise ValueError("invalid payload endpoint gravity admission")
    if budget not in PROPOSAL_BUDGETS:
        raise ValueError("unregistered search proposal budget")
    if output.exists():
        raise ValueError("existing scored arm cannot be overwritten")
    output.mkdir(parents=True,exist_ok=False)
    base=fixed_endpoint_base(task["start"],task["goal"])
    bank,log=proposal_controls(base,count=budget,
        seed=int(hashlib.sha256(f"shared|{task['source_task_id']}|{SEED}".encode()).hexdigest()[:15],16))
    np.savez_compressed(output/"shared_proposals.npz",
                        base=base,proposals=bank)
    payload=Payload(**task["payload"])

    def solve_one(controls:np.ndarray,number:int)->dict:
        folder=output/f"proposal_{number:04d}"
        folder.mkdir()
        sample=folder/"candidate_geometry.npz"
        np.savez_compressed(sample,controls=controls)
        return solve_foreign(source_path=sample,lane=None,payload=payload,
                             effort=original_effort,case_id=task["case_id"],
                             topp_repo=top_repo,topp_python=top_python,
                             output=folder,expected_topp_sha=top_sha)

    report=search(task=task,base=base,arm=arm,oracle=solve_one,
                  seed=SEED,budget=budget,time_budget_s=time_budget_s,
                  precomputed_bank=bank)
    report["shared_proposals_sha256"]=source_sha(output/"shared_proposals.npz")
    immutable_json(output/"search_result.json",report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-json",type=Path,required=True)
    p.add_argument("--arm",choices=ARMS,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--study-root",type=Path,default=OUT,
                   help="Frozen study registry; mandatory for confirmation/holdout.")
    p.add_argument("--topp-repo",type=Path,required=True)
    p.add_argument("--topp-python",type=Path,required=True)
    p.add_argument("--topp-sha",required=True)
    p.add_argument("--effort-bounds-npz",type=Path,required=True)
    p.add_argument("--budget",type=int,default=128)
    p.add_argument("--seconds",type=float,default=30.)
    args=p.parse_args()
    task=json.loads(args.task_json.read_text())
    with np.load(args.effort_bounds_npz) as f:
        effort=np.asarray(f["effort_bounds"])
    result=run_case(task=task,arm=args.arm,output=args.out,
                    top_repo=args.topp_repo,top_python=args.topp_python,
                    top_sha=args.topp_sha,original_effort=effort,
                    budget=args.budget,time_budget_s=args.seconds,
                    study_root=args.study_root)
    print(json.dumps({k:v for k,v in result.items()
                      if k not in ("ledger","preliminary_best")},indent=2))


if __name__=="__main__":
    main()
