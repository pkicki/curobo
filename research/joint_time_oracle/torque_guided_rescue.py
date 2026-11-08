# SPDX-License-Identifier: Apache-2.0
"""Matched-budget torque-LP-dual guided geometric feasibility search.

Designed for the THREE frozen source-task counterexamples only. Neither
baseline is initialized from the known feasible alternative. Both use the
same least-squares quintic approximation of the original *nominal* native
cubic path. This approximation's geometric error and LP status are recorded.

Arms: "dual" (LP envelope gradient), "random_local" (same bottleneck support
with random direction), "random_global" (all free control points).
The first valid constructive timing is a CANDIDATE, not an independent
qualification; run cohort_validate and independent adaptive replay next.

All proposal counts and CPU wall times include local LP, collision and
gradient work. No result is a continuous-time feasibility certificate.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import BSpline

from .cohort_geometry import NativeGeometry, QuinticGeometry
from .geometry_feasibility_core import (
    MIN_GEOMETRY_DEVIATION_RAD, path_distance, sha256_file,
)
from .geometry_feasibility_discover import constructive_timing, source_native, source_tasks
from .geometry_feasibility_scene import IndependentGeometryAuditor
from .torque_active import read_profile, torque_utilization
from .torque_dual_guidance import (
    AlignedPointDynamics, PointLP, normalized_bottlenecks,
    normalized_descent_direction, solve_point_lp,
)
from .cohort_retime import ROOT

ROOTS = {
    "trap_cluttered_08": (
        "geometry_trap_source_20261108",
        "geometry_feasibility_discovery_20261108"),
    "confirm_geom_table_07": (
        "geometry_trap_confirm_20261109",
        "geometry_feasibility_confirmation_20261109"),
    "confirm_geom_cluttered_02": (
        "geometry_trap_confirm_20261109",
        "geometry_feasibility_confirmation_20261109"),
}
SCENARIO="j1234_f0300"
ARMS=("dual","random_local","random_global")
BUDGETS=(16,32,64,128)
RADII=(.16,.30,.50,.80)
OUTPUT=ROOT/"research/joint_time_oracle/results/torque_lp_guided_rescue_20261108"


@dataclass
class Candidate:
    controls: np.ndarray
    score: float
    lp_max: float
    gravity_peak: float
    clearance: dict
    bottlenecks: list[dict]
    deviation_rad: float
    status: str


def fit_nominal_quintic(native:NativeGeometry,q0,q1,*,count:int=22)->tuple[np.ndarray,dict]:
    """Least squares fit for equal-arm initialization; no source W geometry."""
    if count<8:
        raise ValueError("quintic controls too few")
    q0,q1=np.asarray(q0,dtype=float),np.asarray(q1,dtype=float)
    s=np.linspace(0,1,513)
    target=np.asarray(native.evaluate_trimmed(s)[0])
    knots=np.r_[np.zeros(5),np.linspace(0,1,count-4),np.ones(5)]
    basis=BSpline(knots,np.eye(count),5,extrapolate=False)(s)
    controls=np.zeros((count,len(q0)))
    controls[:3]=q0
    controls[-3:]=q1
    free=basis[:,3:-3]
    rhs=target-basis[:,:3]@controls[:3]-basis[:,-3:]@controls[-3:]
    controls[3:-3]=np.linalg.lstsq(free,rhs,rcond=1e-12)[0]
    approx=np.asarray(QuinticGeometry(controls).evaluate(s)[0])
    discrepancy=float(np.max(np.linalg.norm(approx-target,axis=1)))
    return controls,{"maximum_same_progress_fit_deviation_rad":discrepancy,
                     "squared_residual":float(np.sum((approx-target)**2)),
                     "count":count,
                     "fit_note":"approximate original path for SHARED search initialization; original LP obstruction remains on EXACT native cubic"}


def local_support(count:int,progress:float)->np.ndarray:
    knots=np.r_[np.zeros(5),np.linspace(0,1,count-4),np.ones(5)]
    basis=BSpline(knots,np.eye(count),5,extrapolate=False)
    s=np.array([float(progress)])
    influence=np.abs(basis(s)[0])+np.abs(basis(s,nu=1)[0])+np.abs(basis(s,nu=2)[0])
    active=influence>1e-9
    active[:3]=False
    active[-3:]=False
    return active


def random_direction(rng:np.random.Generator,shape:tuple[int,int],
                     support:np.ndarray)->np.ndarray:
    delta=rng.normal(size=shape)
    delta[~support]=0
    return normalized_step(delta)


def normalized_step(direction:np.ndarray)->np.ndarray:
    out=np.asarray(direction,dtype=float).copy()
    out[:3]=0
    out[-3:]=0
    peak=float(np.max(np.linalg.norm(out,axis=1)))
    return out/max(peak,1e-12)


def score_metrics(lp:float,gravity:float,clearance:dict,
                  deviation:float)->float:
    if not np.isfinite(lp) or not np.isfinite(gravity):
        return float("inf")
    penalty=5.*max(gravity-.88,0.)
    penalty+=20.*max(float(clearance.get("position_violation_rad") or 0.),0.)
    for name in ("world_clearance_m","self_clearance_m"):
        value=clearance.get(name)
        if value is not None:
            penalty+=25.*max(.005-float(value),0.)
    penalty+=max(MIN_GEOMETRY_DEVIATION_RAD-deviation,0.)
    return float(max(lp,0.)+penalty)


class Evaluator:
    def __init__(self,task:dict,profile:dict,original:NativeGeometry,
                 *,grid:int=33):
        if grid<9:
            raise ValueError("LP progress grid too small")
        self.task=task
        self.auditor=IndependentGeometryAuditor(task["scene"])
        self.rnea=AlignedPointDynamics()
        self.bounds=np.stack((profile["effort_lower"],profile["effort_upper"]))
        self.original=np.asarray(original.evaluate_trimmed(np.linspace(0,1,513))[0])
        self.grid=np.linspace(.01,.99,grid)
        self.evaluations=0

    def evaluate(self,controls:np.ndarray)->Candidate:
        geom=QuinticGeometry(controls)
        coeff=np.stack([self.rnea.coefficients(geom,float(s)) for s in self.grid],axis=1)
        bottlenecks=normalized_bottlenecks(coeff,self.bounds,top_k=3)
        lp=float(bottlenecks[0]["expansion"])
        q=np.asarray(geom.evaluate(np.linspace(0,1,129))[0])
        clearance=self.auditor.check(q,min_clearance_m=.005)
        grav=float(np.max(torque_utilization(self.auditor.gravity(q),self.bounds*.95)))
        deviation=path_distance(q,self.original)
        value=score_metrics(lp,grav,clearance,deviation)
        self.evaluations+=1
        return Candidate(controls.copy(),value,lp,grav,
                         {k:v for k,v in clearance.items() if k!="gravity_torques"},
                         bottlenecks,deviation,
                         "SCREEN_PASS" if (lp<=1e-7 and grav<=.88 and
                             clearance["sampled_acceptable"] and
                             deviation>=MIN_GEOMETRY_DEVIATION_RAD) else "UNQUALIFIED")

    def suggest(self,incumbent:Candidate,arm:str,rng:np.random.Generator,
                *,iteration:int)->tuple[np.ndarray,dict]:
        bottleneck=incumbent.bottlenecks[iteration%len(incumbent.bottlenecks)]
        progress=float(self.grid[int(bottleneck["index"])])
        support=local_support(len(incumbent.controls),progress)
        detail={"bottleneck_s":progress,
                "dual_lp_expansion":float(bottleneck["expansion"]),
                "method":arm,"radius":RADII[iteration%len(RADII)]}
        if arm=="dual" and bottleneck["expansion"]>1e-9:
            # The LP dual remains fixed during each finite difference
            # ("envelope theorem" locally). Reevaluate the real LP afterward.
            a,m,g=self.rnea.coefficients(
                QuinticGeometry(incumbent.controls),progress)
            lp=solve_point_lp(a,m,g,self.bounds)
            if lp.status=="SOLVED" and lp.expansion>1e-9:
                derivative=self.rnea.dual_gradient(
                    incumbent.controls,progress,lp)
                direction=normalized_descent_direction(derivative)
                if direction is not None:
                    detail["direction_source"]="LP_DUAL_GRADIENT"
                    detail["gradient_norm"]=float(np.linalg.norm(derivative))
                else:
                    detail["direction_source"]="RANDOM_FALLBACK_GRADIENT_DEGENERATE"
                    direction=random_direction(rng,incumbent.controls.shape,support)
            else:
                detail["direction_source"]="RANDOM_FALLBACK_LP_INVALID"
                direction=random_direction(rng,incumbent.controls.shape,support)
        elif arm=="random_local" or arm=="dual":
            direction=random_direction(rng,incumbent.controls.shape,support)
            detail["direction_source"]="RANDOM_LOCAL"
        else:
            mask=np.ones(len(incumbent.controls),dtype=bool)
            mask[:3]=False;mask[-3:]=False
            direction=random_direction(rng,incumbent.controls.shape,mask)
            detail["direction_source"]="RANDOM_GLOBAL"
        direction=normalized_step(direction)
        target=incumbent.controls+RADII[iteration%len(RADII)]*direction
        # Endpoint position and its first two path derivatives remain fixed.
        np.testing.assert_array_equal(target[:3],incumbent.controls[:3])
        np.testing.assert_array_equal(target[-3:],incumbent.controls[-3:])
        return target,detail


def stable_seed(task:str,arm:str,seed:int)->int:
    payload=f"{task}|{arm}|{seed}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:15],16)


def is_witness(candidate:Candidate)->bool:
    return candidate.status=="SCREEN_PASS"


def search(task:dict,profile:dict,native:NativeGeometry,*,arm:str,
           budget:int=128,seed:int=20261108,grid:int=33)->tuple[dict,dict|None]:
    if arm not in ARMS or budget not in BUDGETS:
        raise ValueError("unregistered arm/budget")
    controls,fit=fit_nominal_quintic(native,task["start"],task["goal"])
    evaluator=Evaluator(task,profile,native,grid=grid)
    rng=np.random.default_rng(stable_seed(task["case_id"],arm,seed))
    archive=[]
    stamps=time.perf_counter()
    base=evaluator.evaluate(controls)
    pool=[base]
    details=[{"proposal":0,"status":base.status,"score":base.score,
              "lp_expansion":base.lp_max,"gravity_peak":base.gravity_peak,
              "deviation_rad":base.deviation_rad,"clearance":base.clearance,
              "wall_s":time.perf_counter()-stamps,"rnea_calls":evaluator.rnea.rnea_calls}]
    candidate_witness=None
    chosen_index=None
    if is_witness(base):
        candidate_witness,trial=constructive_timing(
            QuinticGeometry(base.controls),evaluator.auditor,evaluator.bounds)
        details[0]["constructive_timing"]=trial
        if candidate_witness is not None:
            chosen_index=0
            archive=base.controls.copy()
    for iteration in range(1,budget+1):
        if candidate_witness is not None:
            break
        # Both baselines use exactly the same top-four beam size, radii,
        # progress grid, validity screens and constructive timing schedule.
        parent=pool[(iteration-1)%len(pool)]
        proposal,detail=evaluator.suggest(parent,arm,rng,iteration=iteration)
        new=evaluator.evaluate(proposal)
        entry={"proposal":iteration,"parent_score":parent.score,
               "status":new.status,"score":new.score,"lp_expansion":new.lp_max,
               "gravity_peak":new.gravity_peak,"deviation_rad":new.deviation_rad,
               "clearance":new.clearance,"wall_s":time.perf_counter()-stamps,
               "rnea_calls":evaluator.rnea.rnea_calls,**detail}
        details.append(entry)
        if is_witness(new):
            tentative,trial=constructive_timing(
                QuinticGeometry(new.controls),evaluator.auditor,evaluator.bounds)
            entry["constructive_timing"]=trial
            if tentative is not None:
                candidate_witness=tentative
                chosen_index=iteration
                archive=new.controls.copy()
                break
        pool=sorted(pool+[new],key=lambda x:x.score)[:4]
    if candidate_witness is None:
        archive=min(pool,key=lambda x:x.score).controls.copy()
    elapsed=time.perf_counter()-stamps
    result={"case_id":task["case_id"],"arm":arm,"seed":seed,
            "budget":budget,"proposals_evaluated":len(details)-1,
            "evaluation_count":evaluator.evaluations,
            "solver_wall_s_including_gradient_and_screen":elapsed,
            "rnea_calls":evaluator.rnea.rnea_calls,
            "base_fit":fit,"initial_path_lp_expansion":base.lp_max,
            "initial_path_clearance":base.clearance,
            "initial_path_gravity_utilization":base.gravity_peak,
            "candidate_witness_prescreen":candidate_witness is not None,
            "first_witness_proposal":chosen_index,
            "witness_duration_s":float(candidate_witness["time"][-1])
                if candidate_witness is not None else None,
            "best_score":min(c.score for c in pool),
            "ledger":details,
            "claim":"preliminary research screens only; requires independent adaptive validation"}
    return {"controls":archive,"report":result},candidate_witness


def run(root:Path,*,arm_name:str,budget:int,selected:list[str]|None=None,
        seed:int=20261108)->dict:
    if arm_name not in ARMS:
        raise ValueError("unknown controlled search arm")
    if budget not in BUDGETS:
        raise ValueError("unregistered budget")
    output=root/arm_name
    output.mkdir(parents=True,exist_ok=True)
    config={"schema_version":1,"arm":arm_name,"budget":budget,
            "seed":seed,"scenario":SCENARIO,"three_case_registry":ROOTS,
            "count_quintic_controls":22,"lp_grid":33,"beam_size":4,
            "step_radii_rad":list(RADII),
            "initialization":"same nominal original-path quintic least-squares fit in every arm",
            "oracle_controls_forbidden":True,
            "evidence":"prescreen only until independent CPU 8193/adaptive receipts"}
    cfg=output/"experiment_config.json"
    if cfg.exists() and json.loads(cfg.read_text())!=config:
        raise ValueError("matched-budget search configuration changed; use a new root")
    cfg.write_text(json.dumps(config,indent=2)+"\n")
    status={}
    tasks=[]
    for case,(source_name,discovery_name) in ROOTS.items():
        if selected is not None and case not in selected:
            continue
        source=ROOT/"research/joint_time_oracle/results"/source_name
        discovery=ROOT/"research/joint_time_oracle/results"/discovery_name
        source_tasks_list,records=source_tasks(source)
        task=next((t for t in source_tasks_list if t["case_id"]==case),None)
        if task is None:
            raise ValueError("missing frozen task: "+case)
        folder=output/case
        folder.mkdir(exist_ok=True)
        source_profile=discovery/SCENARIO/case/"torque_profile.json"
        profile=read_profile(source_profile)
        ref_receipts=json.loads((discovery/SCENARIO/"w_adaptive_receipts.json").read_text())
        ref_lp=json.loads((discovery/SCENARIO/"refined_lp_summary.json").read_text())
        if (ref_receipts.get(case+":W",{}).get("scaled95") is not True or
            ref_lp.get(case,{}).get("status")!="NUMERICALLY_STRONG_ORIGINAL_PATH_INFEASIBILITY"):
            raise ValueError("registered source case is not independently qualified")
        destination=folder/"torque_profile.json"
        if destination.exists() and sha256_file(destination)!=sha256_file(source_profile):
            raise ValueError("changed source effort profile")
        if not destination.exists():
            destination.write_bytes(source_profile.read_bytes())
        outpath=folder/"search_result.json"
        if outpath.exists():
            existing=json.loads(outpath.read_text())
            if existing["report"]["budget"]!=budget or existing["report"]["seed"]!=seed:
                raise ValueError("existing case used different budget/seed")
            status[case]=existing["report"]
            if (folder/"w_dense.npz").exists():
                tasks.append(task)
            continue
        native=source_native(source,task,records[case])
        res,witness=search(task,profile,native,arm=arm_name,budget=budget,seed=seed)
        np.savez_compressed(folder/"best_controls.npz",controls=res["controls"])
        res["report"].update({
            "task_sha256":task["task_sha256"],
            "torque_profile_sha256":profile["profile_sha256"],
            "source_native_buffers_sha256":sha256_file(source/case/"a10_buffers.npz"),
            "source_witness_not_used_for_initialization":True,
            "controls_sha256":sha256_file(folder/"best_controls.npz"),
        })
        if witness is not None:
            np.savez_compressed(folder/"w_dense.npz",**witness)
            res["report"]["w_dense_sha256"]=sha256_file(folder/"w_dense.npz")
            tasks.append(task)
            (folder/"w_geometry.json").write_text(json.dumps({
                "controls":res["controls"].tolist(),
                "source":"DUAL_GUIDED" if arm_name=="dual" else arm_name.upper(),
                "task_sha256":task["task_sha256"],
                "profile_sha256":profile["profile_sha256"],
                "status":"PRESCREEN_ONLY"},indent=2)+"\n")
        outpath.write_text(json.dumps(res,default=lambda o:o.tolist()
                                 if isinstance(o,np.ndarray) else float(o),
                                 indent=2)+"\n")
        status[case]=res["report"]
        print(arm_name,case,"W_SCREEN" if witness is not None else "NO_W",
              res["report"]["proposals_evaluated"],flush=True)
    # Rebuild the exact validation inventory from SHA-persisted prescreens.
    (output/"tasks.json").write_text(json.dumps(tasks,indent=2)+"\n")
    (output/"pilot.jsonl").write_text("".join(json.dumps({
        "case_id":t["case_id"],"population":t["population"],
        "cohort":"guided_rescue","task_sha256":t["task_sha256"],
        "methods":{"W":{"status":"PRESCREEN_ONLY",
                        "duration_s":status[t["case_id"]]["witness_duration_s"],
                        "verification":{}}}})+"\n" for t in tasks))
    (output/"summary.json").write_text(json.dumps(status,indent=2)+"\n")
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUTPUT)
    p.add_argument("--arm",required=True,choices=ARMS)
    p.add_argument("--budget",type=int,default=128,choices=BUDGETS)
    p.add_argument("--case",action="append")
    p.add_argument("--seed",type=int,default=20261108)
    a=p.parse_args()
    result=run(a.root,arm_name=a.arm,budget=a.budget,
               selected=a.case,seed=a.seed)
    print(json.dumps({"tasks":len(result),
                      "witness_prescreens":sum(x["candidate_witness_prescreen"]
                                               for x in result.values())},indent=2))


if __name__=="__main__":
    main()
