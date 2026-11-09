# SPDX-License-Identifier: Apache-2.0
"""Matched global-search ablation: necessary torque LP gate vs no LP gate.

Every arm consumes the SAME immutable, LP-blind proposal NPZ in the SAME
order. Each uses identical independent sampled geometry, collision, gravity
screens and constructive rest-to-rest timing. Only "lp_gate" performs a
necessary pointwise torque-feasibility LP BEFORE the expensive constructive
timing check. A positive LP is an exclusion certificate for that path;
a zero LP does NOT certify feasibility. The no_lp arm never uses the LP
for candidate selection.

A candidate W remains PRELIMINARY until independently dense/adaptive
physical+Scaled95 audited and hash-linked; all calls must be measured,
including the LP gate's cost, not only number of skipped retiming calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .cohort_geometry import QuinticGeometry
from .cohort_retime import ROOT
from .geometry_feasibility_core import path_distance, sha256_file
from .geometry_feasibility_discover import constructive_timing, source_tasks
from .geometry_feasibility_scene import IndependentGeometryAuditor
from .torque_active import read_profile, torque_utilization
from .torque_dual_guidance import AlignedPointDynamics, solve_point_lp
from .torque_global_bank import (
    DEVELOPMENT_CASE, HOLDOUT_CASES, MAX_PROPOSALS, OUTPUT, SEED,
    make_bank,
)
from .torque_guided_rescue import resolve_registry

ARMS = ("no_lp", "lp_gate")
LP_MESHES = (17, 65)
LP_REJECTION = 0.005
GRAVITY_REJECTION = 0.88
MIN_CLEARANCE_M = 0.005
MIN_DEVIATION_RAD = 0.1


def positive_lp_at_points(rnea: AlignedPointDynamics, geometry: QuinticGeometry,
                          limits: np.ndarray, *,
                          meshes: tuple[int, ...] = LP_MESHES,
                          margin: float = LP_REJECTION) -> dict:
    """A positive *numerically dual-checked* gap at ANY point excludes path."""
    solves = 0
    for mesh in meshes:
        for s in np.linspace(.01, .99, mesh):
            a, m, g = rnea.coefficients(geometry, float(s))
            lp = solve_point_lp(a, m, g, limits)
            solves += 1
            if lp.status != "SOLVED":
                return {"decision": "LP_INCONCLUSIVE_ERROR",
                        "progress": float(s), "evaluations": solves,
                        "solver_status": lp.status}
            if lp.dual_bound > margin and lp.duality_gap <= 1e-6:
                return {"decision": "LP_EXCLUDED", "progress": float(s),
                        "expansion": lp.expansion, "dual_bound": lp.dual_bound,
                        "evaluations": solves,"solver_status":lp.status}
    return {"decision":"LP_NOT_EXCLUDED", "evaluations":solves,
            "note":"zero sampled LP gaps are necessary only, not a feasible timing certificate"}


def screen_geometry(controls, auditor: IndependentGeometryAuditor,
                    limits: np.ndarray, original_q: np.ndarray) -> tuple[dict, QuinticGeometry]:
    geometry = QuinticGeometry(np.asarray(controls,dtype=np.float64))
    positions = geometry.evaluate(np.linspace(0,1,129))[0]
    measured = auditor.check(positions,min_clearance_m=MIN_CLEARANCE_M)
    record = {
        "collision_position_screen":bool(measured["sampled_acceptable"]),
        "world_clearance_m":measured["world_clearance_m"],
        "self_clearance_m":measured["self_clearance_m"],
        "position_violation_rad":measured["position_violation_rad"],
    }
    if not record["collision_position_screen"]:
        return record,geometry
    gravity = float(np.max(torque_utilization(measured["gravity_torques"],
                                             limits * .95)))
    deviation = path_distance(positions,original_q)
    record.update({"gravity_scaled95_peak":gravity,
                   "geometry_deviation_rad":deviation,
                   "cheap_screen_pass":bool(gravity<=GRAVITY_REJECTION
                      and deviation>=MIN_DEVIATION_RAD)})
    return record,geometry


def evaluate_bank(*, arm_name:str, case:str, bank_file:Path,
                  task:dict, profile:dict, native,
                  budget:int=MAX_PROPOSALS) -> tuple[dict,dict|None,np.ndarray|None]:
    if arm_name not in ARMS or budget!=MAX_PROPOSALS:
        raise ValueError("unregistered arm or budget")
    with np.load(bank_file) as z:
        proposals=np.asarray(z["proposals"],dtype=np.float64)
        original_fit=np.asarray(z["base_controls"],dtype=np.float64)
    if proposals.shape!=(budget,22,7) or original_fit.shape!=(22,7):
        raise ValueError("frozen bank shape mismatch")
    limits=np.stack((profile["effort_lower"],profile["effort_upper"]))
    auditor=IndependentGeometryAuditor(task["scene"])
    rnea=AlignedPointDynamics() if arm_name=="lp_gate" else None
    original_q=native.evaluate_trimmed(np.linspace(0,1,513))[0]
    start=time.perf_counter()
    ledger=[]
    candidate_states=None;candidate_controls=None
    skipped_timing=0;timing_calls=0; lp_exclusions=0; lp_errors=0
    for index,controls in enumerate(proposals):
        stage=time.perf_counter()
        entry={"proposal":index+1,"proposal_sha256":__import__("hashlib").sha256(
            np.ascontiguousarray(controls).tobytes()).hexdigest(),
            "arm":arm_name,"status":"PENDING"}
        try:
            cheap,geometry=screen_geometry(controls,auditor,limits,original_q)
            entry.update(cheap)
            if not cheap.get("cheap_screen_pass",False):
                entry["status"]="CHEAP_REJECTED"
            else:
                if arm_name=="lp_gate":
                    lp=positive_lp_at_points(rnea,geometry,limits)
                    entry["torque_lp"]=lp
                    if lp["decision"]=="LP_EXCLUDED":
                        entry["status"]="TORQUE_LP_EXCLUDED"
                        lp_exclusions+=1
                        skipped_timing+=1
                    else:
                        if lp["decision"]=="LP_INCONCLUSIVE_ERROR":
                            lp_errors+=1
                        # A numerical LP error must not falsely reject a
                        # physically feasible candidate. Time it instead.
                        timing_calls+=1
                        states,history=constructive_timing(
                            geometry,auditor,limits)
                        entry["constructive_timing"]=history
                        entry["status"]="W_PRESCREEN" if states is not None else "TIMING_REJECTED"
                        if states is not None:
                            candidate_states=states
                            candidate_controls=controls.copy()
                else:
                    timing_calls+=1
                    states,history=constructive_timing(geometry,auditor,limits)
                    entry["constructive_timing"]=history
                    entry["status"]="W_PRESCREEN" if states is not None else "TIMING_REJECTED"
                    if states is not None:
                        candidate_states=states
                        candidate_controls=controls.copy()
        except Exception as exc:
            entry["status"]="EVALUATION_ERROR"
            entry["error"]=repr(exc)
        entry["elapsed_s"]=time.perf_counter()-start
        entry["candidate_wall_s"]=time.perf_counter()-stage
        entry["auditor_rnea_calls"]=auditor.rnea_calls
        entry["lp_rnea_calls"]=rnea.rnea_calls if rnea is not None else 0
        ledger.append(entry)
        if candidate_states is not None:
            break
    report={
        "schema_version":1,"case_id":case,"arm":arm_name,
        "budget":budget,"bank_sha256":sha256_file(bank_file),
        "proposals_evaluated":len(ledger),"first_screen_proposal":
            ledger[-1]["proposal"] if candidate_states is not None else None,
        "preliminary_witness":candidate_states is not None,
        "witness_duration_s":float(candidate_states["time"][-1])
              if candidate_states is not None else None,
        "total_cpu_search_wall_s":time.perf_counter()-start,
        "skipped_timing_calls_by_lp":skipped_timing,
        "constructive_timing_calls":timing_calls,
        "lp_exclusions":lp_exclusions,"lp_errors":lp_errors,
        "lp_rnea_calls":rnea.rnea_calls if rnea is not None else 0,
        "geometry_and_timing_rnea_calls":auditor.rnea_calls,
        "total_rnea_calls":auditor.rnea_calls+
             (rnea.rnea_calls if rnea is not None else 0),
        "ledger":ledger,"scope":"preliminary only; independent dense/adaptive W audits required",
    }
    return report,candidate_states,candidate_controls


def run(root:Path,*,case:str,arm_name:str,budget:int=MAX_PROPOSALS,
        seed:int=SEED,registry_file:Path|None=None,
        holdout:bool=False)->dict:
    if arm_name not in ARMS or budget!=MAX_PROPOSALS:
        raise ValueError("invalid registered method or budget")
    if case==DEVELOPMENT_CASE:
        if holdout:
            raise ValueError("development case must not be called holdout")
    elif case in HOLDOUT_CASES:
        if not holdout:
            raise ValueError("holdout case forbidden before frozen development")
        from .torque_global_freeze import verify
        verify(root,seed=seed,budget=budget,registry_file=registry_file)
    else:
        raise ValueError("unregistered three-task pilot case")
    metadata=make_bank(root,case,seed=seed,count=budget,registry_file=registry_file)
    registry=resolve_registry(registry_file)
    info=registry[case]
    tasks,rows=source_tasks(info["source_root"])
    task=next((x for x in tasks if x["case_id"]==case),None)
    if task is None:
        raise ValueError("source task absent")
    from .geometry_feasibility_discover import source_native
    native=source_native(info["source_root"],task,rows[case])
    source_profile=info["discovery_root"]/info["scenario_id"]/case/"torque_profile.json"
    profile=read_profile(source_profile)
    if metadata["task_sha256"]!=task["task_sha256"] or (
        metadata["torque_profile_sha256"]!=profile["profile_sha256"]):
        raise ValueError("frozen bank/profile/task mismatch")
    arm=root/arm_name
    folder=arm/case
    folder.mkdir(parents=True,exist_ok=True)
    outfile=folder/"screen_result.json"
    if outfile.exists():
        result=json.loads(outfile.read_text())
        if (result.get("bank_sha256")!=metadata["bank_sha256"] or
            result.get("budget")!=budget or result.get("arm")!=arm_name):
            raise ValueError("existing screened result used different frozen inputs")
    else:
        result,witness,controls=evaluate_bank(
            arm_name=arm_name,case=case,bank_file=root/"bank"/case/"proposals.npz",
            task=task,profile=profile,native=native,budget=budget)
        result.update({"task_sha256":task["task_sha256"],
                       "torque_profile_sha256":profile["profile_sha256"],
                       "seed":seed,
                       "source_original_native_sha256":
                           metadata["source_hashes"]["native_original_cubic"]})
        if witness is not None:
            np.savez_compressed(folder/"w_dense.npz",**witness)
            result["w_dense_sha256"]=sha256_file(folder/"w_dense.npz")
            geometry_file=folder/"w_geometry.json"
            geometry_file.write_text(json.dumps({
                "controls":controls.tolist(),"task_sha256":task["task_sha256"],
                "profile_sha256":profile["profile_sha256"],
                "source":"LP_BLIND_FROZEN_GLOBAL_BANK",
                "status":"PRESCREEN_ONLY"},indent=2)+"\n")
            result["w_geometry_sha256"]=sha256_file(geometry_file)
            selected=result["ledger"][-1]
            (folder/"discovery.json").write_text(json.dumps({
                "status":"CONSTRUCTIVE_WITNESS_PRESCREEN","case_id":case,
                "candidate_attempts":[{
                    "witness_sha256":result["w_dense_sha256"],
                    "witness_geometry_sha256":result["w_geometry_sha256"],
                    "arc_length_aligned_geometry_deviation_rad":
                        selected["geometry_deviation_rad"],
                    "source":"global_bank_"+arm_name
                }]
            },indent=2)+"\n")
        outfile.write_text(json.dumps(result,indent=2)+"\n")
    destination=folder/"torque_profile.json"
    if destination.exists() and sha256_file(destination)!=sha256_file(source_profile):
        raise ValueError("profile changed after screen")
    if not destination.exists():
        destination.write_bytes(source_profile.read_bytes())
    # Rebuild arm inventory from all prescreened cases. Running dev then
    # heldout changes tasks.json, not the individual case-frozen results.
    case_tasks=[]
    for candidate in (DEVELOPMENT_CASE,*HOLDOUT_CASES):
        path=arm/candidate/"screen_result.json"
        if not path.exists():
            continue
        row=json.loads(path.read_text())
        if not row.get("preliminary_witness"):
            continue
        source_tasks_list,_=source_tasks(registry[candidate]["source_root"])
        item=next(x for x in source_tasks_list if x["case_id"]==candidate)
        case_tasks.append(item)
    (arm/"tasks.json").write_text(json.dumps(case_tasks,indent=2)+"\n")
    (arm/"pilot.jsonl").write_text("".join(json.dumps({
        "case_id":item["case_id"],"population":item["population"],
        "cohort":"global_lp_gate","task_sha256":item["task_sha256"],
        "methods":{"W":{"status":"PRESCREEN_ONLY",
                        "duration_s":json.loads((arm/item["case_id"]/"screen_result.json").read_text())["witness_duration_s"],
                        "verification":{}}}
    })+"\n" for item in case_tasks))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUTPUT)
    p.add_argument("--case",required=True)
    p.add_argument("--arm",required=True,choices=ARMS)
    p.add_argument("--seed",type=int,default=SEED)
    p.add_argument("--budget",type=int,default=MAX_PROPOSALS)
    p.add_argument("--registry",type=Path)
    p.add_argument("--holdout",action="store_true")
    a=p.parse_args()
    result=run(a.root,case=a.case,arm_name=a.arm,
               seed=a.seed,budget=a.budget,
               registry_file=a.registry,holdout=a.holdout)
    print(json.dumps({k:v for k,v in result.items() if k!="ledger"},indent=2))


if __name__=="__main__":
    main()
