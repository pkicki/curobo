# SPDX-License-Identifier: Apache-2.0
"""Discover path-dependent torque infeasibility with an explicit constructive witness.

Order of evidence:
1. Read frozen nominal native geometry and a PRE-REGISTERED joint-derating grid.
2. Reject endpoint-static-gravity-infeasible combinations.
3. Require a robust positive relaxed torque LP on the ORIGINAL path.
4. Propose alternative quintic geometry WITHOUT looking at native replan results.
5. Require sampled joint/world/self clearance and a low-gravity envelope.
6. Construct exact smooth rest-to-rest timing, test scaled95 dynamics via
   aligned independent Pinocchio, and save a candidate witness.
7. Freeze everything. A SECOND independent dense/adaptive audit and fresh
   full native planner experiment must follow; this discovery isn't proof.

Scenario-targeted discovery is explicitly adversarial and cannot itself serve
as an unbiased population-level generalization or native failure rate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from .cohort_geometry import NativeGeometry, QuinticGeometry
from .cohort_retime import ROOT
from .extreme_torque_path_lp import inspect_curve
from .geometry_feasibility_core import (
    CANDIDATE_PERTURBATION_RAD, CANDIDATE_SEEDS, GRAVITY_SCALED95_TARGET,
    MIN_GEOMETRY_DEVIATION_RAD, MIN_ORIGINAL_LP_GAP, WITNESS_TIMES_S,
    path_distance, perturb_controls, profile_from_scenario, scenarios,
    sha256_file, smoothstep_controls, timed_states,
)
from .geometry_feasibility_scene import IndependentGeometryAuditor
from .torque_active import checked_bounds, torque_utilization

SOURCE = ROOT / "research/joint_time_oracle/results/cohort_pilot_isolated_20261106"
REFERENCE = ROOT / "research/joint_time_oracle/results/torque_stress_final_20261107/severe"
OUT = ROOT / "research/joint_time_oracle/results/geometry_feasibility_discovery_20261108"


def source_tasks(source:Path)->tuple[list,dict]:
    tasks=json.loads((source/"tasks.json").read_text())
    rows={x["case_id"]:x for x in
          (json.loads(line) for line in (source/"pilot.jsonl").read_text().splitlines()
           if line.strip())}
    if len(tasks)!=len(rows):
        raise ValueError("source inventory does not match captured native trajectories")
    return tasks,rows


def source_native(source:Path,task:dict,row:dict) -> NativeGeometry:
    case=task["case_id"]
    buffers_path=source/case/"a10_buffers.npz"
    with np.load(buffers_path) as z:
        buffers={k:np.asarray(z[k]) for k in z.files}
    lane=int(row["methods"]["A10"]["lane"])
    geometry=NativeGeometry(buffers,lane)
    q0,q1=geometry.evaluate_trimmed(np.array([0.,1.]))[0]
    if (np.max(abs(q0-np.asarray(task["start"])))>1e-4 or
        np.max(abs(q1-np.asarray(task["goal"])))>1e-4):
        raise ValueError("frozen native geometry endpoints mismatch")
    return geometry


def candidate_controls(task:dict,reference:Path,seed:int):
    q0,q1=np.asarray(task["start"],dtype=float),np.asarray(task["goal"],dtype=float)
    bases=[("rest_to_rest_smoothstep",smoothstep_controls(q0,q1))]
    rfile=reference/task["case_id"]/"c_best.json"
    if rfile.exists():
        raw=json.loads(rfile.read_text())
        c=np.asarray(raw["coefficients"],dtype=float)
        if c.ndim==2 and c.shape[1]==len(q0):
            geometry=QuinticGeometry(c)
            endpoints=geometry.evaluate(np.array([0.,1.]))[0]
            if np.max(abs(endpoints[0]-q0))<1e-5 and np.max(abs(endpoints[-1]-q1))<1e-5:
                bases.insert(0,("frozen_D_geometry",c))
    for name,c in bases:
        yield name,c
    rng=np.random.default_rng(seed)
    for amplitude in CANDIDATE_PERTURBATION_RAD:
        for k in range(CANDIDATE_SEEDS):
            parent,c=bases[k%len(bases)]
            # Structured coordinated detours in selected joint directions.
            target_joint=int(rng.integers(0,len(q0)))
            yield (f"perturb_{amplitude:.2f}_{k:03d}_parent_{parent}_j{target_joint+1}",
                   perturb_controls(c,rng,amplitude=amplitude,target_joint=target_joint))


def path_screen(geometry:QuinticGeometry,auditor:IndependentGeometryAuditor,
                limits:np.ndarray,original_positions:np.ndarray,
                *,samples:int=129):
    s=np.linspace(0,1,samples)
    q=np.asarray(geometry.evaluate(s)[0])
    clearance=auditor.check(q,min_clearance_m=.005)
    result={"sampled_acceptable":bool(clearance["sampled_acceptable"]),
            "position_violation_rad":clearance["position_violation_rad"],
            "world_clearance_m":clearance["world_clearance_m"],
            "self_clearance_m":clearance["self_clearance_m"]}
    if not result["sampled_acceptable"]:
        result["reason"]="sampled position/world/self collision"
        return result
    gravity=clearance["gravity_torques"]
    scaled=float(np.max(torque_utilization(gravity,limits*.95)))
    dist=path_distance(q,original_positions)
    result.update({"gravity_scaled95_peak":scaled,
                   "arc_length_aligned_geometry_deviation_rad":dist})
    result["sampled_acceptable"]=bool(
        scaled<=GRAVITY_SCALED95_TARGET and dist>=MIN_GEOMETRY_DEVIATION_RAD)
    if not result["sampled_acceptable"]:
        result["reason"]="insufficient gravity margin or geometry distinction"
    return result


def constructive_timing(geometry,auditor,limits:np.ndarray,
                        *,n:int=257)->tuple[dict|None,list]:
    # All T share q(s(u)) and have dynamic inverse-dynamics torque
    # g(q) + (tau_T1 - g(q))/T**2. Compute one T=1 RNEA sweep.
    reference=timed_states(geometry,1.,n=n)
    q=reference["q"]
    baseline=ROOT/"research/joint_time_oracle/results/native_reconstruction_20261106/baseline_validation_states.npz"
    with np.load(baseline) as z:
        bounds={family:np.asarray(z[family+"_bounds"]) for family in
                ("position","velocity","acceleration","jerk")}
    grav=auditor.gravity(q)
    tau=np.empty_like(grav)
    velocity=np.zeros(auditor.model.nv)
    acceleration=np.zeros(auditor.model.nv)
    for k in range(len(q)):
        auditor.q[auditor.qi]=q[k]
        velocity.fill(0);acceleration.fill(0)
        velocity[auditor.vi]=reference["qd"][k]
        acceleration[auditor.vi]=reference["qdd"][k]
        tau[k]=pin.rnea(auditor.model,auditor.data,auditor.q,velocity,acceleration)[auditor.vi]
    # Static gravity sufficient condition is checked here; this is sampled
    # screening, not a mathematically global path bound.
    outcomes=[]
    for T in WITNESS_TIMES_S:
        torque=grav+(tau-grav)/T**2
        scaled_torque=float(np.max(torque_utilization(torque,limits*.95)))
        utilization={"torque":scaled_torque}
        for family,order in (("velocity",1),("acceleration",2),("jerk",3)):
            values=reference[{"velocity":"qd","acceleration":"qdd","jerk":"jerk"}[family]]/T**order
            lo,hi=bounds[family]
            center=(lo+hi)/2
            low=center+.95*(lo-center);high=center+.95*(hi-center)
            # Signed-side utilization does not apply to nonzero-center families.
            utilization[family]=float(np.max(np.maximum(
                (values-center)/(high-center),
                (center-values)/(center-low))))
        eligible=bool(all(np.isfinite(v) and v<=.995 for v in utilization.values()))
        outcomes.append({"duration_s":T,"max_scaled95_utilization":utilization,
                         "screen_pass":eligible})
        if eligible:
            return timed_states(geometry,T,n=8193),outcomes
    return None,outcomes


def discover(source:Path,reference:Path,output:Path,*,
             max_tasks:int|None=None,min_gap:float=MIN_ORIGINAL_LP_GAP,
             lp_samples:int=65,max_candidates:int=100)->dict:
    source=source.resolve();output=output.resolve()
    tasks,rows=source_tasks(source)
    if max_tasks is not None:
        tasks=tasks[:max_tasks]
    if not tasks or lp_samples<17 or max_candidates<1:
        raise ValueError("empty discovery or invalid budget")
    cfg={"schema_version":1,"source_tasks_sha256":sha256_file(source/"tasks.json"),
         "source":str(source),"reference":str(reference.resolve()),
         "scenarios":scenarios(),"minimum_positive_original_lp_gap":min_gap,
         "lp_prescreen_samples":lp_samples,"max_alternate_geometry_proposals":max_candidates,
         "duration_grid_s":list(WITNESS_TIMES_S),
         "candidate_seeds":CANDIDATE_SEEDS,
         "min_geometry_deviation_rad":MIN_GEOMETRY_DEVIATION_RAD,
         "sampling_contract":"discovery only; independently requalify dense/adaptive witness",
         "negative_control":"original native path under original joint limits",
         "purpose":"find path-dependent feasibility, never infer full planner failure"}
    output.mkdir(parents=True,exist_ok=True)
    register=output/"discovery_registration.json"
    if register.exists() and json.loads(register.read_text())!=cfg:
        raise ValueError("discovery protocol already frozen under a different configuration")
    register.write_text(json.dumps(cfg,indent=2)+"\n")
    model_sha=sha256_file(ROOT/"curobo/content/assets/robot/franka_description/franka_panda.urdf")
    arm_rows={}
    summary={"schema_version":1,"source_count":len(tasks),"scenarios_per_task":len(cfg["scenarios"]),
             "attempted_scenarios":0,"admitted_scenarios":0,"positive_original_lp":0,
             "constructive_witness_prescreens":0,"case_counts":{}}
    cache={}
    for task in tasks:
        case=task["case_id"];scene=task["scene"]
        if scene not in cache:
            cache[scene]=IndependentGeometryAuditor(scene)
        auditor=cache[scene]
        native=source_native(source,task,rows[case])
        native_curve=native.evaluate_trimmed
        full_s=np.linspace(0,1,257)
        original_positions=np.asarray(native_curve(full_s)[0])
        source_file=source/case/"a10_buffers.npz"
        source_geometry_sha=sha256_file(source_file)
        endpoint_grav=auditor.gravity(np.array([task["start"],task["goal"]]))
        counts={"attempted":0,"admitted":0,"original_lp_positive":0,"screened_witness":0}
        for scenario in cfg["scenarios"]:
            regime=scenario["scenario_id"]
            arm=output/regime
            folder=arm/case
            folder.mkdir(parents=True,exist_ok=True)
            profile=profile_from_scenario(
                source_task=task,original_limits=auditor.nominal_effort,
                endpoint_gravity=endpoint_grav,scenario=scenario,
                model_sha256=model_sha,source_geometry_sha256=source_geometry_sha)
            (folder/"torque_profile.json").write_text(json.dumps(profile,indent=2)+"\n")
            counts["attempted"]+=1;summary["attempted_scenarios"]+=1
            record={"case_id":case,"regime":regime,"task_sha256":task["task_sha256"],
                    "profile_sha256":profile["profile_sha256"],
                    "endpoint_admitted":profile["admitted"],
                    "status":"ENDPOINT_REJECTED" if not profile["admitted"] else "PENDING",
                    "source_geometry_sha256":source_geometry_sha,"candidate_attempts":[]}
            if profile["admitted"]:
                summary["admitted_scenarios"]+=1;counts["admitted"]+=1
                limits=np.array([profile["effort_lower"],profile["effort_upper"]])
                original=inspect_curve(native_curve,limits,samples=lp_samples)
                record["original_relaxed_lp"]=original
                if (original.get("status")=="SOLVED" and
                    original.get("strong_fixed_path_infeasibility_witness") and
                    original["max_required_torque_bound_expansion"]>=min_gap):
                    summary["positive_original_lp"]+=1;counts["original_lp_positive"]+=1
                    record["status"]="POSITIVE_ORIGINAL_LP_NEEDS_ALTERNATIVE"
                    seed=int(hashlib.sha256(f"{case}:{regime}".encode()).hexdigest()[:16],16)
                    for k,(name,controls) in enumerate(candidate_controls(task,reference,seed)):
                        if k>=max_candidates:
                            break
                        row={"index":k,"candidate":name}
                        try:
                            geo=QuinticGeometry(controls)
                            screening=path_screen(geo,auditor,limits,original_positions)
                            row.update(screening)
                            if screening["sampled_acceptable"]:
                                witness,attempts=constructive_timing(geo,auditor,limits)
                                row["timing_trials"]=attempts
                                if witness is not None:
                                    path=folder/"w_dense.npz"
                                    np.savez_compressed(path,**witness)
                                    (folder/"w_geometry.json").write_text(json.dumps({
                                        "controls":controls.tolist(),"source":name,
                                        "task_sha256":task["task_sha256"],
                                        "profile_sha256":profile["profile_sha256"],
                                        "status":"PRESCREEN_ONLY",
                                    },indent=2)+"\n")
                                    row["witness_sha256"]=sha256_file(path)
                                    row["witness_duration_s"]=float(witness["time"][-1])
                                    record["status"]="CONSTRUCTIVE_WITNESS_PRESCREEN"
                                    counts["screened_witness"]+=1
                                    summary["constructive_witness_prescreens"]+=1
                                    record["chosen_candidate_index"]=k
                                    record["chosen_duration_s"]=row["witness_duration_s"]
                                    record["candidate_attempts"].append(row)
                                    break
                        except Exception as exc:
                            row["reason"]=str(exc)
                        record["candidate_attempts"].append(row)
                else:
                    record["status"]="ORIGINAL_LP_NOT_INFEASIBLE"
            (folder/"discovery.json").write_text(json.dumps(record,indent=2)+"\n")
            arm_rows.setdefault(regime,[]).append((task,record,profile))
        summary["case_counts"][case]=counts
        print(case,counts,flush=True)
    for regime,values in arm_rows.items():
        arm=output/regime
        (arm/"attempt_inventory.json").write_text(json.dumps([
            record for _,record,_ in values],indent=2)+"\n")
        selected=[task for task,record,_ in values
                  if record["status"]=="CONSTRUCTIVE_WITNESS_PRESCREEN"]
        # All discovered source cases remain in inventory. The native challenge
        # must be restricted to independently qualified W cases, never prescreen.
        (arm/"tasks.json").write_text(json.dumps(selected,indent=2)+"\n")
        (arm/"pilot.jsonl").write_text("".join(json.dumps({
            "case_id":task["case_id"],"cohort":"discovery",
            "population":task["population"],"task_sha256":task["task_sha256"],
            "methods":{"W":{"duration_s":record.get("chosen_duration_s"),
                            "status":"PRESCREEN_ONLY","verification":{}}}})+"\n"
            for task,record,_ in values
            if record["status"]=="CONSTRUCTIVE_WITNESS_PRESCREEN"))
    (output/"discovery_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=SOURCE)
    p.add_argument("--reference",type=Path,default=REFERENCE)
    p.add_argument("--output",type=Path,default=OUT)
    p.add_argument("--max-tasks",type=int)
    p.add_argument("--max-candidates",type=int,default=100)
    p.add_argument("--lp-samples",type=int,default=65)
    a=p.parse_args()
    print(json.dumps(discover(a.source,a.reference,a.output,
                              max_tasks=a.max_tasks,max_candidates=a.max_candidates,
                              lp_samples=a.lp_samples),indent=2))


if __name__=="__main__":
    main()
