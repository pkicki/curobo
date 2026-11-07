# SPDX-License-Identifier: Apache-2.0
"""Generate source-disjoint, larger-excursion Panda problems for torque tests.

This stage only freezes candidate endpoints. Subsequent geometric planning,
native TimeOpt, independent Pinocchio torque export and torque calibration
must be completed BEFORE B/C/D optimization is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .cohort_explore import SCENES, planner_for, digest
from .curobo_adapter import CuroboOracleModel
from .torque_active import torque_utilization

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/joint_time_oracle/results/torque_wide_source_20261107"
OLD = ROOT / "research/joint_time_oracle/results/cohort_pilot_isolated_20261106/tasks.json"


def endpoint_pair_separation(start, goal, historical) -> float:
    """Max of start/goal Euclidean distances under unchanged endpoint order."""
    return min(min(float(np.linalg.norm(start-np.asarray(x[key]))),
                   float(np.linalg.norm(goal-np.asarray(x[key]))))
               for x in historical for key in ("start","goal")) if historical else float("inf")


def generate(output: Path, *, seed: int = 270711, per_population: int = 10,
             min_joint_distance: float = 1.5, min_gravity_utilization: float = .10,
             source_separation: float = .35, max_attempts: int = 5000) -> dict:
    rng=np.random.default_rng(seed)
    old=json.loads(OLD.read_text()) if OLD.exists() else []
    tasks=[];attempts=[]
    for population,scene,_ in SCENES:
        planner=planner_for(scene);model=CuroboOracleModel(planner)
        limits=model.limits()
        home=planner.default_joint_state.position.detach().cpu().numpy().reshape(-1)[:7]
        lo,hi=limits.position
        for index in range(per_population):
            for trial in range(max_attempts):
                start=np.clip(home+rng.normal(0,.75,7),lo+.12,hi-.12)
                goal=np.clip(home+rng.normal(0,.85,7),lo+.12,hi-.12)
                travel=float(np.linalg.norm(goal-start))
                disjoint=endpoint_pair_separation(start,goal,old+tasks)
                reason=None
                if travel<min_joint_distance:
                    reason="insufficient joint displacement"
                elif disjoint<source_separation:
                    reason="near historical or already admitted task endpoints"
                else:
                    q=torch.as_tensor(np.stack([start,goal]),device=model.device,dtype=torch.float32)
                    with torch.no_grad():
                        costs=[float(z.detach().max().item()) for z in model.collision_components(q)]
                        z=torch.zeros_like(q)
                        gravity=model.torque(q,z,z).detach().cpu().numpy()
                    stat=float(np.max(torque_utilization(gravity,limits.torque)))
                    if any(x>1e-7 or not np.isfinite(x) for x in costs):
                        reason="start or goal collision"
                    elif stat<min_gravity_utilization:
                        reason="low static torque demand at both endpoints"
                record={"population":population,"index":index,"attempt":trial,
                        "travel_rad":travel,"source_separation_rad":disjoint,
                        "accepted":reason is None,"rejection":reason}
                attempts.append(record)
                if reason is None:
                    task={"case_id":f"wide_{population}_{index:02d}","cohort":"pilot",
                          "population":population,"scene":scene,"robot":"franka.yml",
                          "start":start.tolist(),"goal":goal.tolist(),
                          "joint_names":model.joint_names,
                          "initial_velocity":[0.]*7,"initial_acceleration":[0.]*7,
                          "terminal_velocity":[0.]*7,"terminal_acceleration":[0.]*7,
                          "payload":None,"dynamic_limit_scale":.95,
                          "seed":seed+len(tasks),"max_planner_attempts":5,
                          "joint_displacement_l2_rad":travel,
                          "max_endpoint_static_torque_utilization":stat}
                    task["task_sha256"]=digest(task)
                    tasks.append(task)
                    break
            else:
                attempts.append({"population":population,"index":index,
                                 "accepted":False,"rejection":"proposal budget exhausted"})
        planner.destroy()
    output.mkdir(parents=True,exist_ok=True)
    (output/"tasks.json").write_text(json.dumps(tasks,indent=2)+"\n")
    (output/"endpoint_proposals.json").write_text(json.dumps(attempts,indent=2)+"\n")
    summary={"source":"new wider-excursion task inventory","seed":seed,
             "selected":len(tasks),"target":per_population*len(SCENES),
             "attempts":len(attempts),"min_joint_distance":min_joint_distance,
             "source_separation":source_separation,
             "min_endpoint_static_torque_utilization":min_gravity_utilization,
             "historical_tasks_sha256":hashlib.sha256(OLD.read_bytes()).hexdigest() if OLD.exists() else None,
             "task_inventory_sha256":hashlib.sha256((output/"tasks.json").read_bytes()).hexdigest(),
             "note":"endpoint candidate admission only; planner/dynamics qualification remains pending"}
    (output/"source_manifest.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--per-population",type=int,default=10)
    parser.add_argument("--seed",type=int,default=270711)
    parser.add_argument("--max-attempts",type=int,default=5000)
    args=parser.parse_args()
    print(json.dumps(generate(args.output,seed=args.seed,
             per_population=args.per_population,max_attempts=args.max_attempts),indent=2))


if __name__=="__main__":
    main()
