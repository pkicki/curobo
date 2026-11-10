# SPDX-License-Identifier: Apache-2.0
"""Source-disjoint Panda payload-hardcase cohort (nominal actuator limits).

Sample endpoint tasks before ANY planner or retimer results. Freeze all
source tasks and full attempt inventory, then expand the SAME task to
paired payload masses/COM offsets; split by SOURCE TASK, not scenario.
This does NOT guarantee that native failures admit another solution;
only independently qualified witness establishes a rescue.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from .geometry_feasibility_scene import IndependentGeometryAuditor
from .geometry_feasibility_taskgen import SCENES
from .payload_model_parity import independent_payload_model
from .payload_timeopt_contract import (
    COM_OFFSETS_M,MASS_KG,OUT,Payload,SEED,canonical_sha,
    immutable_json,validate_fresh_split,
)
from .torque_active import torque_utilization


def sample_endpoint_pairs(*,seed:int=SEED,per_scene:int=60,
                          min_distance:float=1.4,min_separation:float=.5,
                          max_proposals_per_scene:int=20000,
                          historical:list[dict]|None=None)->tuple[list[dict],dict]:
    if per_scene<10 or max_proposals_per_scene<per_scene:
        raise ValueError("invalid prospective cohort size")
    historical=historical or []
    rng=np.random.default_rng(seed)
    home=np.array([0.,-.785,0.,-2.356,0.,1.571,.785])
    inventory=[];attempt_counts={}
    scene_config={k:v for k,v,_ in SCENES}
    for scene_name in ("free","table","cluttered"):
        auditor=IndependentGeometryAuditor(scene_config[scene_name])
        lower,upper=auditor.position_bounds
        reasons=Counter()
        made=0
        for attempt in range(max_proposals_per_scene):
            if made>=per_scene:break
            q0=np.clip(home+rng.normal(0,1.0,7),lower+.13,upper-.13)
            q1=np.clip(home+rng.normal(0,1.0,7),lower+.13,upper-.13)
            if np.linalg.norm(q1-q0)<min_distance:
                reasons["short_travel"]+=1;continue
            existing=historical+inventory
            if any(min(np.linalg.norm(q0-np.asarray(x["start"]))+
                       np.linalg.norm(q1-np.asarray(x["goal"])),
                       np.linalg.norm(q0-np.asarray(x["goal"]))+
                       np.linalg.norm(q1-np.asarray(x["start"])))<min_separation
                   for x in existing):
                reasons["endpoint_duplicate"]+=1;continue
            check=auditor.check(np.stack((q0,q1)),min_clearance_m=.01)
            if not check["sampled_acceptable"]:
                reasons["endpoint_collision_or_position"]+=1;continue
            sid=f"panda_{scene_name}_{made:04d}"
            h=int(canonical_sha([sid,seed])[:8],16)%10
            split="development" if h<4 else "confirmation" if h<7 else "holdout"
            item={"source_task_id":sid,"split":split,"scene":scene_config[scene_name],
                  "population":scene_name,"robot":"franka.yml","start":q0.tolist(),
                  "goal":q1.tolist(),"seed":seed+len(inventory),
                  "source_distribution_seed":seed,
                  "source_acceptance":"joint_pos_world_self_valid_endpoints_only",
                  "task_geometry_collision_clearance_m":float(check["world_clearance_m"])}
            item["source_task_sha256"]=canonical_sha(item)
            inventory.append(item);made+=1
        attempt_counts[scene_name]={"accepted":made,"attempted":attempt+1,
                                    "rejected":dict(reasons)}
        if made<per_scene:
            raise RuntimeError(f"not enough fresh task pairs in {scene_name}: {made}")
    return inventory,attempt_counts


def expand_payloads(source:list[dict],*,masses:tuple[float,...]=MASS_KG,
                    offsets:tuple[tuple[float,float,float],...]=COM_OFFSETS_M)->tuple[list[dict],dict]:
    records=[];stats=Counter()
    # Avoid repeated model rebuilds per source configuration; each
    # (mass,COM) is a distinct *dynamics* scenario with identical effort.
    for mass in masses:
        for com in offsets:
            payload=Payload(float(mass),tuple(map(float,com)))
            model,data,neutral,qi,vi=independent_payload_model(payload)
            v=np.zeros(model.nv);a=np.zeros(model.nv)
            for task in source:
                endpoints=[]
                for point in (task["start"],task["goal"]):
                    q=neutral.copy();q[qi]=point
                    endpoints.append(pin.rnea(model,data,q,v,a)[vi].copy())
                # Standstill endpoints require gravity within the true
                # nominal signed torque limits, no synthetic derating.
                auditor=IndependentGeometryAuditor(task["scene"])
                limits=auditor.nominal_effort
                peak=float(np.max(torque_utilization(np.asarray(endpoints),limits)))
                admitted=peak<=.95
                row=dict(task)
                row.update({
                    "case_id":task["source_task_id"]+"__"+payload.id,
                    "payload":asdict(payload),"joint_limit_scale":1.0,
                    "nominal_actuator_limits":True,
                    "payload_collision_qualified":False,
                    "endpoint_gravity_peak_nominal_utilization":peak,
                    "dynamic_limit_scale":.95,
                    "source_collision_scope":"robot+environment spheres; payload shape NOT included",
                    "admission":"ENDPOINT_ADMITTED" if admitted else "ENDPOINT_GRAVITY_REJECTED"})
                row["task_sha256"]=canonical_sha(row)
                records.append(row)
                stats[row["admission"]]+=1
    validate_fresh_split(records)
    return records,dict(stats)


def prepare(output:Path,*,seed:int=SEED,per_scene:int=60)->dict:
    # Require disjointness with historical source tasks as well.
    import importlib
    from .geometry_feasibility_taskgen import OLD,WIDE,OUT as HIST_TRAPS
    historical=[]
    for folder in (OLD,WIDE,HIST_TRAPS):
        path=folder/"tasks.json"
        if not path.exists():
            raise FileNotFoundError("historical source inventory absent: "+str(path))
        historical.extend(json.loads(path.read_text()))
    source,counts=sample_endpoint_pairs(seed=seed,per_scene=per_scene,
                                        historical=historical)
    cases,admission=expand_payloads(source)
    manifest={"schema":1,"seed":seed,"per_scene":per_scene,
              "source_groups":len(source),"paired_scenario_count":len(cases),
              "sample_counts":counts,"payload_admissions":admission,
              "payload_mass_grid_kg":list(MASS_KG),
              "payload_com_grid_m":[list(x) for x in COM_OFFSETS_M],
              "original_actuator_limits":True,
              "source_disjoint_split":"source_task_id",
              "no_native_or_oracle_outcome_used_for_generation":True,
              "payload_collision_qualified":False,
              "status":"TASKS_ONLY_NEVER_CONFUSE_WITH_NATIVE_FAILURE_OR_FEASIBILITY"}
    immutable_json(output/"source_tasks.json",{"tasks":source})
    immutable_json(output/"tasks.json",{"tasks":cases})
    immutable_json(output/"source_generation.json",manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT/"source")
    parser.add_argument("--seed",type=int,default=SEED)
    parser.add_argument("--per-scene",type=int,default=60)
    args=parser.parse_args()
    print(json.dumps(prepare(args.out,seed=args.seed,per_scene=args.per_scene),indent=2))


if __name__=="__main__":
    main()
