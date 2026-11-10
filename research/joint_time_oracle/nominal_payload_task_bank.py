# SPDX-License-Identifier: Apache-2.0
"""Prospective source-disjoint nominal-effort task bank, paired across payloads.

This generator is BLIND to native planner success/failure and retimer
outcomes. Its broad cohort is a disjoint random start/goal distribution
conditioned only on endpoint collisions and travel. A second, separately
labeled enriched cohort selects high *nominal* payload gravity load.
Never mix their denominators or use discovery tasks for confirmation.

No physical conclusions are drawn from generating a task. Payload
collision/self-collision and native/independent inertia parity are
mandatory scored-run admission checks, not assumed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .cohort_retime import ROOT
from .geometry_feasibility_scene import IndependentGeometryAuditor
from .geometry_feasibility_taskgen import SCENES, endpoint_pair_separation
from .payload_contract import Payload, signed_gravity_profile, sha256_file

SEED=20261110
N_BROAD_PER_SCENE=40
N_ENRICHED_PER_SCENE=20
MAX_ATTEMPTS_PER_SCENE=10000
ENDPOINT_SEPARATION_RAD=.30
MIN_TRAVEL_RAD=1.2
DISCOVERY_ROOTS=(
    ROOT/"research/joint_time_oracle/results/geometry_trap_source_20261108",
    ROOT/"research/joint_time_oracle/results/geometry_trap_confirm_20261109",
)
OUTPUT=ROOT/"research/joint_time_oracle/results/nominal_payload_tasks_20261110"


def hash_json(x:dict)->str:
    return hashlib.sha256(json.dumps(x,sort_keys=True,
        separators=(",",":"),allow_nan=False).encode()).hexdigest()


def fixed_task(case:str,cohort:str,population:str,scene:str|None,
               start:np.ndarray,goal:np.ndarray,seed:int)->dict:
    task={"case_id":case,"cohort":cohort,"population":population,
          "scene":scene,"robot":"franka.yml",
          "start":start.tolist(),"goal":goal.tolist(),
          "joint_names":[f"panda_joint{i}" for i in range(1,8)],
          "initial_velocity":[0.]*7,"initial_acceleration":[0.]*7,
          "terminal_velocity":[0.]*7,"terminal_acceleration":[0.]*7,
          "payload":"SCENARIO_DEFINED_SEPARATELY",
          "nominal_effort_scale":1.0,"seed":seed,
          "max_planner_attempts":5}
    task["task_sha256"]=hash_json(task)
    return task


def historical_inventory(roots:tuple[Path,...],*,require_all:bool=True)->tuple[list,dict]:
    history=[];digests={}
    for root in roots:
        path=root/"tasks.json"
        if not path.exists():
            if require_all:raise FileNotFoundError("source exclusion inventory missing: "+str(path))
            continue
        entries=json.loads(path.read_text())
        history.extend(entries)
        digests[str(path)]=sha256_file(path)
    return history,digests


def generate(root:Path,*,seed:int=SEED,broad:int=N_BROAD_PER_SCENE,
             enriched:int=N_ENRICHED_PER_SCENE,exclusions=DISCOVERY_ROOTS)->dict:
    if broad<1 or enriched<1:
        raise ValueError("both prospective cohorts must be nonempty")
    history,digests=historical_inventory(tuple(exclusions))
    rng=np.random.default_rng(seed)
    tasks=[];attempts=[];quota={}
    home=np.array([0.,-.785,0.,-2.356,0.,1.571,.785])
    for population,scene,_ in SCENES:
        auditor=IndependentGeometryAuditor(scene)
        lo,hi=auditor.position_bounds
        for cohort,count in (("broad",broad),("gravity_enriched",enriched)):
            selected=0
            for attempt in range(MAX_ATTEMPTS_PER_SCENE):
                if selected>=count:break
                q0=np.clip(home+rng.normal(0,1.0,7),lo+.12,hi-.12)
                q1=np.clip(home+rng.normal(0,1.0,7),lo+.12,hi-.12)
                separation=endpoint_pair_separation(q0,q1,history+tasks)
                reason=""
                if np.linalg.norm(q1-q0)<MIN_TRAVEL_RAD:
                    reason="SHORT_TRAVEL"
                elif separation<ENDPOINT_SEPARATION_RAD:
                    reason="SOURCE_OVERLAP"
                elif not auditor.check(np.stack((q0,q1)),min_clearance_m=.01)["sampled_acceptable"]:
                    reason="ENDPOINT_COLLISION_OR_POSITION"
                elif cohort=="gravity_enriched":
                    # Prefilter only using nominal actuator bounds and a
                    # fixed 2-kg payload; these are NOT source outcomes.
                    p=Payload(2.,(0.,0.,.12))
                    qs=q0[None]+np.linspace(0,1,33)[:,None]*(q1-q0)[None]
                    gt=signed_gravity_profile(p,qs)
                    lim=auditor.nominal_effort
                    positive=gt/np.maximum(lim[1],1e-8)
                    negative=gt/np.minimum(lim[0],-1e-8)
                    peak=float(np.max(np.maximum(positive,negative)))
                    if peak<.65:
                        reason="NO_GRAVITY_ENRICHMENT"
                attempts.append({"population":population,"cohort":cohort,
                                 "attempt":attempt,"status":reason or "ACCEPTED",
                                 "travel_rad":float(np.linalg.norm(q1-q0)),
                                 "source_endpoint_separation_rad":separation})
                if reason:continue
                case=f"payload_{cohort}_{population}_{selected:03d}"
                tasks.append(fixed_task(case,cohort,population,scene,
                                        q0,q1,seed+len(tasks)))
                selected+=1
            quota[f"{cohort}:{population}"]={"requested":count,
                                              "accepted":selected}
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    taskfile=root/"tasks.json"
    if taskfile.exists() and json.loads(taskfile.read_text())!=tasks:
        raise ValueError("frozen nominal-payload task inventory differs")
    if not taskfile.exists():taskfile.write_text(json.dumps(tasks,indent=2)+"\n")
    log=root/"all_endpoint_attempts.json"
    if log.exists() and json.loads(log.read_text())!=attempts:
        raise ValueError("frozen endpoint attempt ledger differs")
    if not log.exists():log.write_text(json.dumps(attempts,indent=2)+"\n")
    manifest={"schema_version":1,"seed":seed,"tasks_sha256":sha256_file(taskfile),
              "attempt_log_sha256":sha256_file(log),
              "exclusion_roots_sha256":digests,"quotas":quota,
              "accepted":len(tasks),"attempted":len(attempts),
              "labels":["broad","gravity_enriched"],
              "not_yet_verified":"native planner feasibility and attached-object model/collision parity"}
    out=root/"task_manifest.json"
    if out.exists() and json.loads(out.read_text())!=manifest:
        raise ValueError("frozen task manifest differs")
    if not out.exists():out.write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=OUTPUT)
    p.add_argument("--broad-per-scene",type=int,default=N_BROAD_PER_SCENE)
    p.add_argument("--enriched-per-scene",type=int,default=N_ENRICHED_PER_SCENE)
    p.add_argument("--seed",type=int,default=SEED)
    a=p.parse_args()
    print(json.dumps(generate(a.output,seed=a.seed,
        broad=a.broad_per_scene,enriched=a.enriched_per_scene),indent=2))


if __name__=="__main__":
    main()
