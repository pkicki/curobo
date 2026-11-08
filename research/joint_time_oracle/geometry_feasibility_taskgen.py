# SPDX-License-Identifier: Apache-2.0
"""Source-disjoint low-endpoint/high-interior gravity-barrier task generator.

Selects ENDPOINTS from an independent Pinocchio model BEFORE any native
torque-derated planner is challenged. A straight-line gravity barrier
is only a screening heuristic; it does not establish that the subsequently
planned native cubic path is torque-infeasible. Every proposal is logged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .cohort_explore import SCENES, digest
from .cohort_torque_wide import endpoint_pair_separation
from .geometry_feasibility_scene import IndependentGeometryAuditor
from .geometry_feasibility_core import sha256_file
from .torque_active import torque_utilization
from .cohort_retime import ROOT

OUT=ROOT/"research/joint_time_oracle/results/geometry_trap_source_20261108"
OLD=ROOT/"research/joint_time_oracle/results/cohort_pilot_isolated_20261106"
WIDE=ROOT/"research/joint_time_oracle/results/torque_wide_source_20261107"


def generate(output:Path,*,seed:int=20261108,per_population:int=10,
             max_attempts_per_task:int=2000,
             exclusion_roots:tuple[Path,...]=(OLD,WIDE),
             min_travel_rad:float=1.5,
             max_endpoint_nominal_gravity:.float=0.35,
             min_interior_nominal_gravity:.float=0.38,
             min_barrier_ratio:float=1.25,
             min_source_separation:float=0.40)->dict:
    """Prepare frozen task inventory; only source nominal planning may follow."""
    if not 0<max_endpoint_nominal_gravity<1 or min_barrier_ratio<=1:
        raise ValueError("invalid gravity-barrier selection thresholds")
    historical=[]
    for root in exclusion_roots:
        file=Path(root)/"tasks.json"
        if not file.exists():
            raise FileNotFoundError("historical exclusion inventory missing "+str(file))
        historical.extend(json.loads(file.read_text()))
    rng=np.random.default_rng(seed)
    tasks=[]; attempts=[]; acceptance={}
    for population,scene,_ in SCENES:
        auditor=IndependentGeometryAuditor(scene)
        lo,hi=auditor.position_bounds
        home=np.array([0.,-.785,0.,-2.356,0.,1.571,.785])
        selected=0
        for k in range(per_population):
            found=False
            for trial in range(max_attempts_per_task):
                q0=np.clip(home+rng.normal(0,1.0,7),lo+.12,hi-.12)
                q1=np.clip(home+rng.normal(0,1.0,7),lo+.12,hi-.12)
                distance=float(np.linalg.norm(q1-q0))
                separation=endpoint_pair_separation(q0,q1,historical+tasks)
                reason=None
                if distance<min_travel_rad:
                    reason="short joint-space travel"
                elif separation<min_source_separation:
                    reason="not source-disjoint in endpoint space"
                else:
                    geometry=auditor.check(np.stack((q0,q1)),min_clearance_m=.01)
                    if not geometry["sampled_acceptable"]:
                        reason="invalid or colliding endpoint"
                    else:
                        gravity=auditor.gravity(np.stack((q0,q1)))
                        endpoint=float(np.max(torque_utilization(gravity,auditor.nominal_effort)))
                        if endpoint>max_endpoint_nominal_gravity:
                            reason="endpoint gravity too high"
                        else:
                            line=q0[None]+np.linspace(0,1,41)[:,None]*(q1-q0)[None]
                            source_grav=auditor.gravity(line)
                            middle=float(np.max(torque_utilization(source_grav[4:-4],
                                                                 auditor.nominal_effort)))
                            ratio=middle/max(endpoint,1e-4)
                            if middle<min_interior_nominal_gravity or ratio<min_barrier_ratio:
                                reason="direct-line gravity demand not higher than endpoint"
                attempts.append({"population":population,"index":k,"proposal":trial,
                                 "travel_rad":distance,"separation_rad":separation,
                                 "status":"ACCEPTED" if reason is None else "REJECTED",
                                 "reason":reason})
                if reason is None:
                    task={"case_id":f"trap_{population}_{k:02d}","cohort":"discovery",
                          "population":population,"scene":scene,"robot":"franka.yml",
                          "start":q0.tolist(),"goal":q1.tolist(),
                          "joint_names":[f"panda_joint{i}" for i in range(1,8)],
                          "initial_velocity":[0.]*7,"initial_acceleration":[0.]*7,
                          "terminal_velocity":[0.]*7,"terminal_acceleration":[0.]*7,
                          "payload":None,"dynamic_limit_scale":.95,
                          "seed":seed+len(tasks),"max_planner_attempts":5,
                          "joint_displacement_l2_rad":distance,
                          "max_endpoint_static_torque_utilization":endpoint,
                          "direct_line_interior_torque_utilization":middle,
                          "direct_line_gravity_barrier_ratio":ratio,
                          "task_generator_seed":seed}
                    task["task_sha256"]=digest(task)
                    tasks.append(task)
                    selected+=1
                    found=True
                    break
            if not found:
                attempts.append({"population":population,"index":k,"status":"QUOTA_UNMET",
                                 "reason":"max proposal budget exhausted"})
        acceptance[population]=selected
    output.mkdir(parents=True,exist_ok=True)
    path=output/"tasks.json"
    if path.exists():
        previous=json.loads(path.read_text())
        if previous!=tasks:
            raise ValueError("source task inventory changed: use new output root")
    path.write_text(json.dumps(tasks,indent=2)+"\n")
    (output/"all_endpoint_proposals.json").write_text(json.dumps(attempts,indent=2)+"\n")
    summary={"schema_version":1,"seed":seed,"per_population":per_population,
             "population_admitted":acceptance,"attempts":len(attempts),
             "historical_roots":[str(Path(r).resolve()) for r in exclusion_roots],
             "historical_task_inventory_hashes":[sha256_file(Path(r)/"tasks.json") for r in exclusion_roots],
             "selection_thresholds":{"min_travel_rad":min_travel_rad,
                 "max_endpoint_nominal_gravity":max_endpoint_nominal_gravity,
                 "min_interior_nominal_gravity":min_interior_nominal_gravity,
                 "min_barrier_ratio":min_barrier_ratio,
                 "min_source_separation":min_source_separation},
             "tasks_sha256":sha256_file(path),
             "note":"gravity-barrier SOURCE heuristic; not native-path infeasibility"}
    (output/"task_generator_manifest.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=OUT)
    p.add_argument("--seed",type=int,default=20261108)
    p.add_argument("--per-population",type=int,default=10)
    p.add_argument("--max-attempts-per-task",type=int,default=2000)
    a=p.parse_args()
    print(json.dumps(generate(a.output,seed=a.seed,
          per_population=a.per_population,
          max_attempts_per_task=a.max_attempts_per_task),indent=2))


if __name__=="__main__":
    main()
