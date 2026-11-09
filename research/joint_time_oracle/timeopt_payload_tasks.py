# SPDX-License-Identifier: Apache-2.0
"""Large, deterministic source-disjoint Panda task discovery at NOMINAL effort.

Prepare base start/goal/scene pairs BEFORE running any native planner.
Cross them with payload masses so controls can compare the SAME task
under the SAME original joint effort limits. Endpoint-static-impossible
cells are registered but not counted as geometric planning failures.

The synthetic *physical* payload is a spherical rigid body at the
frozen attached_object frame. This generator does not certify native
collision equivalence: parity of payload sphere geometry and robot RNEA
is a hard separate pre-scoring gate.

No outcome-aware selection occurs when constructing the task inventory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from .cohort_retime import ROOT
from .geometry_feasibility_scene import IndependentGeometryAuditor
from .geometry_feasibility_taskgen import endpoint_pair_separation
from .timeopt_payload_model import (
    MASSES_KG, Payload, classify_endpoint_gravity,
    digest, file_sha256, nominal_effort_bounds, pinocchio_payload_model,
)

OUT = ROOT / "research/joint_time_oracle/results/nominal_payload_source_20261109"
SCENES = (("free", None), ("table", "collision_test.yml"),
          ("cluttered", "collision_primitives_3d.yml"))
SEED = 20261109
TARGET_PER_SCENE = 100
MIN_SEPARATION_RAD = 0.40


def _payload_world_clearance(auditor:IndependentGeometryAuditor,
                             loaded_model, task_q:np.ndarray,
                             radius_m:float)->float:
    """Actual spherical payload/obstacle clearance, independent geometry model.

    This is a sampled source prefilter only, not proof of continuous-time
    collision safety or exact native sphere-model equivalence.
    """
    model, data, neutral, qi, _ = loaded_model
    frame_id = model.getFrameId("panda_hand", pin.BODY)
    if frame_id >= len(model.frames):
        raise ValueError("missing hand frame")
    worst=float("inf")
    for joints in np.asarray(task_q, dtype=float):
        neutral[qi]=joints
        pin.framesForwardKinematics(model,data,neutral)
        center=data.oMf[frame_id].translation
        for _,box_center,rotation,half in auditor.cuboids:
            delta=np.abs(rotation.inv().apply(center-box_center))-half
            distance=np.linalg.norm(np.maximum(delta,0))+min(float(np.max(delta)),0)-radius_m
            worst=min(worst,float(distance))
        for _,sphere_center,sphere_radius in auditor.spheres:
            worst=min(worst,float(np.linalg.norm(center-sphere_center)-sphere_radius-radius_m))
        for _,a,b,r in auditor.capsules:
            offset=b-a
            length=float(offset@offset)
            if length<=0:
                raise ValueError("invalid capsule")
            t=np.clip(((center-a)@offset)/length,0,1)
            closest=a+t*offset
            worst=min(worst,float(np.linalg.norm(center-closest)-r-radius_m))
    return worst


def _gravity_at_endpoints(model_tuple,qpair:np.ndarray)->np.ndarray:
    model,data,neutral,qi,vi=model_tuple
    zero=np.zeros(model.nv)
    result=np.empty_like(qpair)
    for idx,joints in enumerate(qpair):
        neutral[qi]=joints
        result[idx]=pin.rnea(model,data,neutral,zero,zero)[vi]
    return result


def prepare(output:Path,*,seed:int=SEED,per_scene:int=TARGET_PER_SCENE,
            masses:tuple[float,...]=MASSES_KG,
            exclude_roots:tuple[Path,...]=(),
            max_attempts_per_scene:int=40000)->dict:
    if per_scene<1 or max_attempts_per_scene<per_scene:
        raise ValueError("invalid task generation budget")
    values=tuple(float(v) for v in masses)
    if len(values)!=len(set(values)) or not values or values[0]!=0:
        raise ValueError("payload masses must be unique, sorted, and start at zero")
    if values!=tuple(sorted(values)):
        raise ValueError("payload masses must be sorted")
    payloads=[Payload(m) for m in values]
    models={m:pinocchio_payload_model(Payload(m)) for m in values}
    source_history=[]
    exclusion_digests={}
    for root in exclude_roots:
        path=Path(root)/"tasks.json"
        if not path.is_file():
            raise FileNotFoundError("required source-disjoint exclusion inventory: "+str(path))
        source_history.extend(json.loads(path.read_text()))
        exclusion_digests[str(path)]=file_sha256(path)
    nominal=nominal_effort_bounds()
    rng=np.random.default_rng(seed)
    base=[]
    attempts=[]
    for scene_name,scene in SCENES:
        auditor=IndependentGeometryAuditor(scene)
        lo,hi=auditor.position_bounds
        home=np.array([0.,-.785,0.,-2.356,0.,1.571,.785])
        accepted=0
        for trial in range(max_attempts_per_scene):
            if accepted>=per_scene:break
            q0=np.clip(home+rng.normal(0,.92,7),lo+.1,hi-.1)
            q1=np.clip(home+rng.normal(0,.92,7),lo+.1,hi-.1)
            qpair=np.stack((q0,q1))
            gap=endpoint_pair_separation(q0,q1,source_history+base)
            reason=None
            if np.linalg.norm(q1-q0)<1.5:
                reason="SHORT_JOINT_TRAVEL"
            elif gap<MIN_SEPARATION_RAD:
                reason="SOURCE_ENDPOINT_OVERLAP"
            elif not auditor.check(qpair,min_clearance_m=.01)["sampled_acceptable"]:
                reason="ROBOT_ENDPOINT_COLLISION"
            # Screen the physical payload collision for the maximum radius.
            elif _payload_world_clearance(auditor,models[values[0]],qpair,
                                          payloads[0].radius_m)<.01:
                reason="PAYLOAD_ENDPOINT_COLLISION"
            row={"scene":scene_name,"trial":trial,"status":"ACCEPTED" if reason is None else "REJECTED",
                 "reason":reason}
            attempts.append(row)
            if reason is not None:continue
            case=f"np_{scene_name}_{accepted:04d}"
            info={"case_id":case,"population":scene_name,"scene":scene,
                  "robot":"franka.yml","start":q0.tolist(),"goal":q1.tolist(),
                  "joint_names":[f"panda_joint{i}" for i in range(1,8)],
                  "seed":seed+len(base),"max_planner_attempts":5,
                  "endpoint_separation_source_rad":gap,
                  "joint_displacement_l2_rad":float(np.linalg.norm(q1-q0)),
                  "generator_seed":seed,"task_family":"nominal_effort_payload_sweep"}
            info["task_sha256"]=digest(info)
            base.append(info)
            accepted+=1
        if accepted!=per_scene:
            raise ValueError(f"quota unfilled for {scene_name}: {accepted}/{per_scene}")
    tasks=[]
    for b in base:
        scene=b["scene"]
        auditor=IndependentGeometryAuditor(scene)
        qpair=np.stack((b["start"],b["goal"]))
        for payload in payloads:
            gravity=_gravity_at_endpoints(models[payload.mass_kg],qpair)
            endpoint=classify_endpoint_gravity(gravity,nominal)
            record={**b,"case_id":f"{b['case_id']}_m{int(payload.mass_kg*1000):05d}",
                    "source_case_id":b["case_id"],
                    "source_task_sha256":b["task_sha256"],
                    "payload":payload.as_record(),
                    "original_effort_lower":nominal[0].tolist(),
                    "original_effort_upper":nominal[1].tolist(),
                    "actuator_scale_by_joint":[1.]*7,
                    "endpoint":endpoint,
                    "cohort":"source_discovery",
                    "separation_group":"development" if int(digest(b["case_id"])[:2],16)%5<2 else
                    "holdout",
                    "allowed_for_native_challenge":endpoint["endpoint_with_reserved_margin"]}
            record["task_sha256"]=digest(record)
            tasks.append(record)
    output.mkdir(parents=True,exist_ok=True)
    inventory=output/"tasks.json"
    if inventory.exists() and json.loads(inventory.read_text())!=tasks:
        raise ValueError("nominal-payload task inventory already frozen and differs")
    if not inventory.exists():
        inventory.write_text(json.dumps(tasks,indent=2,allow_nan=False)+"\n")
    attemptfile=output/"all_task_proposals.json"
    if attemptfile.exists() and json.loads(attemptfile.read_text())!=attempts:
        raise ValueError("source proposal accounting changed")
    if not attemptfile.exists():
        attemptfile.write_text(json.dumps(attempts,indent=2,allow_nan=False)+"\n")
    manifest={"schema_version":1,"seed":seed,"per_scene":per_scene,
              "base_task_count":len(base),"task_mass_cells":len(tasks),
              "masses_kg":list(values),"original_actuator_effort_unchanged":True,
              "endpoint_admissible":sum(t["allowed_for_native_challenge"] for t in tasks),
              "endpoint_gravity_impossible":sum(t["endpoint"]["category"]=="STATIC_ENDPOINT_IMPOSSIBLE"
                                                for t in tasks),
              "source_disjoint_exclusions":exclusion_digests,
              "task_inventory_sha256":file_sha256(inventory),
              "proposals_sha256":file_sha256(attemptfile),
              "model":"fixed spherical payload at attached_object, pinned Panda robot, nominal joint efforts",
              "limitations":"sampled source endpoint collision only; native model+collision parity required"}
    f=output/"source_manifest.json"
    if f.exists() and json.loads(f.read_text())!=manifest:
        raise ValueError("frozen payload source manifest changed")
    if not f.exists():
        f.write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=OUT)
    p.add_argument("--seed",type=int,default=SEED)
    p.add_argument("--per-scene",type=int,default=TARGET_PER_SCENE)
    p.add_argument("--max-attempts-per-scene",type=int,default=40000)
    p.add_argument("--mass",type=float,action="append")
    p.add_argument("--exclude-root",type=Path,action="append",default=[])
    a=p.parse_args()
    print(json.dumps(prepare(a.output,seed=a.seed,per_scene=a.per_scene,
        masses=tuple(a.mass) if a.mass is not None else MASSES_KG,
        exclude_roots=tuple(a.exclude_root),max_attempts_per_scene=a.max_attempts_per_scene),indent=2))


if __name__=="__main__":
    main()
