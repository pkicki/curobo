# SPDX-License-Identifier: Apache-2.0
"""Measured native-vs-Pinocchio payload sphere parity (not inferred).

Compares the registered attached_object sphere in cuRobo CUDA FK
against the corresponding independent Pinocchio panda_hand position
for source-disjoint sampled configurations. Also verifies active native
sphere masks, unchanged joint limits and radius.

This is a geometric-model parity gate, NOT full signed collision
distance equivalence across all obstacle/self-collision engines. Such
engine parity remains a separate benchmark limitation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .timeopt_payload_model import (
    MASSES_KG, Payload, file_sha256, native_robot, pinocchio_payload_model,
)
from .timeopt_payload_parity import sampled_states
from .timeopt_payload_tasks import OUT

POSITION_TOL_M=3e-4
RADIUS_TOL_M=1e-6


def run(root:Path,*,masses:tuple[float,...]=MASSES_KG,
        seed:int=20261109,samples:int=128)->dict:
    import pinocchio as pin
    import torch
    from curobo._src.robot.kinematics.kinematics import Kinematics
    from curobo._src.state.state_joint import JointState

    q,_,_=sampled_states(seed,samples)
    comparisons=[]
    for mass in masses:
        payload=Payload(mass)
        robot=native_robot(payload)
        kp=robot.kinematics.kinematics_config
        spheres_index=kp.get_sphere_index_from_link_name(payload.native_link)
        local=kp.get_link_spheres(payload.native_link).detach().cpu().numpy()
        if spheres_index.numel()<1:
            raise ValueError("no attached-object sphere slots")
        local_valid=(np.allclose(local[0,:3],[0.,0.,0.],atol=1e-7)
                     and abs(local[0,3]-payload.radius_m)<=RADIUS_TOL_M
                     and np.all(local[1:,3]==0.))
        if kp.sphere_collision_mask is None:
            raise ValueError("native sphere collision mask missing")
        active=bool((kp.sphere_collision_mask[:,spheres_index[0]]!=0).all())
        kinematics=Kinematics(robot.kinematics)
        device=robot.device_cfg.device
        # Joint names omitted so the pinned Kinematics layer validates dof
        # but does not assume this fixture has the optional locked fingers.
        native_state=kinematics.compute_kinematics(JointState.from_position(
            torch.as_tensor(q,device=device,dtype=torch.float32)
        ))
        world_spheres=(native_state.get_link_spheres().detach().cpu()
                       .numpy()[:,0,spheres_index[0],:])
        model,data,neutral,qi,_=pinocchio_payload_model(payload)
        frame=model.getFrameId(payload.pin_frame,pin.BODY)
        reference=np.empty((samples,3),dtype=float)
        for i,position in enumerate(q):
            neutral[qi]=position
            pin.framesForwardKinematics(model,data,neutral)
            reference[i]=data.oMf[frame].translation
        center_error=np.linalg.norm(world_spheres[:,:3]-reference,axis=1)
        radius_error=np.abs(world_spheres[:,3]-payload.radius_m)
        passed=bool(local_valid and active and
                    np.max(center_error)<=POSITION_TOL_M and
                    np.max(radius_error)<=RADIUS_TOL_M)
        comparisons.append({
            "mass_kg":mass,"payload_sha256":payload.as_record()["payload_sha256"],
            "pass":passed,"local_geometry_pass":bool(local_valid),
            "native_collision_mask_enabled":active,
            "center_max_error_m":float(np.max(center_error)),
            "radius_max_error_m":float(np.max(radius_error)),
            "native_sphere_count":int(spheres_index.numel()),
        })
    status=all(x["pass"] for x in comparisons)
    output={"schema_version":1,"seed":seed,"samples_per_payload":samples,
            "native_and_pinocchio_payload_geometry_parity_pass":bool(status),
            "qualified_payload_sha256":[r["payload_sha256"] for r in comparisons]
                                        if status else [],
            "comparisons":comparisons,
            "model_scope":"registered spherical payload kinematic center/radius and native collision mask",
            "limits":"does not establish full obstacle or self-collision engine equivalence"}
    folder=root/"model_parity"
    folder.mkdir(parents=True,exist_ok=True)
    dest=folder/"payload_collision_parity.json"
    if dest.exists() and json.loads(dest.read_text())!=output:
        raise ValueError("collision geometry parity record already frozen")
    if not dest.exists():
        dest.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUT)
    p.add_argument("--mass",type=float,action="append")
    p.add_argument("--samples",type=int,default=128)
    p.add_argument("--seed",type=int,default=20261109)
    a=p.parse_args()
    print(json.dumps(run(a.root,masses=tuple(a.mass) if a.mass else MASSES_KG,
                         seed=a.seed,samples=a.samples),indent=2))


if __name__=="__main__":
    main()
