# SPDX-License-Identifier: Apache-2.0
"""Native cuRobo GPU RNEA vs independent Pinocchio payload parity gate.

Only when EVERY proposed payload mass passes at sampled q,qd,qdd
plus registered collision/sphere-model parity may a payload cohort
be scored. We explicitly do NOT claim collision-model parity from
a dynamics-only comparison. An independent collision parity receipt
must be provided by the agent after matching attached-object spheres
and native collision configuration: this tool stores the numerical
RNEA result but refuses to mark model_and_collision_parity_pass until
that separate digest-linked check is present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .timeopt_payload_model import (
    MASSES_KG, Payload, compare_rnea, file_sha256, native_robot,
)
from .timeopt_payload_tasks import OUT


def sampled_states(seed:int,n:int)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    if n<32:raise ValueError("at least 32 independent test states required")
    rng=np.random.default_rng(seed)
    center=np.array([0.,-.785,0.,-2.356,0.,1.571,.785])
    q=np.clip(center+rng.normal(0,.5,(n,7)),-2.2,2.2)
    # Around rest and dynamically excited states, include both signs.
    v=rng.uniform(-1.25,1.25,size=(n,7))
    a=rng.uniform(-4.,4.,size=(n,7))
    v[:n//8]=0;a[:n//8]=0
    return q,v,a


def evaluate(root:Path,*,masses:tuple[float,...]=MASSES_KG,
             seed:int=20261109,n:int=128,
             collision_receipt:Path|None=None)->dict:
    import torch

    from curobo._src.robot.dynamics.dynamics import Dynamics
    from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
    from curobo._src.state.state_joint import JointState

    q,qd,qdd=sampled_states(seed,n)
    if not torch.cuda.is_available():
        raise RuntimeError("native-vs-independent payload RNEA parity requires CUDA")
    records=[]
    for mass in masses:
        payload=Payload(mass)
        robot=native_robot(payload)
        config=DynamicsCfg(kinematics_config=robot.kinematics.kinematics_config,
                           device_cfg=robot.device_cfg)
        dyn=Dynamics(config)
        dyn.setup_batch_size(n)
        device=robot.device_cfg.device
        state=JointState(
            position=torch.as_tensor(q,device=device,dtype=torch.float32),
            velocity=torch.as_tensor(qd,device=device,dtype=torch.float32),
            acceleration=torch.as_tensor(qdd,device=device,dtype=torch.float32),
        )
        tau=dyn.compute_inverse_dynamics(state).detach().cpu().numpy().copy()
        row=compare_rnea(q,qd,qdd,tau,payload)
        records.append(row)
    signatures=[Payload(m).as_record()["payload_sha256"] for m in masses]
    collision=None
    if collision_receipt is not None:
        collision=json.loads(collision_receipt.read_text())
        if not collision.get("native_and_pinocchio_payload_geometry_parity_pass"):
            raise ValueError("native-vs-independent attached-object collision parity unqualified")
        if set(collision.get("qualified_payload_sha256",[]))!=set(signatures):
            raise ValueError("collision parity masses not the same as dynamics parity")
    # Explicitly fail closed; collision parity must not be fabricated here.
    gate=bool(collision is not None and all(row["parity_pass"] for row in records))
    receipt={"schema_version":1,"seed":seed,"samples_per_payload":n,
             "payload_mass_kg":list(masses),"dynamics":records,
             "payload_sha256_qualified":signatures if gate else [],
             "collision_parity_receipt_sha256":file_sha256(collision_receipt)
                if collision_receipt is not None else None,
             "model_and_collision_parity_pass":gate,
             "original_effort_unchanged":True,
             "notes":"a separate measured native-vs-Pinocchio attached-object sphere collision parity receipt is mandatory"}
    folder=root/"model_parity"
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/"payload_parity.json"
    if path.exists() and json.loads(path.read_text())!=receipt:
        raise ValueError("previous parity receipt changed")
    if not path.exists():path.write_text(json.dumps(receipt,indent=2)+"\n")
    np.savez_compressed(folder/"parity_states.npz",q=q,qd=qd,qdd=qdd)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUT)
    p.add_argument("--mass",type=float,action="append")
    p.add_argument("--seed",type=int,default=20261109)
    p.add_argument("--samples",type=int,default=128)
    p.add_argument("--collision-receipt",type=Path)
    a=p.parse_args()
    print(json.dumps(evaluate(a.root,masses=tuple(a.mass) if a.mass else MASSES_KG,
        seed=a.seed,n=a.samples,collision_receipt=a.collision_receipt),indent=2))


if __name__=="__main__":
    main()
