# SPDX-License-Identifier: Apache-2.0
"""Independent payload world AND self sphere collision under fixed attachment.

Matches the nominal cuRobo franka.yml self-collision ignore dictionary
and padding policy, with one spherical payload attached at panda_hand.
Qualify engine-vs-engine parity separately: this is a Pinocchio/sphere
model physical sampled gate, not proof of native collision kernel
equivalence or continuous-time obstacle clearance.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin
import yaml

from .cohort_retime import ROOT
from .geometry_feasibility_scene import IndependentGeometryAuditor
from .timeopt_payload_model import Payload,pinocchio_payload_model


def payload_clearances(q:np.ndarray,*,payload:Payload,
                       auditor:IndependentGeometryAuditor,
                       min_clearance_m:float=.005)->dict:
    positions=np.asarray(q,dtype=float)
    if positions.ndim!=2 or positions.shape[1]!=7 or not np.isfinite(positions).all():
        raise ValueError("invalid positions for payload collision")
    cfg=yaml.safe_load((ROOT/"curobo/content/configs/robot/franka.yml").read_text())
    cfg=cfg["robot_cfg"]["kinematics"]
    ignore=cfg["self_collision_ignore"]
    padding=cfg["self_collision_buffer"]
    excluded=np.array([
        link==payload.native_link or
        link in ignore.get(payload.native_link,()) or
        payload.native_link in ignore.get(link,())
        for link in auditor.links],dtype=bool)
    if np.all(excluded):
        raise ValueError("all robot collision spheres excluded from payload self collision")
    model,data,neutral,qi,_=pinocchio_payload_model(payload)
    hand=model.getFrameId(payload.pin_frame,pin.BODY)
    world=float("inf")
    self_clear=float("inf")
    for joints in positions:
        neutral[qi]=joints
        pin.framesForwardKinematics(model,data,neutral)
        center=data.oMf[hand].translation
        spheres=np.empty_like(auditor.local)
        for indices,frame in auditor.grouped:
            pose=data.oMf[frame]
            spheres[indices]=auditor.local[indices]@pose.rotation.T+pose.translation
        d=np.linalg.norm(spheres-center,axis=1)-auditor.radii-payload.radius_m
        buffer=np.array([
            padding.get(payload.native_link,0)+padding.get(link,0)
            for link in auditor.links],dtype=float)
        self_clear=min(self_clear,float(np.min(d[~excluded]-buffer[~excluded])))
        for _,box_center,rotation,half in auditor.cuboids:
            delta=np.abs(rotation.inv().apply(center-box_center))-half
            world=min(world,float(np.linalg.norm(np.maximum(delta,0))
                +min(float(np.max(delta)),0.)-payload.radius_m))
        for _,sphere_center,radius in auditor.spheres:
            world=min(world,float(np.linalg.norm(center-sphere_center)
                                  -radius-payload.radius_m))
        for _,a,b,radius in auditor.capsules:
            vector=b-a
            length=float(vector@vector)
            if length<=0:raise ValueError("zero-length capsule")
            z=np.clip(((center-a)@vector)/length,0,1)
            world=min(world,float(np.linalg.norm(center-(a+z*vector))
                                  -radius-payload.radius_m))
    return {"world_clearance_m":world,"self_clearance_m":self_clear,
            "sampled_acceptable":bool(world>=min_clearance_m
                                       and self_clear>=min_clearance_m),
            "samples":len(positions),
            "policy":"same franka.yml ignored payload-link pairs, independent Pinocchio sphere centers",
            "scope":"sampled only"}
