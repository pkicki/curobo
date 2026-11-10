# SPDX-License-Identifier: Apache-2.0
"""Independent Panda sphere-collision model with a tool-local payload sphere.

Uses the SAME sphere center and radius as the opt-in native
configured_native_robot configuration. Gripper/finger self-collision
exemptions are inherited from the frozen franka.yml; the attached
sphere is not treated as a separately articulated body. This is
sampled collision geometry, not exact mesh collision certification.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin
import yaml

from .cohort_retime import ROOT
from .geometry_feasibility_scene import IndependentGeometryAuditor
from .payload_contract import Payload


class PayloadGeometryAuditor(IndependentGeometryAuditor):
    def __init__(self,scene:str|None,payload:Payload):
        super().__init__(scene)
        self.payload=payload.validate()
        if self.payload.mass_kg == 0:
            return
        cfg=yaml.safe_load((ROOT/"curobo/content/configs/robot/franka.yml").read_text())
        k=cfg["robot_cfg"]["kinematics"]
        link=self.payload.tool_frame
        self.links.append(link)
        self.local=np.vstack((self.local,np.asarray(self.payload.com_tool_m)))
        self.radii=np.r_[self.radii,self.payload.radius_m]
        self.grouped=[
            (np.asarray([i for i,x in enumerate(self.links) if x==name],dtype=int),
             self.model.getFrameId(name,pin.BODY))
            for name in sorted(set(self.links))
        ]
        if any(fid>=len(self.model.frames) for _,fid in self.grouped):
            raise ValueError("attached tool frame missing from Pinocchio")
        ignore=k["self_collision_ignore"]
        padding=k["self_collision_buffer"]
        pairs=[(i,j) for i in range(len(self.links))
                        for j in range(i+1,len(self.links))
                        if self.links[i]!=self.links[j]
                        and self.links[j] not in ignore.get(self.links[i],[])
                        and self.links[i] not in ignore.get(self.links[j],[])]
        if not pairs:
            raise ValueError("no active self-collision sphere pairs")
        self.ia,self.ib=np.asarray(pairs,dtype=int).T
        self.self_limits=(self.radii[self.ia]+self.radii[self.ib]+
            np.asarray([padding.get(self.links[i],0)+padding.get(self.links[j],0)
                        for i,j in pairs]))
        self.payload_sphere_index=len(self.links)-1
        self.attached_collision_model="panda_hand tool-local sphere"

    def parity_receipt(self)->dict:
        return {"schema_version":1,
                "payload":self.payload.serialize(),
                "attached_collision_model":"panda_hand sphere",
                "attachment_included":self.payload.mass_kg==0 or (
                    len(self.local)==len(self.links) and
                    np.allclose(self.local[-1],self.payload.com_tool_m) and
                    self.radii[-1]==self.payload.radius_m),
                "native_collision_parity":"MUST_COMPARE_NATIVE_AND_INDEPENDENT_DISTANCES_ON_FROZEN_Q_SAMPLES",
                "physical_collision_scope":"sampled robot and attached sphere vs world/self sphere only"}
