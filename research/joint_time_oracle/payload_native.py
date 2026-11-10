# SPDX-License-Identifier: Apache-2.0
"""Opt-in original-effort cuRoboV2 robot with added tool-link rigid inertia.

This updates the CUDA model's link mass, center and inertia together
using the parallel-axis theorem. A native/Pinocchio RNEA parity witness
must still be obtained: the native and Pinocchio link conventions may
differ. Attached-object collision geometry is an ADDITIONAL gate,
not implicitly satisfied by changing inertia. No production defaults
or global RobotCfg cache is modified.
"""
from __future__ import annotations

import numpy as np

from .payload_contract import Payload


def inertia_matrix(six:np.ndarray)->np.ndarray:
    a=np.asarray(six,dtype=float)
    if a.shape!=(6,) or not np.isfinite(a).all():
        raise ValueError("invalid rigid inertia vector")
    ixx,iyy,izz,ixy,ixz,iyz=a
    return np.array([[ixx,ixy,ixz],[ixy,iyy,iyz],[ixz,iyz,izz]])


def inertia_six(mat:np.ndarray)->np.ndarray:
    a=np.asarray(mat,dtype=float)
    if a.shape!=(3,3) or not np.allclose(a,a.T,atol=1e-9):
        raise ValueError("inertia tensor must be symmetric")
    return np.array([a[0,0],a[1,1],a[2,2],a[0,1],a[0,2],a[1,2]])


def parallel_axis(mass:float,offset:np.ndarray)->np.ndarray:
    d=np.asarray(offset,dtype=float)
    return mass*((d@d)*np.eye(3)-np.outer(d,d))


def combined_link_inertia(old_mass_com:np.ndarray,old_inertia6:np.ndarray,
                          payload:Payload)->tuple[np.ndarray,np.ndarray]:
    """Combine payload at link-local origin, preserving angular momentum."""
    p=payload.validate()
    old=np.asarray(old_mass_com,dtype=float)
    if old.shape!=(4,) or not np.isfinite(old).all() or old[3]<0:
        raise ValueError("bad link mass/COM")
    m0=float(old[3]);m=p.mass_kg
    if m==0:
        return old.copy(),np.asarray(old_inertia6,dtype=float).copy()
    mnew=m0+m
    c0=old[:3];cp=np.asarray(p.com_tool_m,dtype=float)
    center=(m0*c0+m*cp)/mnew
    first=inertia_matrix(old_inertia6)
    second=np.diag(np.asarray(p.inertia_diagonal_kg_m2))
    # Link inertias are assumed to be specified about their own COM.
    new=first+parallel_axis(m0,c0-center)+second+parallel_axis(m,cp-center)
    if np.min(np.linalg.eigvalsh(new))<=0:
        raise ValueError("combined inertia nonpositive")
    return np.r_[center,mnew],inertia_six(new)


def configured_native_robot(payload:Payload,*,strict_collision:bool=True):
    """Create independent native RobotCfg; only output with collision gate."""
    from curobo._src.types.robot import RobotCfg
    from curobo.content import get_robot_configs_path
    from curobo._src.util.config_io import join_path
    from curobo._src.util.config_io import resolve_config
    config=resolve_config(join_path(get_robot_configs_path(),"franka.yml"))
    # The native CUDA sphere collision model must include the payload
    # proxy BEFORE constructing the planner and its cached collision model.
    kinematics=config["robot_cfg"]["kinematics"]
    spheres=kinematics["collision_spheres"]
    if payload.mass_kg:
        spheres.setdefault("panda_hand",[]).append({
            "center":list(payload.com_tool_m),"radius":float(payload.radius_m)})
    robot=RobotCfg.create(config)
    params=robot.kinematics.kinematics_config
    original=params.joint_limits.effort.clone()
    if payload.mass_kg:
        mcom=params.get_link_masses_com("panda_hand").detach().cpu().numpy()
        inertia=params.get_link_inertia("panda_hand").detach().cpu().numpy()
        updated,six=combined_link_inertia(mcom,inertia,payload)
        import torch
        params.update_link_mass("panda_hand",float(updated[3]))
        params.update_link_com("panda_hand",torch.as_tensor(updated[:3],
            dtype=original.dtype,device=original.device))
        params.update_link_inertia("panda_hand",torch.as_tensor(six,
            dtype=original.dtype,device=original.device))
    if not np.array_equal(original.detach().cpu().numpy(),
                          params.joint_limits.effort.detach().cpu().numpy()):
        raise ValueError("original manufacturer effort limits modified")
    # A robot collision sphere is a conservative payload proxy; it
    # participates in the same native world/self sphere model as other
    # hand spheres. Full parity with independent sphere geometry and
    # physical object shape is still a separate experiment gate.
    if strict_collision and payload.mass_kg and (
        not spheres.get("panda_hand")):
        raise ValueError("native payload collision sphere unavailable")
    return robot


def main():
    import argparse,json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mass",type=float,default=0.)
    p.add_argument("--com-z",type=float,default=.12)
    a=p.parse_args()
    model=configured_native_robot(Payload(a.mass_kg,(0.,0.,a.com_z)),
                                  strict_collision=False)
    print(json.dumps({"status":"NATIVE_INERTIA_ONLY_NOT_COLLISION_QUALIFIED",
                      "model_type":type(model).__name__}))


if __name__=="__main__":
    main()
