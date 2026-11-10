# SPDX-License-Identifier: Apache-2.0
"""Native/independent attached-point-mass models with unchanged effort limits.

Both kinematic models require an attached_object fixed to panda_hand.
This implementation modifies INERTIA ONLY. The physical payload collision
shape is not modeled by either side, so results are mass-only dynamics
studies, NOT payload-collision-qualified robotic executions.
"""
from __future__ import annotations

import numpy as np

from .payload_timeopt_contract import Payload, exact_effort_contract
from .cohort_retime import dynamics as independent_nominal_dynamics


def independent_payload_model(payload:Payload):
    import pinocchio as pin

    model,data,q,qi,vi=independent_nominal_dynamics()
    if payload.mass_kg:
        # franka.yml attaches the extra link to panda_hand with identity transform.
        # Its URDF does not contain attached_object: use the exact panda_hand frame.
        frame=model.frames[model.getFrameId("panda_hand",pin.BODY)]
        inertia=pin.Inertia(payload.mass_kg,np.asarray(payload.com_xyz_m,dtype=float),
                            np.zeros((3,3)))
        model.inertias[frame.parentJoint]+=frame.placement.act(inertia)
        data=model.createData()
    return model,data,q,qi,vi


def native_payload_robot(payload:Payload):
    import torch
    from curobo._src.types.robot import RobotCfg
    from curobo._src.util.config_io import join_path, resolve_config
    from curobo.content import get_robot_configs_path

    cfg=resolve_config(join_path(get_robot_configs_path(),"franka.yml"))
    robot=RobotCfg.create(cfg)
    kp=robot.kinematics.kinematics_config
    mass_link="attached_object"
    if mass_link not in kp.link_name_to_idx_map:
        raise ValueError("native franka config lacks attached_object payload link")
    idx=kp.link_name_to_idx_map[mass_link]
    m=kp.link_masses_com
    if m is None or m.ndim!=2 or m.shape[1]!=4:
        raise ValueError("native CUDA dynamics do not expose payload mass and COM")
    before=m[idx].detach().clone()
    # The native extended attached_object link is separate from the hand.
    # Adding point-mass inertia includes baseline link masses (if nonzero).
    m[idx,:3]=torch.as_tensor(payload.com_xyz_m,dtype=m.dtype,device=m.device)
    m[idx,3]=before[3]+payload.mass_kg
    if not torch.isclose(m[idx,3],before[3]+payload.mass_kg):
        raise RuntimeError("payload mass not installed")
    original=RobotCfg.create(resolve_config(join_path(get_robot_configs_path(),"franka.yml")))
    eff=kp.joint_limits.effort.detach().cpu().numpy()
    reference=original.kinematics.kinematics_config.joint_limits.effort.detach().cpu().numpy()
    contract=exact_effort_contract(eff,reference)
    return robot,{"payload_link":mass_link,"index":int(idx),
                  "native_original_mass_kg":float(before[3]),
                  "native_new_mass_kg":float(m[idx,3]),
                  "native_com_local_xyz_m":list(payload.com_xyz_m),
                  "effort_parity":contract,
                  "payload_collision_qualified":False}


def assert_native_independent_dynamics_parity(native_dynamics,independent,
                                              samples:np.ndarray,*,atol_nm:float=.20,
                                              rtol:float=.03)->dict:
    """Mandatory *external* validation before admitting ANY payload scenario.

    native_dynamics takes q,qd,qdd arrays and returns signed torque in
    panda_joint1..7 order. Compare independent Pinocchio with varying
    states, including a static gravity test and nonzero velocities.
    """
    import pinocchio as pin
    model,data,neutral,qi,vi=independent
    arr=np.asarray(samples,dtype=np.float64)
    if arr.ndim!=3 or arr.shape[1:]!=(3,7) or len(arr)<8:
        raise ValueError("parity grid must contain >=8 q,qd,qdd samples")
    v=np.zeros(model.nv);a=np.zeros(model.nv)
    expected=[]
    for q,qd,qdd in arr:
        state=neutral.copy();state[qi]=q
        v.fill(0);a.fill(0);v[vi]=qd;a[vi]=qdd
        expected.append(pin.rnea(model,data,state,v,a)[vi])
    expected=np.asarray(expected)
    actual=np.asarray(native_dynamics(arr[:,0],arr[:,1],arr[:,2]))
    if actual.shape!=expected.shape or not np.isfinite(actual).all():
        raise ValueError("invalid native torque model output")
    errors=np.abs(actual-expected)
    rel=errors/np.maximum(np.abs(expected),1)
    if np.any(errors>atol_nm+rtol*np.abs(expected)):
        raise ValueError(f"native/Pinocchio payload dynamics disagree: max_abs={errors.max()}")
    return {"samples":len(arr),"max_abs_nm":float(errors.max()),
            "max_rel":float(rel.max()),"atol_nm":atol_nm,"rtol":rtol,
            "parity":"PASS_MODEL_ALIGNED_SAMPLED",
            "caution":"mass-only dynamics comparison; independent payload collision not qualified"}
