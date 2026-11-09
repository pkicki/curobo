# SPDX-License-Identifier: Apache-2.0
"""Exact GPU native TimeOpt replay at S32768 for selected payload successes.

No waypoint fitting or time-domain interpolation. The frozen cuRobo
native cubic path + timing controls are copied into a new
TrajectoryTimeOpt and resampled using the exact native kernels.
The existing S8193 frozen trajectory MUST replay bitwise-close before
the S32768 path is saved. Independent Pinocchio dynamics/collision
audits happen in a separate CPU process; this GPU replay does not
itself qualify native trajectory feasibility.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


from .timeopt_payload_model import file_sha256,from_record,native_robot
from .timeopt_payload_tasks import OUT


def resample(root:Path,case:str,tag:str,*,samples:int=32768)->dict:
    import torch
    from curobo._src.motion.motion_trajectory_timeopt import TrajectoryTimeOpt
    from curobo._src.motion.motion_trajectory_timeopt_cfg import TrajectoryTimeOptCfg

    if samples<8193:
        raise ValueError("native independent replay requires S32768 or larger")
    task=next((x for x in json.loads((root/"tasks.json").read_text())
               if x["case_id"]==case),None)
    if task is None:raise ValueError("unknown source payload case")
    folder=root/"native_full"/case
    item=json.loads((folder/"native_candidates.json").read_text())[tag]
    if item["status"]!="SUCCESS_NATIVE_UNQUALIFIED":
        raise ValueError("native candidate was not a successful selected lane")
    original=folder/(tag.lower()+"_dense.npz")
    if file_sha256(original)!=item["dense_sha256"]:
        raise ValueError("native source trajectory SHA mismatch")
    native=folder/(tag.lower()+"_buffers.npz")
    inputfile=folder/(tag.lower()+"_native_input.npz")
    if (file_sha256(native)!=item["exact_native_cubic_sha256"] or
        file_sha256(inputfile)!=item["native_input_sha256"]):
        raise ValueError("original TimeOpt buffers changed")
    with np.load(native) as f:
        buffers={k:np.asarray(f[k]) for k in f.files}
    with np.load(inputfile) as f:
        inputs={k:np.asarray(f[k]) for k in f.files}
    with np.load(original) as f:
        old={k:np.asarray(f[k]) for k in f.files}
    payload=from_record(task["payload"])
    robot=native_robot(payload)
    device=robot.device_cfg.device
    bound=np.stack((task["original_effort_lower"],task["original_effort_upper"]))
    cfg=TrajectoryTimeOptCfg(
        robot=robot.kinematics,batch_size=len(buffers["controls"]),
        n_knots=inputs["knot"].shape[1],use_cuda_graph=False,
        torque_lower=torch.as_tensor(bound[0],device=device),
        torque_upper=torch.as_tensor(bound[1],device=device),
        limit_scale=.95,jerk_limit_scale=.95)
    replay=TrajectoryTimeOpt(cfg)
    replay._setup_path(reference=True)
    for key,attr in (
        ("controls","_controls"),("start_boundary","_start_boundary"),
        ("goal","_goal"),("timing","_timing"),
        ("integrated_times","_times"),("success","_success"),
        ("stationary","_stationary")):
        getattr(replay,attr).copy_(torch.as_tensor(
            buffers[key],device=getattr(replay,attr).device,
            dtype=getattr(replay,attr).dtype))
    replay._has_result=True
    lane=int(item["lane"])
    duration=float(buffers["integrated_times"][lane,-1])
    if not (np.isfinite(duration) and duration>0):
        raise ValueError("invalid source native duration")
    def sample(n:int):
        dur=np.asarray(buffers["integrated_times"][:,-1],dtype=float)
        dur=np.where(np.isfinite(dur)&(dur>0),dur,duration)
        u=torch.linspace(0,1,n,device=device)
        t=torch.as_tensor(dur,device=device,dtype=u.dtype)[:,None]*u[None,:]
        states=replay.sample(t.contiguous())
        return {k:np.asarray(getattr(states,v)[lane].detach().cpu())
                for k,v in (("q","position"),("qd","velocity"),
                            ("qdd","acceleration"),("jerk","jerk"))}|{
                "time":np.asarray(t[lane].detach().cpu())}
    base=sample(len(old["time"]))
    error=float(np.max(abs(base["q"]-old["q"])))
    if error>2e-5 or abs(float(base["time"][-1])-float(old["time"][-1]))>1e-4:
        raise ValueError("native 8193 source replay mismatch: "+str(error))
    replay.reset_cuda_graph()
    high=sample(samples)
    target=folder/(tag.lower()+f"_s{samples}.npz")
    if target.exists():raise ValueError("native large replay already exists")
    np.savez_compressed(target,**high)
    return {"case_id":case,"tag":tag,"samples":samples,
            "source_sha256":item["dense_sha256"],
            "replay_source_max_position_difference_rad":error,
            "native_large_sha256":file_sha256(target),
            "scope":"native exact resampling only; requires independent Pinocchio audit"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUT)
    p.add_argument("--case",required=True)
    p.add_argument("--tag",required=True)
    p.add_argument("--samples",type=int,default=32768)
    a=p.parse_args()
    print(json.dumps(resample(a.root,a.case,a.tag,samples=a.samples),indent=2))


if __name__=="__main__":
    main()
