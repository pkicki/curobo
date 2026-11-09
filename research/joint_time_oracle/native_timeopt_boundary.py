# SPDX-License-Identifier: Apache-2.0
"""Independent physical audit of actual native post-TimeOpt sampled states.

The native collision/dynamics masks are produced on post-TimeOpt states,
whereas prior independent geometry checks measured PRE-TimeOpt paths.
This diagnostic pairs the same native lane's captured post-TimeOpt
position/velocity/acceleration/jerk states with independent Pinocchio
RNEA and analytic sphere checks. Unavailable or invalid post samples
are UNKNOWN, never proof of a native stage failure.

Research-only probe can perturb CUDA execution timing. Compare results
against an uninstrumented control; never compare instrumented latencies.
No sampled result is a continuous-time certificate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial import cKDTree

from .cohort_geometry import NativeGeometry
from .geometry_feasibility_core import sha256_file
from .geometry_feasibility_scene import IndependentGeometryAuditor
from .torque_active import read_profile
from .cohort_retime import ROOT


def worst_violation(values:np.ndarray,bounds:np.ndarray,*,scale:float=1.,
                    centered:bool=True)->float:
    a=np.asarray(values,dtype=float)
    low,high=np.asarray(bounds,dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("nonfinite state/bounds")
    if centered:
        center=(low+high)/2.
        lower=center+scale*(low-center)
        upper=center+scale*(high-center)
    else:
        lower=low*scale;upper=high*scale
    return float(max(0.,np.max(lower-a),np.max(a-upper)))


def compare_native_mask(native:bool|None,independent:bool|None)->str:
    if native is None or independent is None:
        return "UNKNOWN"
    if native and independent:
        return "BOTH_PASS"
    if native and not independent:
        return "NATIVE_PASS_INDEPENDENT_FAIL"
    if not native and independent:
        return "NATIVE_FAIL_INDEPENDENT_PASS_DIFFERENT_CHECK_POSSIBLE"
    return "BOTH_FAIL"


def audited_post_lane(states:dict, lane:int, auditor:IndependentGeometryAuditor,
                      bounds:dict,torque_bounds:np.ndarray,
                      *,task:dict, pre_positions:np.ndarray|None=None)->dict:
    time=np.asarray(states["time"][lane],dtype=float)
    q=np.asarray(states["q"][lane],dtype=float)
    qd=np.asarray(states["qd"][lane],dtype=float)
    qdd=np.asarray(states["qdd"][lane],dtype=float)
    jerk=np.asarray(states["jerk"][lane],dtype=float)
    if (q.ndim!=2 or q.shape[1]!=7 or len(time)<33 or
        qd.shape!=q.shape or qdd.shape!=q.shape or jerk.shape!=q.shape or
        not all(np.isfinite(z).all() for z in (time,q,qd,qdd,jerk)) or
        np.any(np.diff(time)<=0)):
        return {"status":"POST_TIMEOPT_SAMPLE_UNUSABLE",
                "reason":"nonfinite, insufficient or nonmonotone sampled states"}
    check=auditor.check(q,min_clearance_m=.005)
    vel=np.zeros(auditor.model.nv)
    acc=np.zeros(auditor.model.nv)
    torques=np.empty_like(q)
    for i,item in enumerate(q):
        auditor.q[auditor.qi]=item
        vel.fill(0.);acc.fill(0.)
        vel[auditor.vi]=qd[i];acc[auditor.vi]=qdd[i]
        torques[i]=pin.rnea(auditor.model,auditor.data,auditor.q,vel,acc)[auditor.vi]
        auditor.rnea_calls+=1
    raw={"position":q,"velocity":qd,"acceleration":qdd,
         "jerk":jerk,"torque":torques}
    family={}
    for name,arr in raw.items():
        lowhigh=torque_bounds if name=="torque" else bounds[name]
        family[name]={
            "physical_violation":worst_violation(arr,lowhigh),
            "scaled95_violation":worst_violation(arr,lowhigh,scale=.95,
                                             centered=(name!="torque"))
                             if name!="position" else worst_violation(arr,lowhigh),
        }
    start=np.asarray(task["start"],dtype=float)
    goal=np.asarray(task["goal"],dtype=float)
    endpoint_pos=float(max(np.max(np.abs(q[0]-start)),
                           np.max(np.abs(q[-1]-goal))))
    path_deviation=None
    if pre_positions is not None:
        dist,_=cKDTree(pre_positions).query(q,k=1)
        path_deviation=float(np.max(dist))
    return {"status":"INDEPENDENT_POST_TIMEOPT_SAMPLED",
            "samples":len(time),"duration_s":float(time[-1]),
            "position_endpoints_max_abs_rad":endpoint_pos,
            "endpoint_velocity_inf":float(np.max(np.abs(qd[[0,-1]]))),
            "endpoint_acceleration_inf":float(np.max(np.abs(qdd[[0,-1]]))),
            "families":family,
            "world_clearance_m":check["world_clearance_m"],
            "self_clearance_m":check["self_clearance_m"],
            "geometry_screen_pass":check["sampled_acceptable"],
            "post_positions_to_pre_curve_nearest_sample_max_rad":path_deviation,
            "physical_sampled_pass":bool(check["sampled_acceptable"] and
                endpoint_pos<1e-3 and
                all(f["physical_violation"]<1e-6 for f in family.values())),
            "scaled95_sampled_pass":bool(check["sampled_acceptable"] and
                endpoint_pos<1e-3 and
                all(f["scaled95_violation"]<1e-6 for f in family.values())),
            "warning":"sampled post-TimeOpt states, not continuous-time certificate; nearest pre-curve difference uses sampled approximation only"}


def analyze(native_arm:Path,profile_arm:Path,*,case_filter:str|None=None)->dict:
    tasks=json.loads((profile_arm/"tasks.json").read_text())
    with np.load(ROOT/"research/joint_time_oracle/results/native_reconstruction_20261106/baseline_validation_states.npz") as f:
        bounds={k:np.asarray(f[k+"_bounds"]) for k in
                ("position","velocity","acceleration","jerk")}
    result={}
    for task in tasks:
        case=task["case_id"]
        if case_filter and case!=case_filter:continue
        folder=native_arm/case
        profile=read_profile(profile_arm/case/"torque_profile.json")
        torque_limits=np.stack((profile["effort_lower"],profile["effort_upper"]))
        auditor=IndependentGeometryAuditor(task["scene"])
        entries=[]
        invs=sorted(folder.glob("*_pretimeopt_inventory.json"))
        for inv in invs:
            for record in json.loads(inv.read_text()):
                tag=record.get("tag")
                if not isinstance(tag,str) or "/" in tag or ".." in tag:
                    raise ValueError("unsafe native capture tag")
                entry={"tag":tag,"status":"POST_STATES_UNAVAILABLE",
                       "native_validation_masks":record.get("timeopt_validation_masks"),
                       "post_sample_error":record.get("post_timeopt_sampling_error"),
                       "lanes":[]}
                path=folder/(tag+"_post_timeopt_states.npz")
                if not path.exists():
                    entries.append(entry);continue
                if sha256_file(path)!=record.get("post_timeopt_state_sha256"):
                    entry["status"]="POST_CAPTURE_SHA_MISMATCH"
                    entries.append(entry);continue
                with np.load(path) as z:
                    data={k:np.asarray(z[k]) for k in z.files}
                prefile=folder/(tag+"_native_geometry.npz")
                pre=None
                if prefile.exists() and sha256_file(prefile)==record.get("native_cubic_geometry_sha256"):
                    with np.load(prefile) as f:
                        pre={k:np.asarray(f[k]) for k in f.files}
                count=len(data["valid_duration_mask"])
                if data["q"].shape[0]!=count:
                    entry["status"]="POST_CAPTURE_SHAPE_MISMATCH"
                    entries.append(entry);continue
                for lane in range(count):
                    mask={}
                    native_masks=record.get("timeopt_validation_masks",{})
                    if isinstance(native_masks,dict):
                        for key,arr in native_masks.items():
                            mask[key]=bool(arr[lane]) if len(arr)==count else None
                    if not bool(data["valid_duration_mask"][lane]):
                        entry["lanes"].append({"lane":lane,
                            "status":"INVALID_NATIVE_TIMING_DURATION",
                            "native_masks":mask})
                        continue
                    pre_q=None
                    if pre is not None:
                        try:
                            pre_q=NativeGeometry(pre,lane).evaluate_trimmed(
                                np.linspace(0,1,513))[0]
                        except Exception:
                            pass
                    audit=audited_post_lane(data,lane,auditor,bounds,torque_limits,
                                             task=task,pre_positions=pre_q)
                    audit["lane"]=lane
                    audit["native_masks"]=mask
                    native_collision=mask.get("collision_valid")
                    native_dyn=mask.get("limits_and_dynamics_valid")
                    if audit["status"]=="INDEPENDENT_POST_TIMEOPT_SAMPLED":
                        audit["mask_agreement"]={
                            "collision":compare_native_mask(native_collision,
                                          audit["geometry_screen_pass"]),
                            "dynamics":compare_native_mask(native_dyn,
                                 all(f["physical_violation"]<1e-6
                                     for name,f in audit["families"].items()
                                     if name!="position")),
                        }
                    entry["lanes"].append(audit)
                entry["status"]="AUDITED_POST_SAMPLES"
                entries.append(entry)
        result[case]=entries
    out=native_arm/"post_timeopt_independent_diagnostics.json"
    out.write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-arm",type=Path,required=True)
    p.add_argument("--profile-arm",type=Path,required=True)
    p.add_argument("--case")
    a=p.parse_args()
    result=analyze(a.native_arm,a.profile_arm,case_filter=a.case)
    print(json.dumps({"cases":len(result),"snapshots":sum(len(x) for x in result.values()),
                      "audited_lanes":sum(sum(y["status"]=="INDEPENDENT_POST_TIMEOPT_SAMPLED"
                                               for y in rec["lanes"])
                                          for x in result.values() for rec in x)},indent=2))


if __name__=="__main__":
    main()
