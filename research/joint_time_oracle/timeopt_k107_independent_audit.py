# SPDX-License-Identifier: Apache-2.0
"""Independent K107 timing-spline replay on exact quintic geometry + payload.

Do NOT use K107's native screen or solver dynamics as the certification
authority. Rebuild the original quintic and K107 cubic squared-speed
spline independently in SciPy, obtain physical q,qd,qdd,jerk, then use
independent Pinocchio (same attached spherical mass) for RNEA.
Validate signed actuator/kinematic Scaled95 limits, endpoints,
robot/environment and payload world collision on S8193 and event-aware
S32768 separately. This is sampled, NOT a continuous-time certificate.

No torque-rate constraint is imposed in the COMMON 4-family contract.
K107's original five-family results are not a matching comparator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.interpolate import BSpline

from .cohort_geometry import QuinticGeometry
from .geometry_feasibility_scene import IndependentGeometryAuditor
from .timeopt_oracle_contract import CONSTRAINT_FAMILIES, check_response
from .timeopt_payload_model import (
    Payload, file_sha256, pinocchio_payload_model,
)
from .timeopt_payload_tasks import _payload_world_clearance

GRIDS=(8193,32768)
MAX_SAMPLED_VIOLATION=1e-6
KINEMATIC_KEYS=("velocity","acceleration","jerk")


def event_grid(geometry:QuinticGeometry,k:int,count:int)->np.ndarray:
    tk=np.r_[np.zeros(3),np.linspace(0,1,k-2),np.ones(3)]
    internal=np.r_[geometry.spline.t,tk]
    values=np.unique(internal[(internal>0)&(internal<1)])
    eps=np.array([-1e-6,0,1e-6])
    events=np.clip((values[:,None]+eps).reshape(-1),0,1)
    return np.unique(np.r_[np.linspace(0,1,count),events,0.,1.])


def reconstruct(geometry:QuinticGeometry,controls:np.ndarray,s:np.ndarray)->dict:
    coeff=np.asarray(controls,dtype=float)
    s=np.asarray(s,dtype=float)
    if coeff.ndim!=1 or len(coeff)<8 or not np.isfinite(coeff).all():
        raise ValueError("invalid squared-speed coefficients")
    knots=np.r_[np.zeros(3),np.linspace(0,1,len(coeff)-2),np.ones(3)]
    xfun=BSpline(knots,coeff,3,extrapolate=False)
    x=np.asarray(xfun(s),dtype=float)
    if (np.any(~np.isfinite(x)) or x.min()<-1e-8 or
        np.any(x[1:-1]<=0)):
        raise ValueError("nonpositive path interior squared speed")
    x=np.maximum(x,0)
    dx=np.asarray(xfun(s,nu=1),dtype=float)
    ddx=np.asarray(xfun(s,nu=2),dtype=float)
    q,qs,qss,qsss=geometry.evaluate(s)
    speed=np.sqrt(x)
    qd=qs*speed[:,None]
    qdd=qss*x[:,None]+.5*qs*dx[:,None]
    jerk=(qsss*(x*speed)[:,None]
          +1.5*qss*(dx*speed)[:,None]
          +.5*qs*(ddx*speed)[:,None])
    return {"progress":s,"q":q,"qd":qd,"qdd":qdd,
            "jerk":jerk,"x":x,"dx":dx,"ddx":ddx}


def elapsed_time(controls:np.ndarray,s:np.ndarray)->float:
    coeff=np.asarray(controls,dtype=float)
    knots=np.r_[np.zeros(3),np.linspace(0,1,len(coeff)-2),np.ones(3)]
    fn=BSpline(knots,coeff,3,extrapolate=False)
    from numpy.polynomial.legendre import leggauss
    nodes,weights=leggauss(8)
    length=np.diff(s)
    sample=s[:-1,None]+length[:,None]*(nodes[None]+1.)/2
    x=fn(sample)
    if np.any(x<=0) or not np.isfinite(x).all():
        return float("inf")
    return float(np.sum(length*np.sum(weights[None]/(2*np.sqrt(x)),axis=1)))


def max_bounds(values:np.ndarray,bounds:np.ndarray,*,scaled:bool)->dict:
    bound=np.asarray(bounds,dtype=float)
    if bound.shape!=(2,7) or np.asarray(values).shape[-1]!=7:
        raise ValueError("invalid bound or family")
    if scaled:
        # Position bounds are checked separately, so these are the four
        # signed dynamic families. Match historical Scaled95 convention:
        # torque scales towards zero, kinematics about midpoint.
        center=(bound[0]+bound[1])/2
        bound=np.stack((center+.95*(bound[0]-center),
                        center+.95*(bound[1]-center)))
    low=bound[0][None]-values
    high=values-bound[1][None]
    violation=np.maximum(low,high)
    i,j=np.unravel_index(np.argmax(violation),violation.shape)
    range_width=np.maximum(bound[1]-bound[0],1e-12)
    # Signed torque limits may be asymmetric; torque scaling centered
    # at zero is handled by the caller.
    return {"max_violation":float(max(0,violation[i,j])),
            "normalized_margin":float(np.min(np.minimum(
                values-bound[0],bound[1]-values)/range_width)),
            "limiting_joint_1_based":int(j+1),"limiting_sample":int(i),
            "limiting_side":"upper" if high[i,j]>=low[i,j] else "lower"}


def _torques(values:dict,payload:Payload)->np.ndarray:
    model,data,neutral,qi,vi=pinocchio_payload_model(payload)
    out=np.empty_like(values["q"])
    v=np.zeros(model.nv)
    a=np.zeros(model.nv)
    for i,q in enumerate(values["q"]):
        neutral[qi]=q
        v.fill(0);a.fill(0)
        v[vi]=values["qd"][i]
        a[vi]=values["qdd"][i]
        out[i]=pin.rnea(model,data,neutral,v,a)[vi]
    return out


def audit(req:dict,resp:dict,scene:str|None,*,grid_count:int)->dict:
    check_response(req,resp)
    if resp["status"]!="FEASIBLE_SCREENED":
        return {"status":"SKIPPED_NOT_SCREENED_FEASIBLE"}
    path=req["path"]
    geometry=QuinticGeometry(np.asarray(path["controls"],dtype=float))
    xcontrols=np.asarray(resp["timing_coefficients"],dtype=float)
    if len(xcontrols)!=107:
        raise ValueError("K107 must return exactly 107 timing coefficients")
    s=event_grid(geometry,len(xcontrols),grid_count)
    sampled=reconstruct(geometry,xcontrols,s)
    duration=elapsed_time(xcontrols,s)
    payload=Payload(**{k:req["payload"][k] for k in (
        "mass_kg","radius_m","native_link","pin_frame")})
    torques=_torques(sampled,payload)
    sampled["torque"]=torques
    auditor=IndependentGeometryAuditor(scene)
    geomcheck=auditor.check(sampled["q"],min_clearance_m=.005)
    payload_clearance=_payload_world_clearance(
        auditor,pinocchio_payload_model(payload),sampled["q"],payload.radius_m)
    families={}
    for family in CONSTRAINT_FAMILIES:
        bounds=np.asarray(req["constraints"][family],dtype=float)
        if family=="torque":
            # Native TimeOpt Scaled95 shrinks signed actuator bounds
            # toward zero rather than about their midpoint.
            m95=max_bounds(sampled[family],bounds*.95,scaled=False)
        else:
            m95=max_bounds(sampled[{"velocity":"qd","acceleration":"qdd",
                                    "jerk":"jerk"}[family]],bounds,scaled=True)
        values=sampled["torque"] if family=="torque" else sampled[
            {"velocity":"qd","acceleration":"qdd","jerk":"jerk"}[family]]
        physical=max_bounds(values,bounds,scaled=False)
        families[family]={"physical":physical,"scaled95":m95}
    endpoint_velocity=float(np.max(abs(sampled["qd"][[0,-1]])))
    endpoint_pos=float(max(
        np.max(abs(sampled["q"][0]-np.asarray(path["endpoint_q"][0]))),
        np.max(abs(sampled["q"][-1]-np.asarray(path["endpoint_q"][1])))))
    limits=np.asarray(
        __import__("research.joint_time_oracle.timeopt_payload_model",
                   fromlist=["nominal_effort_bounds"]).nominal_effort_bounds()
    )
    if not np.array_equal(limits,np.asarray(req["constraints"]["torque"])):
        raise ValueError("original nominal torque limits mismatch")
    success=bool(np.isfinite(duration) and duration>0 and
        all(row[typ]["max_violation"]<=MAX_SAMPLED_VIOLATION
            for row in families.values() for typ in ("physical","scaled95"))
        and geomcheck["sampled_acceptable"] and payload_clearance>=.005
        and endpoint_velocity<=1e-5 and endpoint_pos<=1e-6)
    return {"status":"SAMPLED_PHYSICAL_SCALED95_PASS" if success else
                     "INDEPENDENT_PHYSICAL_OR_SCALED95_FAIL",
            "grid_count_requested":grid_count,
            "actual_sample_count":len(s),
            "duration_s":duration,
            "online_duration_s":resp["duration_s"],
            "relative_online_duration_error":abs(duration/resp["duration_s"]-1),
            "families":families,
            "native_sphere_geometry_clearance_m":geomcheck["world_clearance_m"],
            "self_sphere_clearance_m":geomcheck["self_clearance_m"],
            "payload_world_clearance_m":payload_clearance,
            "endpoint_velocity_inf":endpoint_velocity,
            "endpoint_position_inf":endpoint_pos,
            "geometric_path_sha256":path["path_sha256"],
            "task_sha256":req["task_sha256"],
            "payload_sha256":req["payload"]["payload_sha256"],
            "independent_qualified_sampled":success,
            "scope":"independent Pinocchio S8193/S32768 sampled and event-aware; no continuous-time torque/collision certificate"}


def qualify(folder:Path,source_tasks:Path)->dict:
    source=json.loads(source_tasks.read_text())
    tasks={t["case_id"]:t for t in source}
    rows=[]
    for reqfile in sorted((folder/"oracle").glob("*.request.json")):
        req=json.loads(reqfile.read_text())
        if req["request_id"]=="warmup_000":
            continue
        resfile=reqfile.with_name(reqfile.name.replace(".request.json",".response.json"))
        if not resfile.exists():continue
        res=json.loads(resfile.read_text())
        if res.get("status")!="FEASIBLE_SCREENED":continue
        scene=tasks[req["task_id"]]["scene"]
        record={"request_file_sha256":file_sha256(reqfile),
                "response_file_sha256":file_sha256(resfile),
                "request_id":req["request_id"],
                "path_sha256":req["path"]["path_sha256"]}
        try:
            low=audit(req,res,scene,grid_count=GRIDS[0])
            high=audit(req,res,scene,grid_count=GRIDS[1]) if (
                low["independent_qualified_sampled"]) else None
            record.update({"S8193":low,"S32768":high,
                "qualified":bool(high is not None and
                                 high["independent_qualified_sampled"])})
        except Exception as exc:
            record.update({"qualified":False,"status":"AUDIT_ERROR",
                           "reason":repr(exc)})
        rows.append(record)
    output={"schema_version":1,"candidate_count":len(rows),
            "qualified_count":sum(x["qualified"] for x in rows),
            "candidates":rows,
            "warning":"source task/scene and payload parity receipts must be frozen and supplied for full benchmark"}
    out=folder/"independent_k107_qualification.json"
    if out.exists() and json.loads(out.read_text())!=output:
        raise ValueError("immutable K107 qualification changed")
    if not out.exists():out.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm-case-folder",type=Path,required=True)
    p.add_argument("--source-tasks",type=Path,required=True)
    a=p.parse_args()
    result=qualify(a.arm_case_folder,a.source_tasks)
    print(json.dumps({"qualified":result["qualified_count"],
                      "candidates":result["candidate_count"]},indent=2))


if __name__=="__main__":
    main()
