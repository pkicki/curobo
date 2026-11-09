# SPDX-License-Identifier: Apache-2.0
"""Margin-aware fixed-geometry retiming of frozen IPOPT P trajectories.

Two distinct arms, never altering source W/P files:
  dilation: registered finite rho grid, x_rho=x_P/rho^2. Screen using
            aligned independent Pinocchio and the analytic torque law.
  tighten_bXXX: IPOPT re-optimization with all signed Scaled95 dynamic
            bounds tightened by beta. Screen the result independently.

All R outputs are PRELIMINARY until separate cohort_validate physical
and Scaled95 dense+adaptive audit. Source P was physical-qualified but
NOT Scaled95 qualified. No continuous-time guarantee, global optimum,
real hardware actuator rating, or latency benefit is claimed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pinocchio as pin

from .cohort_geometry import QuinticGeometry
from .cohort_retime import ROOT, dynamics, problem, solve, states
from .torque_active import read_profile
from .torque_margin_core import (
    CASE_IDS, DILATION_FACTORS, TIGHTENING_FACTORS,
    immutable_json, pilot_limits, sha256_file, tighten_pointwise_problem,
)
from .torque_margin_prepare import OUTPUT


def source_contract(root:Path,case:str)->tuple[Path,dict,dict]:
    if case not in CASE_IDS:
        raise ValueError("unregistered source case")
    registration=json.loads((root/"margin_registration.json").read_text())
    source={row["case_id"]:row for row in registration["sources"]}[case]
    src=Path(registration["source_root"])/registration["source_arm"]/case
    for filename,digest in source["source_files_sha256"].items():
        if sha256_file(src/filename)!=digest:
            raise ValueError("frozen P/W source checksum mismatch: "+filename)
    profile=read_profile(src/"torque_profile.json")
    if profile["profile_sha256"]!=source["profile_sha256"]:
        raise ValueError("source effort profile mismatch")
    return src,source,profile


def baseline_audit_samples(sample:dict,*,limit:int=4097)->dict:
    """Compute independent RNEA tau and gravity ONCE for deterministic rho scan."""
    if not sample or "q" not in sample:
        raise ValueError("missing IPOPT P sampled trajectory")
    total=len(sample["q"])
    idx=np.unique(np.round(np.linspace(0,total-1,min(total,limit))).astype(int))
    selected={k:np.asarray(v)[idx] if np.asarray(v).shape[:1]==(total,)
              else np.asarray(v) for k,v in sample.items()}
    model,data,neutral,qi,vi=dynamics()
    q=neutral.copy()
    v=np.zeros(model.nv);a=np.zeros(model.nv)
    tau=np.empty_like(selected["q"],dtype=float)
    g=np.empty_like(tau)
    for k,positions in enumerate(selected["q"]):
        q[qi]=positions
        v.fill(0.);a.fill(0.)
        g[k]=pin.rnea(model,data,q,v,a)[vi]
        v[vi]=selected["qd"][k]
        a[vi]=selected["qdd"][k]
        tau[k]=pin.rnea(model,data,q,v,a)[vi]
    selected["gravity"]=g
    return {"selected":selected,"torque":tau,"sampling_indices":idx.tolist(),
            "samples":len(idx),"source_sample_count":total,
            "scope":"independent Pinocchio sampled source P inverse dynamics and gravity"}


def physical_limits(profile:dict)->dict:
    baseline=ROOT/"research/joint_time_oracle/results/native_reconstruction_20261106/baseline_validation_states.npz"
    with np.load(baseline) as ref:
        bounds={name:np.asarray(ref[name+"_bounds"]) for name in
                ("position","velocity","acceleration","jerk")}
        native=np.asarray(ref["torque_bounds"])
    expected=np.stack((profile["nominal_effort_lower"],profile["nominal_effort_upper"]))
    if not np.allclose(native,expected,atol=1e-5,rtol=1e-5):
        raise ValueError("independent nominal dynamics limits changed")
    bounds["torque"]=np.stack((profile["effort_lower"],profile["effort_upper"]))
    return bounds


def store_r(arm:Path,case:str,states_r:dict,source:dict,source_geometry:dict,
            *,method:str,extra:dict)->dict:
    dest=arm/case
    dest.mkdir(parents=True,exist_ok=True)
    path=dest/"r_dense.npz"
    metadata=dest/"r_registration.json"
    if path.exists() or metadata.exists():
        raise ValueError("frozen R already exists; do not overwrite")
    np.savez_compressed(path,**states_r)
    geometry_file=dest/"r_geometry.json"
    geometry_file.write_text(json.dumps({
        "controls":source_geometry["controls"],
        "source":"SAME_FROZEN_P_W_QUINTIC_GEOMETRY",
        "task_sha256":source["task_sha256"],
        "profile_sha256":source["profile_sha256"]},indent=2)+"\n")
    document={"schema_version":1,"case_id":case,"method":method,
              "source_files_sha256":source["source_files_sha256"],
              "source_W_duration_s":source["source_W_duration_s"],
              "source_P_duration_s":source["source_P_duration_s"],
              "profile_sha256":source["profile_sha256"],
              "source_case_sha256":source["task_sha256"],
              "r_sha256":sha256_file(path),
              "r_geometry_sha256":sha256_file(geometry_file),
              "r_duration_s":float(states_r["time"][-1]),
              "stage":"PRELIMINARY; must independently validate R and R_adaptive",
              **extra}
    immutable_json(metadata,document)
    return document


def reeval(geometry:QuinticGeometry,coeff:np.ndarray,*,samples:int=8193,
           extra=None)->dict:
    if not np.isfinite(coeff).all() or np.min(coeff)<=0:
        raise ValueError("nonpositive/nonfinite squared-speed controls")
    return states(geometry.evaluate,coeff,count=samples,
                  knots=geometry.spline.t,extra=extra)


def dilation(root:Path,case:str)->dict:
    src,source,profile=source_contract(root,case)
    with np.load(src/"p_timing_controls.npz") as z:
        controls=np.asarray(z["squared_speed_controls"],dtype=float)
    geo=json.loads((src/"w_geometry.json").read_text())
    geometry=QuinticGeometry(np.asarray(geo["controls"],dtype=float))
    with np.load(src/"p_refined_dense.npz") as z:
        sample={k:np.asarray(z[k]) for k in z.files}
    baseline=baseline_audit_samples(sample)
    bounds=physical_limits(profile)
    outcome={"case_id":case,"method":"dilation","source_P_duration_s":
             float(sample["time"][-1]),"source_P_controls_sha256":
             source["source_files_sha256"]["p_timing_controls.npz"],
             "screen_samples":baseline["samples"],"attempts":[],
             "selected_rho":None,"selected_R_qualified":False,
             "timing_search":"frozen nonmonotone-safe finite rho grid"}
    winner=None
    stamp=time.perf_counter()
    for rho in DILATION_FACTORS:
        row=pilot_limits(baseline["selected"],baseline["torque"],bounds,rho)
        outcome["attempts"].append(row)
        if row["conservative_screen_pass"]:
            winner=rho
            break
    outcome["screen_wall_s"]=time.perf_counter()-stamp
    arm=root/"dilation"
    if winner is not None:
        initial=controls/winner**2
        reconstructed=reeval(geometry,initial)
        verification={"rho":winner,"scaled_controls_min":float(initial.min()),
                      "source_W_qualified":True}
        record=store_r(arm,case,reconstructed,source,geo,
                       method="dilation",extra=verification)
        outcome.update({"selected_rho":winner,"r_sha256":record["r_sha256"],
                        "preliminary_R_duration_s":record["r_duration_s"]})
        np.savez_compressed(arm/case/"r_speed_controls.npz",
                            squared_speed_controls=initial,
                            original_p_controls=controls,
                            rho=np.array(winner))
    else:
        outcome["reason"]="NO_REGISTERED_DILATION_PASSED_SAMPLE_MARGIN_GATE"
    immutable_json(arm/case/"screen_result.json",outcome)
    return outcome


def tightening(root:Path,case:str,beta:float,*,maxiter:int=200,grid:int=513)->dict:
    if beta not in TIGHTENING_FACTORS:
        raise ValueError("unregistered beta")
    src,source,profile=source_contract(root,case)
    geo=json.loads((src/"w_geometry.json").read_text())
    geometry=QuinticGeometry(np.asarray(geo["controls"],dtype=float))
    with np.load(src/"p_timing_controls.npz") as z:
        pcontrols=np.asarray(z["squared_speed_controls"],dtype=float)
    name=f"tight_b{int(round(beta*1000)):03d}"
    arm=root/name
    record={"case_id":case,"method":name,"beta":beta,
            "source_P_controls_sha256":source["source_files_sha256"]["p_timing_controls.npz"],
            "qualified":False,"registered_maxiter":maxiter,"grid":grid}
    profile_path=src/"torque_profile.json"
    previous=os.environ.get("ORACLE_TORQUE_PROFILE")
    if previous is not None:
        raise ValueError("ambient torque-profile override forbidden")
    try:
        os.environ["ORACLE_TORQUE_PROFILE"]=str(profile_path.resolve())
        start=time.perf_counter()
        p=problem(geometry.evaluate,geometry.spline.t,grid,pcontrols)
        p=tighten_pointwise_problem(p,beta)
        # P is a much better starting shape than zero speed, and beta
        # reduces the admissible envelope without altering geometry.
        solution,information=solve(p,pcontrols,maxiter=maxiter)
        record["complete_solver_wall_s"]=time.perf_counter()-start
        record["solver"]=information
        if (information.get("solver_converged") and
            information.get("primal_min_margin",-1e9)>=-1e-6 and
            np.isfinite(solution).all() and np.min(solution)>0):
            result=reeval(geometry,solution)
            with np.load(src/"p_dense.npz") as z:
                source_p={k:np.asarray(z[k]) for k in z.files}
            # Exact geometry must remain unchanged, even after time optimization.
            if (np.max(np.abs(result["q"][0]-source_p["q"][0]))>1e-7 or
                np.max(np.abs(result["q"][-1]-source_p["q"][-1]))>1e-7):
                raise ValueError("retimed source geometry endpoints changed")
            candidate=store_r(arm,case,result,source,geo,
                              method=name,extra={"beta":beta})
            np.savez_compressed(arm/case/"r_speed_controls.npz",
                                squared_speed_controls=solution,
                                initial_p_controls=pcontrols,
                                beta=np.array(beta))
            record["r_sha256"]=candidate["r_sha256"]
            record["preliminary_R_duration_s"]=candidate["r_duration_s"]
        else:
            record["reason"]="IPOPT_UNQUALIFIED_OR_INFEASIBLE"
    except Exception as exc:
        record["status"]="NUMERICAL_OR_RUNTIME_ERROR"
        record["reason"]=repr(exc)
    finally:
        os.environ.pop("ORACLE_TORQUE_PROFILE",None)
    immutable_json(arm/case/"screen_result.json",record)
    return record


def rebuild_arm_inventory(root:Path,arm_name:str):
    registered=json.loads((root/"source_tasks.json").read_text())
    arm=root/arm_name
    tasks=[t for t in registered if (arm/t["case_id"]/"r_dense.npz").is_file()]
    task_file=arm/"tasks.json"
    if task_file.exists():
        old=json.loads(task_file.read_text())
        new_by_id={t["case_id"]:t for t in tasks}
        if any(new_by_id.get(x["case_id"])!=x for x in old):
            raise ValueError("previously recorded retiming task changed or disappeared")
    task_file.write_text(json.dumps(tasks,indent=2)+"\n")
    pilot=[{"case_id":t["case_id"],"population":t["population"],
            "cohort":"margin_retiming","task_sha256":t["task_sha256"],
            "methods":{"R":{"status":"PRELIMINARY",
                      "duration_s":json.loads(
                         (arm/t["case_id"]/"r_registration.json").read_text()
                      )["r_duration_s"],"verification":{}}}} for t in tasks]
    file=arm/"pilot.jsonl"
    payload="".join(json.dumps(x)+"\n" for x in pilot)
    if file.exists():
        old_rows=[json.loads(line) for line in file.read_text().splitlines() if line.strip()]
        new_rows={r["case_id"]:r for r in pilot}
        if any(new_rows.get(x["case_id"])!=x for x in old_rows):
            raise ValueError("previous retiming evaluation row changed or disappeared")
    # Append-only inclusion permits prospective holdouts after auditing the
    # development case; rerun independent audits on expanded full inventory.
    file.write_text(payload)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUTPUT)
    p.add_argument("--case",required=True,choices=CASE_IDS)
    p.add_argument("--method",required=True,choices=("dilation","tighten"))
    p.add_argument("--beta",type=float,choices=TIGHTENING_FACTORS)
    p.add_argument("--maxiter",type=int,default=200)
    p.add_argument("--grid",type=int,default=513)
    p.add_argument("--holdout",action="store_true",
                   help="Required for table/cluttered confirmation after method freeze.")
    a=p.parse_args()
    if a.case!=CASE_IDS[0]:
        if not a.holdout:
            p.error("holdout case requires --holdout")
        from .torque_margin_freeze import verify
        verify(a.root)
    elif a.holdout:
        p.error("development case must not use --holdout")
    if a.method=="dilation":
        result=dilation(a.root,a.case)
        arm="dilation"
    else:
        if a.beta is None:
            p.error("tighten requires --beta")
        result=tightening(a.root,a.case,a.beta,maxiter=a.maxiter,grid=a.grid)
        arm=f"tight_b{int(round(a.beta*1000)):03d}"
    rebuild_arm_inventory(a.root,arm)
    print(json.dumps({k:v for k,v in result.items() if k!="attempts"},
                     indent=2,default=str))


if __name__=="__main__":
    main()
