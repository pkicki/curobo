# SPDX-License-Identifier: Apache-2.0
"""Prepare frozen extreme torque limits and time-dilated alternative witnesses.

Run in model-aligned independent Pinocchio env. Reuses EXACT frozen D geometry
and its original physical-time samples. A passed torque PRESCREEN is NOT a
validated witness; run exact-spline adaptive replay and independent Pinocchio
RNEA/world/self collision checks afterward.

This preparatory phase must finish and be frozen BEFORE checking whether the
full native planner succeeds under any of the new profiles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from .cohort_retime import dynamics
from .extreme_torque_core import (
    DEFAULT_FACTORS, DEFAULT_SLOWDOWNS, TORQUE_HEADROOM,
    derived_profile, dilate_states, regime_name,
    screen_slowdowns, sha256_file, source_manifest,
)
from .torque_active import read_profile, write_profile

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"research/joint_time_oracle/results/torque_stress_final_20261107/severe"
OUTPUT=ROOT/"research/joint_time_oracle/results/extreme_torque_witness_20261108"


def independent_torque_and_gravity(states:dict) -> tuple[np.ndarray,np.ndarray]:
    model,data,q,qi,vi=dynamics()
    qsrc=np.asarray(states["q"],dtype=np.float64)
    vsrc=np.asarray(states["qd"],dtype=np.float64)
    asrc=np.asarray(states["qdd"],dtype=np.float64)
    if qsrc.ndim!=2 or qsrc.shape[1]!=7 or vsrc.shape!=qsrc or asrc.shape!=qsrc:
        raise ValueError("source path dimensions mismatch")
    gravity=np.empty_like(qsrc);tau=np.empty_like(qsrc)
    v=np.zeros(model.nv);a=np.zeros(model.nv)
    for k in range(len(qsrc)):
        q[qi]=qsrc[k]
        v.fill(0);a.fill(0)
        gravity[k]=pin.rnea(model,data,q,v,a)[vi]
        v[vi]=vsrc[k];a[vi]=asrc[k]
        tau[k]=pin.rnea(model,data,q,v,a)[vi]
    return tau,gravity


def freeze(source:Path,output:Path,*,factors=DEFAULT_FACTORS,
           slowdowns=DEFAULT_SLOWDOWNS) -> dict:
    source=source.resolve();output=output.resolve()
    tasks=json.loads((source/"tasks.json").read_text())
    if not tasks or len({x["case_id"] for x in tasks})!=len(tasks):
        raise ValueError("empty or duplicate source case inventory")
    source_sha=sha256_file(source/"tasks.json")
    source_receipt_path=source/"adaptive_receipts.json"
    source_receipts=json.loads(source_receipt_path.read_text()) if source_receipt_path.exists() else {}
    plan={"source_tasks_sha256":source_sha,
          "factor_grid":list(factors),"slowdown_grid":list(slowdowns),
          "independent_dynamics":"model-aligned Pinocchio RNEA, unchanged Panda inertials",
          "selection":"shortest preregistered slowdown satisfying Scaled95 torque with margin",
          "additional_headroom":TORQUE_HEADROOM,
          "source_geometry":"frozen severe D, NOT regenerated after native outcomes",
          "endpoint_contract":"rest-to-rest; no nonzero endpoint derivative contracts",
          "proof_level":"prescreen only; independent exact-spline adaptive validation required",
          "comparison":"same original scene, endpoints, model, and dynamic limits except torque"}
    manifest_file=output/"registration.json"
    if manifest_file.exists() and json.loads(manifest_file.read_text())!=plan:
        raise ValueError("different experiment already frozen in output directory")
    output.mkdir(parents=True,exist_ok=True)
    manifest_file.write_text(json.dumps(plan,indent=2)+"\n")
    profiles_by_arm={regime_name(float(f)):[] for f in factors}
    attempts={regime_name(float(f)):[] for f in factors}
    tasks_by_arm={regime_name(float(f)):[] for f in factors}
    rows_by_arm={regime_name(float(f)):[] for f in factors}
    for task in tasks:
        case=task["case_id"]
        original_profile=read_profile(source/case/"torque_profile.json")
        if original_profile["source_case_id"]!=case:
            raise ValueError("original profile/case mismatch")
        historical_receipt=source_receipts.get(case+":D",{})
        old_valid=(historical_receipt.get("physical") is True and
                   historical_receipt.get("torque_profile_sha256")==original_profile["profile_sha256"])
        original_path=source/case/"d_dense.npz"
        original_audit=source/case/"d_pin_torque.npz"
        raw=None;reference_torque=None;gravity=None;source_error=None
        if not old_valid:
            source_error="frozen D lacks independent adaptive physical qualification receipt"
        try:
            if source_error:
                raise ValueError(source_error)
            with np.load(original_path) as data:
                raw={k:np.asarray(data[k]) for k in data.files}
            reference_torque,gravity=independent_torque_and_gravity(raw)
            if not original_audit.exists():
                raise ValueError("missing frozen independent D torque audit")
            with np.load(original_audit) as data:
                reference_time=np.asarray(data["time"])
                saved_torque=np.asarray(data["torque"])
            if (reference_time.shape!=raw["time"].shape or
                not np.allclose(reference_time,raw["time"],atol=1e-7,rtol=1e-7)):
                raise ValueError("frozen torque audit time grid differs from D source")
            residual=float(np.max(abs(saved_torque-reference_torque)))
            if residual>1e-4:
                raise ValueError(f"independent RNEA audit mismatch {residual} Nm")
        except (OSError,ValueError,KeyError) as exc:
            source_error=str(exc)
        for factor in factors:
            factor=float(factor)
            arm_name=regime_name(factor)
            arm=output/arm_name
            folder=arm/case
            folder.mkdir(parents=True,exist_ok=True)
            profile=derived_profile(original_profile,factor,source_sha=task["task_sha256"])
            write_profile(folder/"torque_profile.json",profile)
            profiles_by_arm[arm_name].append(profile.to_json())
            status={"case_id":case,"factor":factor,"admitted":profile.admitted,
                    "source_error":source_error,
                    "source_profile_sha256":original_profile["profile_sha256"],
                    "derived_profile_sha256":profile.to_json()["profile_sha256"],
                    "source_D_dense_sha256":sha256_file(original_path) if original_path.exists() else None,
                    "source_D_audit_sha256":sha256_file(original_audit) if original_audit.exists() else None,
                    "witness_screen_pass":False,
                    "screening_only":True}
            if not profile.admitted:
                status["reason"]=profile.rejection_reason
            else:
                tasks_by_arm[arm_name].append(task)
                if raw is None or source_error:
                    status["reason"]=source_error or "missing D trajectory"
                else:
                    effort=np.stack((profile.effort_lower,profile.effort_upper))
                    rho,checked=screen_slowdowns(reference_torque,gravity,effort,
                                                 slowdowns=slowdowns)
                    status["candidate_slowdowns"]=checked
                    status["witness_screen_pass"]=rho is not None
                    if rho is None:
                        status["reason"]="no preregistered slowdown passes Scaled95 torque prescreen"
                    else:
                        states=dilate_states(raw,rho)
                        wfile=folder/"w_dense.npz"
                        if wfile.exists():
                            # Preserve proof provenance: never overwrite witnesses in a resumed study.
                            with np.load(wfile) as previous:
                                for key in ("time","q","qd","qdd","jerk","progress","controls","knots"):
                                    if key in states:
                                        if key not in previous or previous[key].shape!=states[key].shape or not np.allclose(
                                            previous[key],states[key],rtol=0,atol=1e-12):
                                            raise ValueError("existing witness differs; use a new output root")
                        else:
                            np.savez_compressed(wfile,**states)
                        status.update({"witness_slowdown":rho,
                                       "witness_duration_s":float(states["time"][-1]-states["time"][0]),
                                       "witness_dense_sha256":sha256_file(wfile),
                                       "nominal_source_duration_s":float(raw["time"][-1]-raw["time"][0]),
                                       "prescreen_scaled95_peak":next(x["predicted_scaled95_torque_utilization"] for x in checked if x["slowdown"]==rho)})
                        (folder/"w_source.json").write_text(json.dumps(status,indent=2)+"\n")
            if not (folder/"w_source.json").exists():
                (folder/"w_source.json").write_text(json.dumps(status,indent=2)+"\n")
            attempts[arm_name].append(status)
            rows_by_arm[arm_name].append({
                "case_id":case,"cohort":"pilot","population":task["population"],
                "task_sha256":task["task_sha256"],
                "methods":{"W":{"duration_s":status.get("witness_duration_s"),
                                 "verification":{},"status":"PRESCREEN_ONLY" if status["witness_screen_pass"]
                                 else "NO_WITNESS"}}})
        print(case,{"admitted":sum(v[-1]["admitted"] for v in attempts.values()),
                    "screened":sum(v[-1]["witness_screen_pass"] for v in attempts.values())},flush=True)
    registry={}
    for name in sorted(tasks_by_arm):
        arm=output/name
        (arm/"tasks.json").write_text(json.dumps(tasks_by_arm[name],indent=2)+"\n")
        accepted={x["case_id"] for x in tasks_by_arm[name]}
        (arm/"pilot.jsonl").write_text("".join(
            json.dumps(row)+"\n" for row in rows_by_arm[name] if row["case_id"] in accepted))
        (arm/"attempt_inventory.json").write_text(json.dumps(attempts[name],indent=2)+"\n")
        (arm/"all_profiles.json").write_text(json.dumps(profiles_by_arm[name],indent=2)+"\n")
        registry[name]={"factor":int(name[-4:])/1000,
                        "attempted":len(tasks),"endpoint_admitted":len(accepted),
                        "witness_prescreen_pass":sum(x["witness_screen_pass"] for x in attempts[name]),
                        "tasks_sha256":sha256_file(arm/"tasks.json")}
    summary=source_manifest(source,list(registry.values()),plan)
    summary["arms"]=registry
    # Hash covers the complete manifest, not just the preregistered input contract.
    summary.pop("manifest_sha256",None)
    from .torque_active import canonical_hash
    summary["manifest_sha256"]=canonical_hash(summary)
    (output/"manifest.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=SOURCE)
    p.add_argument("--output",type=Path,default=OUTPUT)
    args=p.parse_args()
    summary=freeze(args.source,args.output)
    print(json.dumps(summary["arms"],indent=2))


if __name__=="__main__":
    main()
