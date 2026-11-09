# SPDX-License-Identifier: Apache-2.0
"""Strict SHA-linked independent dense+adaptive R physical/Scaled95 receipts.

Never count an IPOPT convergence flag or approximate dilation pilot as
success. Accept only R states independently checked by cohort_validate
in a separate process on >=8193 base plus adaptive samples, with signed
torque bounds, world/self spheres, endpoint contracts and exact frozen
geometry/source SHA. These are sampled, not continuous-time certificates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cohort_torque_report import sampled_acceptance
from .geometry_feasibility_core import sha256_file
from .torque_active import read_profile
from .torque_margin_prepare import OUTPUT


def issue(arm:Path)->dict:
    tasks=json.loads((arm/"tasks.json").read_text())
    campaign_root=arm.parent
    frozen_root=json.loads((campaign_root/"margin_registration.json").read_text())
    from .cohort_retime import ROOT
    if any(sha256_file(ROOT/name)!=digest for name,digest
           in frozen_root["model_inputs_sha256"].items()):
        raise ValueError("frozen source robot dynamics/collision model changed")
    source_records={x["case_id"]:x for x in frozen_root["sources"]}
    source_arm=Path(frozen_root["source_root"])/frozen_root["source_arm"]
    base_file=arm/"independent_sampled.json"
    adaptive_file=arm/"independent_adaptive.json"
    if not base_file.exists() or not adaptive_file.exists():
        raise FileNotFoundError("both independent R sampled and R_adaptive audits required")
    base=json.loads(base_file.read_text())
    adaptive=json.loads(adaptive_file.read_text())
    results={}
    for task in tasks:
        case=task["case_id"]
        folder=arm/case
        reasons=[]
        record={"case_id":case,"physical":False,"scaled95":False,
                "reason":[],"scope":"independent dense+adaptive sampled only; NOT continuous-time certificate"}
        try:
            profile=read_profile(folder/"torque_profile.json")
            frozen=json.loads((folder/"r_registration.json").read_text())
            reference=source_records[case]
            source_folder=source_arm/case
            if any(sha256_file(source_folder/key)!=value for key,value
                   in reference["source_files_sha256"].items()):
                reasons.append("original W/P evidence changed after source freeze")
            if frozen["source_files_sha256"]!=reference["source_files_sha256"]:
                reasons.append("candidate derived from changed source P/W")
            original_geo=json.loads((source_folder/"w_geometry.json").read_text())
            proposed_geo=json.loads((folder/"r_geometry.json").read_text())
            if not np.array_equal(
                np.asarray(original_geo["controls"],dtype=float),
                np.asarray(proposed_geo["controls"],dtype=float)):
                reasons.append("fixed original W/P geometry changed")
            geom=folder/"r_geometry.json"
            dense=folder/"r_dense.npz"
            ad=folder/"r_adaptive_dense.npz"
            replay=json.loads((folder/"r_adaptive_replay.json").read_text())
            if profile["source_task_sha256"]!=task["task_sha256"] or (
                frozen["source_case_sha256"]!=task["task_sha256"]):
                reasons.append("task SHA mismatch")
            if profile["profile_sha256"]!=frozen["profile_sha256"]:
                reasons.append("source torque profile mismatch")
            if sha256_file(dense)!=frozen["r_sha256"] or (
                sha256_file(geom)!=frozen["r_geometry_sha256"]):
                reasons.append("frozen R trajectory or geometry changed")
            controls_file=folder/"r_speed_controls.npz"
            controls_receipt=json.loads((folder/"r_controls_receipt.json").read_text())
            if (controls_receipt.get("r_speed_controls_sha256")!=sha256_file(controls_file)
                or controls_receipt.get("r_dense_sha256")!=frozen["r_sha256"]
                or controls_receipt.get("source_P_controls_sha256")
                   !=reference["source_files_sha256"]["p_timing_controls.npz"]):
                reasons.append("R timing controls are not linked to frozen source P/W")
            if replay.get("status")!="REPLAYED" or (
                replay.get("source_r_sha256")!=frozen["r_sha256"]) or (
                replay.get("r_adaptive_sha256")!=sha256_file(ad)) or (
                replay.get("r_speed_controls_sha256")!=sha256_file(controls_file)):
                reasons.append("adaptive R reconstruction missing or changed")
            a=base.get(case+":R")
            b=adaptive.get(case+":R_adaptive")
            if not isinstance(a,dict) or not isinstance(b,dict):
                reasons.append("missing independent R or R_adaptive sample audit")
            else:
                with np.load(dense) as f:
                    n1=len(f["time"]);duration=float(f["time"][-1])
                with np.load(ad) as f:
                    n2=len(f["time"])
                if n1<8193 or n2<n1 or a.get("samples")!=n1 or b.get("samples")!=n2:
                    reasons.append("independent sample-count discrepancy")
                if a.get("source_trajectory_sha256")!=sha256_file(dense):
                    reasons.append("independent base validation is stale or unbound to R NPZ")
                if b.get("source_trajectory_sha256")!=sha256_file(ad):
                    reasons.append("independent adaptive validation is stale or unbound to R_adaptive NPZ")
                if duration>=frozen.get("source_W_duration_s",float("inf")):
                    # Source W duration must be obtained from the frozen
                    # input registration and compared by the report as well.
                    record["not_shorter_than_source_W"]=True
                valid1,why1=sampled_acceptance(a,profile,scaled=False)
                valid2,why2=sampled_acceptance(b,profile,scaled=False)
                valid3,why3=sampled_acceptance(a,profile,scaled=True)
                valid4,why4=sampled_acceptance(b,profile,scaled=True)
                record.update({"base_samples":n1,"adaptive_samples":n2,
                               "duration_s":duration,
                               "base_physical":valid1,"adaptive_physical":valid2,
                               "base_scaled95":valid3,"adaptive_scaled95":valid4,
                               "independent_reason_by_mode":{
                                   "physical_base":why1,"physical_adaptive":why2,
                                   "scaled95_base":why3,"scaled95_adaptive":why4}})
                record["physical"]=bool(not reasons and valid1 and valid2)
                record["scaled95"]=bool(record["physical"] and valid3 and valid4)
            record.update({"r_sha256":sha256_file(dense),
                           "r_adaptive_sha256":sha256_file(ad),
                           "r_geometry_sha256":sha256_file(geom),
                           "profile_sha256":profile["profile_sha256"]})
            if not record["physical"]:
                reasons.append("independent physical audit failed")
            if not record["scaled95"]:
                reasons.append("independent Scaled95 audit failed")
        except Exception as exc:
            reasons.append(repr(exc))
        record["reason"]=reasons
        results[case+":R"]=record
    (arm/"r_adaptive_receipts.json").write_text(json.dumps(results,indent=2)+"\n")
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUTPUT)
    p.add_argument("--arm",required=True)
    a=p.parse_args()
    records=issue(a.root/a.arm)
    print(json.dumps({"cases":len(records),
                      "qualified_physical":sum(r["physical"] for r in records.values()),
                      "qualified_scaled95":sum(r["scaled95"] for r in records.values())},indent=2))


if __name__=="__main__":
    main()
