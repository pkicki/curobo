# SPDX-License-Identifier: Apache-2.0
"""Fail-closed independent sampled witness qualification and frozen SHA checks.

A constructive smoothstep trajectory is accepted ONLY after independent
cohort_validate Pinocchio/sphere audits of 8193 original samples and the
additional exact-analytic adaptive samples. The receipt is *not* a
continuous-time mathematical feasibility certificate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cohort_torque_report import TOLERANCES, ENDPOINT_TOL
from .geometry_feasibility_core import sha256_file
from .geometry_feasibility_discover import OUT
from .torque_active import read_profile


def checks(diag:dict,mode:str):
    return {family:bool(
        isinstance(diag.get("families",{}).get(family,{}).get(mode,{}).get("violation"),(int,float))
        and np.isfinite(diag["families"][family][mode]["violation"])
        and 0<=diag["families"][family][mode]["violation"]<=tol)
        for family,tol in TOLERANCES.items()}


def issue(arm:Path)->dict:
    base_path=arm/"independent_sampled.json"
    adaptive_path=arm/"independent_adaptive.json"
    if not base_path.exists() or not adaptive_path.exists():
        raise ValueError("both independent base and adaptive audits required")
    base=json.loads(base_path.read_text())
    adaptive=json.loads(adaptive_path.read_text())
    tasks=json.loads((arm/"tasks.json").read_text())
    out={}
    for task in tasks:
        case=task["case_id"];folder=arm/case
        result={"case_id":case,"physical":False,"scaled95":False,
                "collision":False,"endpoint":False,
                "reasons":[],"scope":"dense+adaptive sampled Pinocchio/spheres; NOT continuous-time certificate"}
        reasons=result["reasons"]
        try:
            prof=read_profile(folder/"torque_profile.json")
            result["torque_profile_sha256"]=prof["profile_sha256"]
            if prof["source_task_sha256"]!=task["task_sha256"]:
                reasons.append("task/profile hash mismatch")
            discovery=json.loads((folder/"discovery.json").read_text())
            if discovery["status"]!="CONSTRUCTIVE_WITNESS_PRESCREEN":
                reasons.append("not a registered prescreened witness")
            bfile=folder/"w_dense.npz";afile=folder/"w_adaptive_dense.npz"
            result["base_sha256"]=sha256_file(bfile)
            result["adaptive_sha256"]=sha256_file(afile)
            winning=discovery["candidate_attempts"][-1]
            if winning.get("witness_sha256")!=result["base_sha256"]:
                reasons.append("source witness SHA changed")
            geomfile=folder/"w_geometry.json"
            result["witness_geometry_sha256"]=sha256_file(geomfile)
            if winning.get("witness_geometry_sha256")!=result["witness_geometry_sha256"]:
                reasons.append("quintic controls SHA changed after candidate selection")
            if winning.get("arc_length_aligned_geometry_deviation_rad",0.)<0.10:
                reasons.append("selected alternative path not geometrically distinct")
            replay_file=folder/"w_adaptive_replay.json"
            replay=json.loads(replay_file.read_text()) if replay_file.exists() else {}
            if replay.get("status")!="REPLAYED":
                reasons.append("exact analytic adaptive replay missing or failed")
            if replay.get("adaptive_sha256")!=result["adaptive_sha256"]:
                reasons.append("adaptive replay SHA mismatch")
            b=base.get(case+":W");d=adaptive.get(case+":W_adaptive")
            if not isinstance(b,dict) or not isinstance(d,dict):
                reasons.append("missing independent base or adaptive audit")
            else:
                for name,diag in (("base",b),("adaptive",d)):
                    if diag.get("torque_profile_sha256")!=prof["profile_sha256"]:
                        reasons.append(name+" profile SHA mismatch")
                with np.load(bfile) as f:
                    bn=len(f["time"])
                with np.load(afile) as f:
                    an=len(f["time"])
                if bn<8193 or an<bn or b.get("samples")!=bn or d.get("samples")!=an:
                    reasons.append("independent sampled count mismatch")
                result["base_samples"]=bn;result["adaptive_samples"]=an
                end={}
                for key,tol in ENDPOINT_TOL.items():
                    value=d.get("endpoints",{}).get(key)
                    end[key]=bool(isinstance(value,(int,float)) and np.isfinite(value)
                                  and abs(value)<=tol)
                result["endpoint_checks"]=end
                result["endpoint"]=bool(not reasons and all(end.values()))
                collision_flags=[]
                for diag in (b,d):
                    world=diag.get("world_min_clearance_m")
                    selfclear=diag.get("self_min_clearance_m")
                    collision_flags.append(bool((world is None or
                        (np.isfinite(world) and world>=-1e-7))
                        and selfclear is not None and np.isfinite(selfclear)
                        and selfclear>=-1e-7))
                result["collision"]=bool(not reasons and all(collision_flags))
                family={mode:(checks(b,mode),checks(d,mode))
                        for mode in ("physical","scaled95")}
                result["family_checks"]={
                    mode:{"base":pair[0],"adaptive":pair[1]} for mode,pair in family.items()}
                physical=all(all(x.values()) for x in family["physical"])
                scaled=all(all(x.values()) for x in family["scaled95"])
                result["physical"]=bool(not reasons and result["endpoint"]
                                        and result["collision"] and physical)
                result["scaled95"]=bool(result["physical"] and scaled)
            if not result["physical"]:
                reasons.append("full independent physical validation failed")
            if not result["scaled95"]:
                reasons.append("full independent Scaled95 validation failed")
        except Exception as exc:
            reasons.append(repr(exc))
        out[case+":W"]=result
    (arm/"w_adaptive_receipts.json").write_text(json.dumps(out,indent=2)+"\n")
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUT)
    p.add_argument("--scenario",required=True)
    a=p.parse_args()
    result=issue(a.root/a.scenario)
    print(json.dumps({"total":len(result),
                      "physical":sum(x["physical"] for x in result.values()),
                      "scaled95":sum(x["scaled95"] for x in result.values())},indent=2))


if __name__=="__main__":
    main()
