# SPDX-License-Identifier: Apache-2.0
"""Independent post-mortem of captured native pre-TimeOpt geometries.

Run in the pinned Pinocchio process after the separate instrumented CUDA
planner run. A captured native cubic curve is not a path feasibility proof.
For every captured lane, this tool checks the relaxed torque LP along the
EXACT native cubic, and independently samples joint/world/self collision.
Never infer a specific pipeline failure stage from unavailable pre-TimeOpt
geometry or TimeOpt's success bit alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cohort_geometry import NativeGeometry
from .extreme_torque_path_lp import inspect_curve
from .geometry_feasibility_scene import IndependentGeometryAuditor
from .geometry_feasibility_core import sha256_file
from .torque_active import read_profile

def diagnose(native_arm:Path,profile_arm:Path,*,samples:int=257)->dict:
    tasks=json.loads((profile_arm/"tasks.json").read_text())
    outcomes={}
    for task in tasks:
        case=task["case_id"]
        folder=native_arm/case
        prof=read_profile(profile_arm/case/"torque_profile.json")
        limits=np.stack((prof["effort_lower"],prof["effort_upper"]))
        auditor=IndependentGeometryAuditor(task["scene"])
        inventories=sorted(folder.glob("f*_pretimeopt_inventory.json"))
        records=[]
        for inv in inventories:
            inventory=json.loads(inv.read_text())
            for item in inventory:
                result={"tag":item.get("tag"),"capture_state":item.get("state"),
                        "profile_sha256":prof["profile_sha256"],
                        "finalizer_returned":item.get("finalizer_returned"),
                        "timeopt_validation_masks":item.get("timeopt_validation_masks"),
                        "pre_snapshot_success":item.get("pre_success"),
                        "pre_snapshot_path_valid":item.get("pre_path_valid"),
                        "lanes":[]}
                # use the known snapshot filename to prevent path injection
                local=folder/(str(item["tag"])+"_native_geometry.npz")
                if not local.exists():
                    result["status"]="NO_CUBIC_GEOMETRY_CAPTURED"
                elif sha256_file(local)!=item.get("native_cubic_geometry_sha256"):
                    result["status"]="CAPTURE_SHA_MISMATCH"
                else:
                    with np.load(local) as z:
                        data={k:np.asarray(z[k]) for k in z.files}
                    for lane in range(data["controls"].shape[0]):
                        row={"lane":lane}
                        try:
                            geometry=NativeGeometry(data,lane)
                            lp=inspect_curve(geometry.evaluate_trimmed,limits,
                                             samples=samples)
                            positions=geometry.evaluate_trimmed(
                                np.linspace(0,1,257))[0]
                            scene=auditor.check(positions,min_clearance_m=.005)
                            row.update({"status":"DIAGNOSED","lp":lp,
                                        "position_violation_rad":scene["position_violation_rad"],
                                        "world_clearance_m":scene["world_clearance_m"],
                                        "self_clearance_m":scene["self_clearance_m"],
                                        "sampled_geometry_valid":scene["sampled_acceptable"],
                                        "native_success_flag":bool(data["success"][lane])
                                          if "success" in data else None})
                        except Exception as exc:
                            row.update({"status":"DIAGNOSTIC_ERROR","reason":repr(exc)})
                        result["lanes"].append(row)
                    result["status"]="DIAGNOSED_WITH_INDEPENDENT_LP"
                records.append(result)
        if not inventories:
            records.append({"status":"NO_PRETIMEOPT_INVOCATION_RECORDED",
                            "reason":"could fail before solver finalizer; not evidence of geometric infeasibility"})
        outcomes[case]=records
    out=native_arm/"pretimeopt_diagnostics.json"
    out.write_text(json.dumps(outcomes,indent=2)+"\n")
    return outcomes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-arm",type=Path,required=True,
                   help="instrumented arm/native_full root")
    p.add_argument("--profile-arm",type=Path,required=True,
                   help="source arm with frozen tasks and effort limits")
    p.add_argument("--samples",type=int,default=257)
    a=p.parse_args()
    result=diagnose(a.native_arm,a.profile_arm,samples=a.samples)
    print(json.dumps({"cases":len(result),
                      "recorded_invocations":sum(len(v) for v in result.values())},indent=2))


if __name__=="__main__":
    main()
