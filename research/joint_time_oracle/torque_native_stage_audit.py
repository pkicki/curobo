# SPDX-License-Identifier: Apache-2.0
"""Independent offline LP/collision audit of rejected native cubic lanes.

The instrumented native-stage replay may snapshot TimeOpt's geometric
cubic controls even when no output is accepted. Every such lane is
tested against the frozen torque limits using the same independent
Pinocchio path dynamics and sphere/scene collision model as the witness
study. This is a candidate-level diagnostic, NOT a demonstration that a
rejected path admits an integrated globally feasible timing.

Failure to capture any path is explicitly UNOBSERVABLE, not "infeasible".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cohort_geometry import NativeGeometry
from .cohort_retime import ROOT
from .geometry_feasibility_core import path_distance, sha256_file
from .geometry_feasibility_scene import IndependentGeometryAuditor
from .torque_dual_guidance import AlignedPointDynamics, normalized_bottlenecks
from .torque_full_planner import geometry_sha
from .torque_active import read_profile
from .torque_native_stage_trace import CASES, SCENARIO, OUTPUT


class TrimmedNativeAdapter:
    def __init__(self,native):
        self.native=native
    def evaluate(self,progress):
        return self.native.evaluate_trimmed(progress)


def audit_lane(buffers:dict,lane:int,profile:dict,scene:str,
               auditor:IndependentGeometryAuditor,rnea:AlignedPointDynamics,
               *,samples:int=129)->dict:
    native=NativeGeometry(buffers,lane)
    adapter=TrimmedNativeAdapter(native)
    s=np.linspace(.01,.99,samples)
    try:
        q=np.asarray(adapter.evaluate(np.linspace(0,1,samples+2))[0])
        clearance=auditor.check(q,min_clearance_m=0.)
        effort=np.stack((profile["effort_lower"],profile["effort_upper"]))
        coeff=np.stack([rnea.coefficients(adapter,float(progress))
                        for progress in s],axis=1)
        worst=normalized_bottlenecks(coeff,effort,top_k=1)[0]
        return {"status":"AUDITED",
                "geometry_sha256":geometry_sha(buffers,lane),
                "maximum_relaxed_torque_expansion":float(worst["expansion"]),
                "worst_progress":float(s[int(worst["index"])]),
                "world_clearance_m":clearance["world_clearance_m"],
                "self_clearance_m":clearance["self_clearance_m"],
                "position_violation_rad":clearance["position_violation_rad"],
                "sampled_collision_free":clearance["sampled_acceptable"],
                "note":"necessary path-local torque LP; no integrated timing or independent trajectory qualification"}
    except Exception as exc:
        return {"status":"INVALID_OR_NOT_RECONSTRUCTIBLE","reason":repr(exc)}


def run(case:str,output:Path,*,samples:int=129)->dict:
    if case not in CASES:
        raise ValueError("case is not in frozen qualified registry")
    source_name,discovery_name=CASES[case]
    source=ROOT/"research/joint_time_oracle/results"/source_name
    discovery=ROOT/"research/joint_time_oracle/results"/discovery_name
    task=next(x for x in json.loads((source/"tasks.json").read_text())
              if x["case_id"]==case)
    profile=read_profile(discovery/SCENARIO/case/"torque_profile.json")
    auditor=IndependentGeometryAuditor(task["scene"])
    rnea=AlignedPointDynamics()
    rows={}
    # Track exact cubic geometry hashes across ALL instrumented restarts,
    # not only within each restart's internal lanes.
    seen=set()
    folder=output/case
    for restart in sorted(x for x in folder.glob("F???") if x.is_dir()):
        record_file=restart/"trace_summary.json"
        if not record_file.exists():
            continue
        trace=json.loads(record_file.read_text())
        group=[]
        if trace.get("stage_trace_status")!="REPRODUCIBLE":
            rows[restart.name]={"status":"REPLAY_NOT_REPRODUCIBLE",
                                "trace_status":trace.get("stage_trace_status"),
                                "audited_lanes":[]}
            continue
        for snap in sorted(restart.glob("post_timeopt_*.npz")):
            with np.load(snap) as f:
                buffers={name:np.asarray(f[name]) for name in f.files}
            lane_count=len(buffers["controls"])
            results=[]
            for lane in range(lane_count):
                record=audit_lane(buffers,lane,profile,task["scene"],auditor,rnea,
                                  samples=samples)
                record.update({"lane":lane,"snapshot_file":snap.name,
                               "snapshot_sha256":sha256_file(snap)})
                if record.get("geometry_sha256") in seen:
                    record["duplicate_exact_geometry_in_case"]=True
                else:
                    record["duplicate_exact_geometry_in_case"]=False
                seen.add(record.get("geometry_sha256"))
                results.append(record)
            group.append({"snapshot":snap.name,"lanes":results})
        rows[restart.name]={
            "status":"GEOMETRIC_LANES_CAPTURED" if group else
                     "NO_POST_TIMEOPT_GEOMETRY_OBSERVABLE",
            "snapshots":group,"pre_timeopt_snapshot_count":
                trace.get("pre_timeopt_snapshots",0),
            "post_timeopt_snapshot_count":
                trace.get("post_timeopt_path_snapshots",0),
        }
    outfile=folder/"independent_native_stage_geometry_audit.json"
    outfile.write_text(json.dumps(rows,indent=2)+"\n")
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUTPUT)
    p.add_argument("--case",action="append",choices=tuple(CASES))
    p.add_argument("--samples",type=int,default=129)
    a=p.parse_args()
    records={}
    for case in a.case or list(CASES):
        records[case]=run(case,a.root,samples=a.samples)
    print(json.dumps({"cases":len(records),
                      "audited_restarts":sum(len(v) for v in records.values())},indent=2))


if __name__=="__main__":
    main()
