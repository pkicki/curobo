# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only capture of *pre-TimeOpt* native geometric candidates.

The original 32-restart failure receipts did not retain paths rejected by
TimeOpt. This opt-in, research-only rerun wraps TrajOptSolver._finalize_result,
which receives the result-processor's interpolation JointState immediately
BEFORE invoking native timeopt. It snapshots every lane even on failed
result returns. Post-timeopt cubic controls and validity masks are saved when
available, but are never invented for failures. This is an instrumented
REPLAY, not a modification of the original frozen native failure experiment.

Instrumentation synchronizes GPU and adds overhead; NEVER use its wall
times as normal planning latency. Any disagreement with the frozen native
outcome is recorded as NONREPRODUCIBLE, not silently reconciled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import numpy as np
import torch

from .cohort_retime import ROOT
from .geometry_feasibility_core import sha256_file
from .torque_full_planner import make_planner
from .torque_active import read_profile
from curobo.types import JointState

CASES={
    "trap_cluttered_08":("geometry_trap_source_20261108",
                        "geometry_feasibility_discovery_20261108"),
    "confirm_geom_table_07":("geometry_trap_confirm_20261109",
                             "geometry_feasibility_confirmation_20261109"),
    "confirm_geom_cluttered_02":("geometry_trap_confirm_20261109",
                                "geometry_feasibility_confirmation_20261109"),
}
SCENARIO="j1234_f0300"
OUTPUT=ROOT/"research/joint_time_oracle/results/torque_lp_native_stage_trace_20261108"


def snapshot_joint_state(value,*,keys=("position","knot","knot_dt","velocity",
                                       "acceleration","jerk","dt")):
    out={}
    for name in keys:
        v=getattr(value,name,None)
        if isinstance(v,torch.Tensor):
            out[name]=v.detach().cpu().numpy().copy()
    if "position" not in out:
        raise ValueError("missing pre-TimeOpt joint positions")
    return out


class FinalizeRecorder:
    """Install on only ONE fresh planner instance; always restore the hook."""

    def __init__(self,planner,folder:Path):
        self.solver=planner.trajopt_solver
        self.folder=folder
        self.original=None
        self.events=[]

    def __enter__(self):
        self.folder.mkdir(parents=True,exist_ok=True)
        self.original=self.solver._finalize_result

        def intercepted(*args,**kwargs):
            event={"finalize_index":len(self.events),
                   "pre_status":"UNAVAILABLE","post_status":"NOT_EXECUTED"}
            i=len(self.events)
            try:
                native=self.solver._result_processor.get_interpolation_input()
                arrays=snapshot_joint_state(native)
                filename=self.folder/f"pre_timeopt_{i:03d}.npz"
                np.savez_compressed(filename,**arrays)
                event.update({"pre_status":"RECORDED",
                              "pre_file":filename.name,
                              "pre_sha256":sha256_file(filename),
                              "pre_lanes":int(len(arrays["position"])),
                              "pre_shape":list(arrays["position"].shape)})
            except Exception as exc:
                event["pre_error"]=repr(exc)
            result=None
            try:
                result=self.original(*args,**kwargs)
                event["post_status"]="RETURNED"
                if hasattr(result,"success"):
                    event["selected_success_mask"]=result.success.detach().cpu().numpy().tolist()
                if isinstance(getattr(result,"debug_info",None),dict):
                    masks=result.debug_info.get("validation",{})
                    event["selected_validation_masks"]={
                        name:val.detach().cpu().numpy().tolist()
                        for name,val in masks.items() if isinstance(val,torch.Tensor)}
                return result
            except BaseException as exc:
                event["post_status"]="EXCEPTION"
                event["post_error"]=repr(exc)
                raise
            finally:
                # The native TimeOpt instance may exist and contain valid
                # geometric controls even when all final success flags fail.
                try:
                    solver_timeopt=getattr(self.solver,"_timeopt",None)
                    opt=getattr(solver_timeopt,"optimizer",None)
                    if opt is not None:
                        keep={}
                        for name,attr in (
                            ("controls","_controls"),("start_boundary","_start_boundary"),
                            ("goal","_goal"),("timing","_timing"),
                            ("integrated_times","_times"),("success","_success"),
                            ("stationary","_stationary")):
                            value=getattr(opt,attr,None)
                            if isinstance(value,torch.Tensor):
                                keep[name]=value.detach().cpu().numpy().copy()
                        if all(key in keep for key in
                               ("controls","start_boundary","goal")):
                            filename=self.folder/f"post_timeopt_{i:03d}.npz"
                            np.savez_compressed(filename,**keep)
                            event.update({"post_path_buffers":filename.name,
                                          "post_sha256":sha256_file(filename),
                                          "post_geometry_lanes":int(len(keep["controls"]))})
                    masks=getattr(solver_timeopt,"validation_masks",None)
                    if isinstance(masks,dict):
                        event["native_timeopt_validation_masks"]={
                            k:v.detach().cpu().numpy().tolist() for k,v in masks.items()
                            if isinstance(v,torch.Tensor)}
                except Exception as exc:
                    event["post_capture_error"]=repr(exc)
                self.events.append(event)

        self.solver._finalize_result=intercepted
        return self

    def __exit__(self,*exc):
        self.solver._finalize_result=self.original
        (self.folder/"stage_events.json").write_text(
            json.dumps(self.events,indent=2,default=str)+"\n")


def run_case(case:str,restarts:list[int],*,output:Path=OUTPUT)->dict:
    if case not in CASES:
        raise ValueError("only frozen qualified cases can be traced")
    source_name,discovery_name=CASES[case]
    source=ROOT/"research/joint_time_oracle/results"/source_name
    experiment=ROOT/"research/joint_time_oracle/results"/discovery_name
    tasks=json.loads((source/"tasks.json").read_text())
    task=next(t for t in tasks if t["case_id"]==case)
    profile=read_profile(experiment/SCENARIO/case/"torque_profile.json")
    frozen=experiment/SCENARIO/"native_full"/case/"native_candidates.json"
    if not frozen.exists():
        raise FileNotFoundError("frozen full-native restart results unavailable")
    reference=json.loads(frozen.read_text())
    out=output/case
    out.mkdir(parents=True,exist_ok=True)
    records={}
    for restart in restarts:
        if restart not in range(32):
            raise ValueError("only registered 0..31 restarts are allowed")
        name=f"F{restart:03d}"
        prior=reference.get(name)
        if not prior or prior["status"] not in ("SUCCESS","FAILED"):
            raise ValueError("missing frozen native reference for "+name)
        folder=out/name
        folder.mkdir(exist_ok=True)
        rowfile=folder/"trace_summary.json"
        if rowfile.exists():
            row=json.loads(rowfile.read_text())
            records[name]=row
            continue
        seed=int(task["seed"]+291337*restart)
        torch.manual_seed(seed)
        np.random.seed(seed%(2**32))
        planner=None
        row={"case_id":case,"restart":restart,"seed":seed,
             "frozen_native_status":prior["status"],
             "frozen_profile_sha256":profile["profile_sha256"],
             "frozen_original_record_sha256":sha256_file(frozen),
             "stage_trace_status":"ERROR",
             "timings_are_NOT_normal_planning_latency":True}
        try:
            planner=make_planner(profile,task["scene"],seed,lanes=4,use_cuda_graph=False)
            planner.warmup(enable_graph=False,num_warmup_iterations=2)
            device=planner.default_joint_state.position.device
            names=[f"panda_joint{i}" for i in range(1,8)]
            start=JointState.from_position(
                torch.as_tensor(task["start"],device=device,dtype=torch.float32)[None],
                joint_names=names)
            goal=JointState.from_position(
                torch.as_tensor(task["goal"],device=device,dtype=torch.float32)[None],
                joint_names=names)
            planner.reset_seed()
            with FinalizeRecorder(planner,folder) as recorder:
                result=planner.plan_cspace(goal,start,max_attempts=5)
            new_status="SUCCESS" if result is not None and bool(result.success.any()) else "FAILED"
            row.update({"rerun_status":new_status,
                        "native_attempts":planner.last_planning_attempts,
                        "stage_events":recorder.events,
                        "pre_timeopt_snapshots":sum(x["pre_status"]=="RECORDED"
                                                   for x in recorder.events),
                        "post_timeopt_path_snapshots":sum("post_path_buffers" in x
                                                         for x in recorder.events),
                        "stage_trace_status":"REPRODUCIBLE" if new_status==prior["status"]
                            else "NONREPRODUCIBLE_NATIVE_STATUS"})
        except Exception as exc:
            row["reason"]=repr(exc)
            row["traceback"]=traceback.format_exc()
        finally:
            if planner is not None:
                planner.destroy()
        rowfile.write_text(json.dumps(row,indent=2,default=str)+"\n")
        records[name]=row
        print(case,name,row["stage_trace_status"],flush=True)
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",action="append",choices=tuple(CASES))
    p.add_argument("--restart",type=int,action="append")
    p.add_argument("--output",type=Path,default=OUTPUT)
    args=p.parse_args()
    if args.restart is None:
        restarts=[0]
    else:
        restarts=sorted(set(args.restart))
    if not restarts:
        raise ValueError("empty trace restart inventory")
    outcomes={}
    for name in args.case or list(CASES):
        outcomes[name]=run_case(name,restarts,output=args.output)
    print(json.dumps({"cases":len(outcomes),
                      "restarts":len(restarts)},indent=2))


if __name__=="__main__":
    main()
