# SPDX-License-Identifier: Apache-2.0
"""Staged native cuRoboV2-Fast challenge on ORIGINAL effort limits + payload.

Native plans under an independently specified rigid payload are NOT
automatically independently qualified. A separate payload RNEA+collision
parity test and physical/Scaled95 audit MUST pass before declaring
a native success or failure for a physics-based scientific comparison.

Fresh planners are built per restart; setup/warmup excluded from
online plan wall time but reported. This is a research discovery harness,
not a production latency benchmark. Unknown runtime errors are never
called planning failures. No synthetic joint-effort derating.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .timeopt_payload_model import Payload, from_record, file_sha256, native_robot
from .timeopt_payload_tasks import OUT

RESTART_LEVELS = (1, 4, 16, 32)


def _safe_write(path:Path,data:dict)->None:
    payload=json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if path.exists():
        previous=json.loads(path.read_text())
        # Appending new restart rows is supported, changing old rows is not.
        if previous==data:return
        if not (path.name=="native_candidates.json" and isinstance(previous,dict)
                and all(data.get(k)==v for k,v in previous.items())):
            raise ValueError("previous frozen native evidence changed")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(payload)


def _make_planner(task:dict, seed:int,lanes:int):
    import torch

    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

    payload=from_record(task["payload"])
    robot=native_robot(payload)
    cfg=MotionPlannerCfg.create(robot=robot,scene_model=task["scene"],
        graph_planner_algorithm="rrt",enable_timeopt=True,
        use_cuda_graph=False,random_seed=seed,num_trajopt_seeds=lanes,
        max_batch_size=1)
    planner=MotionPlanner(cfg)
    expected=np.stack((task["original_effort_lower"],task["original_effort_upper"]))
    actual=(planner.trajopt_solver.kinematics.config.kinematics_config.joint_limits.effort
            .detach().cpu().numpy())
    if not np.allclose(expected,actual,atol=1e-6,rtol=1e-6):
        planner.destroy()
        raise RuntimeError("native planner original effort limits changed")
    return planner


def execute(root:Path,*,restarts:int,lanes:int=4,max_attempts:int=5,
            warmup:int=2,case_filter:str|None=None,mass_filter:float|None=None,
            max_tasks:int|None=None,parity_receipt:Path|None=None)->dict:
    if restarts not in RESTART_LEVELS or lanes<1 or max_attempts<1:
        raise ValueError("unregistered native challenge budget")
    tasks=json.loads((root/"tasks.json").read_text())
    selected=[x for x in tasks if x["allowed_for_native_challenge"]
              and (case_filter is None or x["case_id"]==case_filter)
              and (mass_filter is None or x["payload"]["mass_kg"]==mass_filter)]
    if max_tasks is not None:
        if max_tasks<1:raise ValueError("invalid task cap")
        selected=selected[:max_tasks]
    if parity_receipt is None:
        raise ValueError("native payload parity receipt required before scored planning")
    checked=json.loads(parity_receipt.read_text())
    required_masses={x["payload"]["payload_sha256"] for x in selected}
    if not checked.get("model_and_collision_parity_pass") or not required_masses<=set(
        checked.get("payload_sha256_qualified",[])
    ):
        raise ValueError("not all payload models qualified by native/independent parity")
    dirname=root/"native_full"
    dirname.mkdir(parents=True,exist_ok=True)
    config={"schema_version":1,"restarts":restarts,"lanes":lanes,
            "max_attempts":max_attempts,"warmup":warmup,
            "all_source_sha256":file_sha256(root/"tasks.json"),
            "source_parity_sha256":file_sha256(parity_receipt),
            "original_effort_unchanged":True,
            "notes":"setup/warmup untimed; native pass/failure needs independent audit"}
    # Multiple restart budgets update common prefix; preserve already
    # scored restart rows, keep each arm's selection manifest separate.
    selection=dirname/("selection_r"+str(restarts)+".json")
    _safe_write(selection,{**config,"case_ids":[x["case_id"] for x in selected]})
    summary={}
    for task in selected:
        import torch
        from curobo.types import JointState

        folder=dirname/task["case_id"]
        folder.mkdir(parents=True,exist_ok=True)
        inventory=folder/"native_candidates.json"
        rows=json.loads(inventory.read_text()) if inventory.exists() else {}
        for restart in range(restarts):
            tag=f"F{restart:03d}"
            if tag in rows and rows[tag]["status"] in ("SUCCESS_NATIVE_UNQUALIFIED",
                                                       "FAIL_NATIVE_BUDGET"):
                continue
            seed=int(task["seed"]+291337*restart)
            record={"tag":tag,"seed":seed,"restart":restart,
                    "status":"ERROR","task_sha256":task["task_sha256"],
                    "payload_sha256":task["payload"]["payload_sha256"]}
            planner=None
            try:
                torch.manual_seed(seed)
                np.random.seed(seed%(2**32))
                stamp=time.perf_counter()
                planner=_make_planner(task,seed,lanes)
                planner.warmup(enable_graph=False,num_warmup_iterations=warmup)
                torch.cuda.synchronize()
                record["setup_warmup_wall_s"]=time.perf_counter()-stamp
                names=[f"panda_joint{i}" for i in range(1,8)]
                device=planner.default_joint_state.position.device
                start=JointState.from_position(
                    torch.as_tensor(task["start"],device=device,dtype=torch.float32)[None],
                    joint_names=names)
                goal=JointState.from_position(
                    torch.as_tensor(task["goal"],device=device,dtype=torch.float32)[None],
                    joint_names=names)
                planner.reset_seed()
                torch.cuda.synchronize()
                begin=time.perf_counter()
                result=planner.plan_cspace(goal,start,max_attempts=max_attempts)
                torch.cuda.synchronize()
                record["plan_wall_s"]=time.perf_counter()-begin
                record["internal_planning_attempts"]=planner.last_planning_attempts
                if result is None or not bool(result.success.any()):
                    record["status"]="FAIL_NATIVE_BUDGET"
                else:
                    # Persist exact cubic geometry plus TimeOpt timing,
                    # NOT just samples returned by the public trajectory.
                    # Reuse the proven reconstruction path from the
                    # synthetic study with a NOMINAL effort receipt.
                    from .torque_full_planner import captured_candidate
                    receipt={
                        "effort_lower":task["original_effort_lower"],
                        "effort_upper":task["original_effort_upper"],
                        "profile_sha256":task["payload"]["payload_sha256"],
                    }
                    dense,metadata,buffers=captured_candidate(
                        planner,result,task,receipt)
                    path=folder/(tag.lower()+"_dense.npz")
                    native_path=folder/(tag.lower()+"_buffers.npz")
                    input_path=folder/(tag.lower()+"_native_input.npz")
                    if any(x.exists() for x in (path,native_path,input_path)):
                        raise ValueError("native replay source exists, refusing overwrite")
                    np.savez_compressed(path,**dense)
                    np.savez_compressed(native_path,**buffers["retimer"])
                    np.savez_compressed(input_path,**buffers["native"])
                    record.update(metadata)
                    record.update({"status":"SUCCESS_NATIVE_UNQUALIFIED",
                                   "dense_sha256":file_sha256(path),
                                   "exact_native_cubic_sha256":file_sha256(native_path),
                                   "native_input_sha256":file_sha256(input_path),
                                   "duration_s":float(metadata["duration_s"]),
                                   "independent_payload_qualified":False})
            except Exception as exc:
                record["status"]="ERROR"
                record["reason"]=repr(exc)
                record["traceback"]=traceback.format_exc()
            finally:
                if planner is not None:planner.destroy()
            rows[tag]=record
            _safe_write(inventory,rows)
        values=list(rows.values())
        summary[task["case_id"]]={
            "attempted":len(values),
            "native_success_unqualified":sum(x["status"]=="SUCCESS_NATIVE_UNQUALIFIED" for x in values),
            "budget_failures":sum(x["status"]=="FAIL_NATIVE_BUDGET" for x in values),
            "errors":sum(x["status"]=="ERROR" for x in values),
            "note":"no independent post-validation yet; no real-world safety claim"}
    out=dirname/("native_summary_r"+str(restarts)+".json")
    _safe_write(out,summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUT)
    p.add_argument("--restarts",type=int,required=True,choices=RESTART_LEVELS)
    p.add_argument("--lanes",type=int,default=4)
    p.add_argument("--max-attempts",type=int,default=5)
    p.add_argument("--warmup",type=int,default=2)
    p.add_argument("--case")
    p.add_argument("--mass",type=float)
    p.add_argument("--max-tasks",type=int)
    p.add_argument("--parity-receipt",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(execute(a.root,restarts=a.restarts,lanes=a.lanes,
        max_attempts=a.max_attempts,warmup=a.warmup,case_filter=a.case,
        mass_filter=a.mass,max_tasks=a.max_tasks,
        parity_receipt=a.parity_receipt),indent=2))


if __name__=="__main__":
    main()
