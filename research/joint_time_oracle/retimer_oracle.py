# SPDX-License-Identifier: Apache-2.0
"""Versioned fail-closed research bridge for TOPP-RA+Newton and native TimeOpt.

The K107 implementation lives in a DIFFERENT pinned repository. Do not
silently substitute IPOPT, a local LP, or fabricated timing when K107 is
unavailable. A plugin adapter invoked with --request/--response must
produce the exact versioned JSON response, including model and payload
contract SHA, physical constraint families and a timing law. A reported
solution is PRELIMINARY until independent payload-aligned physical and
collision verification. Numerical failures remain distinct from
proven path infeasibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FAMILIES=("position","velocity","acceleration","jerk","torque")
RESPONSE_STATUS=("FEASIBLE_PRELIMINARY","INFEASIBLE_CERTIFIED",
                 "INFEASIBLE_HEURISTIC","NUMERICAL_FAILURE",
                 "UNSUPPORTED_MODEL","TIMEOUT","ERROR")
VERSION="payload-timeopt-bridge-v1"


def digest(obj)->str:
    return hashlib.sha256(json.dumps(obj,sort_keys=True,
      separators=(",",":"),allow_nan=False).encode()).hexdigest()


def json_atomic(path:Path,data:dict):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text())!=data:
            raise ValueError("existing immutable oracle artifact changed")
        return
    temp=path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    temp.replace(path)


@dataclass(frozen=True)
class Oracle:
    name:str
    argv:tuple[str,...]
    engine_commit:str
    timeout_s:float=120.

    def validate(self):
        if self.name not in ("native_timeopt","toppra_newton_k107"):
            raise ValueError("unregistered retimer engine")
        if not self.argv or not self.engine_commit or self.timeout_s<=0:
            raise ValueError("unregistered engine command/revision")
        if any(not isinstance(a,str) or not a for a in self.argv):
            raise ValueError("argv must be nonempty literal tokens")
        return self


def make_request(case_id:str,controls:np.ndarray,*,source_sha:str,
                 model_contract:dict,limit_contract:dict,engine:Oracle)->dict:
    engine.validate()
    x=np.asarray(controls,dtype=np.float64)
    if x.shape!=(22,7) or not np.isfinite(x).all():
        raise ValueError("expected 22-by-7 finite quintic controls")
    if model_contract.get("nominal_effort_scale")!=1.:
        raise ValueError("do not use joint effort derating in nominal payload cohort")
    if not model_contract.get("native_pinocchio_payload_parity_verified"):
        raise ValueError("payload dynamics parity must be independently verified")
    if not model_contract.get("attached_collision_parity_verified"):
        raise ValueError("payload collision representation must be independently verified")
    if set(limit_contract.get("families",()))!=set(FAMILIES):
        raise ValueError("retimer physical-family contract mismatch")
    if not limit_contract.get("signed_torque_bounds_sha256"):
        raise ValueError("missing actual signed original actuator limits")
    if not limit_contract.get("endpoint_contract"):
        raise ValueError("missing timing endpoint contract")
    path_hash=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    fields={"schema":VERSION,"case_id":case_id,"engine":engine.name,
            "engine_commit":engine.engine_commit,
            "source_task_sha256":source_sha,
            "path_controls_sha256":path_hash,
            "path_type":"quintic_clamped_22",
            "quintic_controls":x.tolist(),
            "model_contract":model_contract,
            "constraint_contract":limit_contract,
            "note":"research-only retimer evaluation; not independently certified"}
    fields["request_sha256"]=digest(fields)
    return fields


def make_native_cubic_request(case_id:str,buffers:dict,*,source_sha:str,
                              model_contract:dict,limit_contract:dict,
                              engine:Oracle)->dict:
    """Exact native cubic export: NO conversion to a different quintic path.

    The source arrays must be the original native finalizer's cubic
    control/start/goal buffers from the SAME lane. The K107 adapter must
    replay native q,qs,qss,qsss (including one-sided knot events) on
    this exact geometry before comparing solver feasibility.
    """
    engine.validate()
    required=("controls","start_boundary","goal")
    if any(k not in buffers for k in required):
        raise ValueError("native cubic buffers missing")
    arrays={k:np.asarray(buffers[k],dtype=np.float64) for k in required}
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("nonfinite native path buffers")
    if arrays["controls"].ndim!=2 or arrays["controls"].shape[1]!=7:
        raise ValueError("expected single-lane native cubic controls")
    if arrays["goal"].shape!=(7,):
        raise ValueError("expected single-lane native goal")
    if model_contract.get("nominal_effort_scale")!=1.0:
        raise ValueError("nominal payload cohort must retain original efforts")
    if not model_contract.get("native_pinocchio_payload_parity_verified") or (
        not model_contract.get("attached_collision_parity_verified")):
        raise ValueError("model parity required")
    if set(limit_contract.get("families",()))!=set(FAMILIES):
        raise ValueError("dynamics family mismatch")
    h=hashlib.sha256()
    for key in required:
        v=np.ascontiguousarray(arrays[key])
        h.update(key.encode());h.update(v.dtype.str.encode());h.update(v.tobytes())
    result={"schema":VERSION,"case_id":case_id,"engine":engine.name,
            "engine_commit":engine.engine_commit,
            "source_task_sha256":source_sha,
            "path_type":"native_cubic_exact",
            "path_controls_sha256":h.hexdigest(),
            "native_cubic_buffers":{k:v.tolist() for k,v in arrays.items()},
            "model_contract":model_contract,
            "constraint_contract":limit_contract,
            "note":"exact native cubic replay required; no approximate re-fitting"}
    result["request_sha256"]=digest(result)
    return result


def validate_response(request:dict,response:dict)->dict:
    if response.get("schema")!=VERSION or (
        response.get("request_sha256")!=request["request_sha256"]):
        raise ValueError("oracle response does not match exact input request")
    if response.get("engine")!=request["engine"] or (
        response.get("engine_commit")!=request["engine_commit"]):
        raise ValueError("retimer engine revision mismatch")
    if response.get("path_controls_sha256")!=request["path_controls_sha256"]:
        raise ValueError("retimer silently changed geometric path")
    if response.get("path_type")!=request.get("path_type"):
        raise ValueError("retimer response changed exact path representation")
    if response.get("model_contract_sha256")!=digest(request["model_contract"]):
        raise ValueError("retimer used mismatched payload model")
    if response.get("constraint_contract_sha256")!=digest(request["constraint_contract"]):
        raise ValueError("retimer used mismatched physical limits")
    status=response.get("status")
    if status not in RESPONSE_STATUS:
        raise ValueError("unknown/missing retimer outcome")
    result=dict(response)
    if status=="FEASIBLE_PRELIMINARY":
        t=float(response.get("duration_s",float("nan")))
        if not (np.isfinite(t) and t>0):
            raise ValueError("invalid purportedly feasible timing")
        samples=response.get("timing_law")
        if not isinstance(samples,dict) or "progress" not in samples or "time" not in samples:
            raise ValueError("missing exact timed solution reconstruction")
        progress=np.asarray(samples["progress"],dtype=float)
        stamps=np.asarray(samples["time"],dtype=float)
        if (len(progress)<129 or progress.shape!=stamps.shape or
            not all(np.isfinite(a).all() for a in (progress,stamps)) or
            np.any(np.diff(progress)<=0) or np.any(np.diff(stamps)<=0) or
            abs(progress[0])>1e-7 or abs(progress[-1]-1)>1e-7):
            raise ValueError("invalid retimer progress/time samples")
        if abs(stamps[-1]-t)>1e-5 or abs(stamps[0])>1e-6:
            raise ValueError("duration and timing-sample interval differ")
        if not response.get("family_screen"):
            raise ValueError("no physical-family screen values")
    elif status.startswith("INFEASIBLE"):
        if status=="INFEASIBLE_CERTIFIED" and not response.get("certificate"):
            raise ValueError("do not claim infeasibility without a certificate")
    if "bottleneck_s" in response and response["bottleneck_s"] is not None:
        s=float(response["bottleneck_s"])
        if not np.isfinite(s) or not 0<=s<=1:
            raise ValueError("invalid bottleneck progress")
    return result


def run_oracle(request:dict,engine:Oracle,out:Path)->dict:
    engine.validate()
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rq=out/"request.json";rs=out/"response.json"
    json_atomic(rq,request)
    if rs.exists():
        return validate_response(request,json.loads(rs.read_text()))
    cmd=[*engine.argv,"--request",str(rq),"--response",str(rs)]
    start=time.perf_counter()
    try:
        work=subprocess.run(cmd,capture_output=True,text=True,
                            timeout=engine.timeout_s,check=False)
    except subprocess.TimeoutExpired:
        result={"status":"TIMEOUT","elapsed_s":time.perf_counter()-start,
                "note":"not evidence of path infeasibility"}
        json_atomic(out/"invocation.json",result)
        return result
    elapsed=time.perf_counter()-start
    invocation={"elapsed_s":elapsed,"exit_code":work.returncode,
                "stdout_tail":work.stdout[-4096:],"stderr_tail":work.stderr[-4096:],
                "argv":cmd,"engine_commit":engine.engine_commit}
    json_atomic(out/"invocation.json",invocation)
    if work.returncode!=0 or not rs.exists():
        return {"status":"ERROR","reason":"subprocess error or response absent",
                "elapsed_s":elapsed}
    result=validate_response(request,json.loads(rs.read_text()))
    result["wall_s"]=elapsed
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request",required=True,type=Path)
    p.add_argument("--response",required=True,type=Path)
    p.add_argument("--engine",required=True,choices=("native_timeopt","toppra_newton_k107"))
    p.add_argument("--engine-commit",required=True)
    p.add_argument("--command",nargs="+",required=True)
    a=p.parse_args()
    request=json.loads(a.request.read_text())
    engine=Oracle(a.engine,tuple(a.command),a.engine_commit)
    response=run_oracle(request,engine,a.response.parent)
    print(json.dumps({"status":response["status"]},indent=2))


if __name__=="__main__":
    main()
