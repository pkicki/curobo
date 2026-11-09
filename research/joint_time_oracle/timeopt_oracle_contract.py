# SPDX-License-Identifier: Apache-2.0
"""Fail-closed external retiming interface for K107 TOPP-RA+Newton.

Do not silently substitute TOPP-RA, native cuRobo TimeOpt, or IPOPT
for a requested K107 run. A worker must declare the ACTUAL algorithm,
exact quintic geometric path, payload/dynamics hash, unchanged nominal
effort limits, requested endpoint contract, and active constraint families.

This is a real subprocess transport and shared evaluation contract.
The external K107 worker must be implemented and parity-qualified in
the installed time_optimal_curobo environment before scored experiments.
Absent worker, unsupported endpoint or constraint, or mismatched
dynamics returns ERROR/UNSUPPORTED, NEVER "infeasible".
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from .cohort_geometry import QuinticGeometry
from .timeopt_payload_model import Payload, digest

SCHEMA=1
CONSTRAINT_FAMILIES=("velocity","acceleration","jerk","torque")
ENDPOINT_CONTRACT="zero_velocity_free_acceleration"
K107_ROUTE="TOPP_RA_NEWTON_K107_16"
MAX_WORKER_SECONDS=90.0


def path_record(controls:np.ndarray)->dict:
    path=QuinticGeometry(np.asarray(controls,dtype=np.float64))
    c=np.asarray(controls,dtype=np.float64)
    if c.shape!=(22,7) or not np.isfinite(c).all():
        raise ValueError("K107 path contract requires 22 finite quintic controls x seven joints")
    state=path.evaluate(np.array([0.,1.]))
    if not np.all(np.isfinite(np.asarray(state))):
        raise ValueError("path boundary derivatives are nonfinite")
    row={"basis":"clamped_uniform_bspline","degree":5,
         "knots":path.spline.t.tolist(),"controls":c.tolist(),
         "path_progress_range":[0.,1.],
         "endpoint_q":[state[0][0].tolist(),state[0][-1].tolist()],
         "endpoint_qs":[state[1][0].tolist(),state[1][-1].tolist()]}
    row["path_sha256"]=digest(row)
    return row


def request(case:dict,controls:np.ndarray,*,request_id:str,retimer:str=K107_ROUTE,
            limit_contract:dict|None=None)->dict:
    if retimer not in (K107_ROUTE,"NATIVE_CUROBO_TIMEOPT","REFERENCE_IPOPT"):
        raise ValueError("unknown retimer")
    payload=Payload(**{k:case["payload"][k] for k in (
        "mass_kg","radius_m","native_link","pin_frame")})
    if payload.as_record()["payload_sha256"]!=case["payload"]["payload_sha256"]:
        raise ValueError("payload source SHA mismatch")
    if any(x!=1. for x in case.get("actuator_scale_by_joint",[])):
        raise ValueError("this study forbids synthetic actuator derating")
    path=path_record(controls)
    expected=np.stack((case["original_effort_lower"],
                       case["original_effort_upper"]))
    if expected.shape!=(2,7) or not np.all(expected[0]<0) or not np.all(expected[1]>0):
        raise ValueError("invalid signed nominal effort limits")
    if limit_contract is None:
        raise ValueError("measured original velocity/acceleration/jerk bounds must be supplied")
    contract=dict(limit_contract)
    if set(contract)!=set(CONSTRAINT_FAMILIES):
        raise ValueError("exact four-family contract required; torque-rate excluded explicitly")
    if not np.array_equal(np.asarray(contract["torque"]),expected):
        raise ValueError("worker torque limits differ from native original actuator effort")
    for key in CONSTRAINT_FAMILIES:
        a=np.asarray(contract[key],dtype=float)
        if a.shape!=(2,7) or not np.isfinite(a).all() or not np.all(a[0]<a[1]):
            raise ValueError("invalid "+key+" constraints")
    obj={"schema_version":SCHEMA,"request_id":request_id,
         "task_id":case["case_id"],"task_sha256":case["task_sha256"],
         "path":path,"retimer_requested":retimer,
         "payload":case["payload"],"constraints":contract,
         "endpoint_contract":ENDPOINT_CONTRACT,
         "torque_rate_constraint":"DISABLED_IN_COMMON_FOUR_FAMILY_COMPARISON",
         "required_model_parity":"native-vs-independent-RNEA and collision checked",
         "instructions":"return exact status, duration, stage timing and constraint utilization; no inferred feasibility"}
    obj["request_sha256"]=digest(obj)
    return obj


def check_response(req:dict,res:dict)->dict:
    if not isinstance(res,dict):
        raise ValueError("K107 worker response must be a JSON object")
    for field,source in (("request_sha256","request_sha256"),
                         ("path_sha256","path_sha256"),
                         ("task_sha256","task_sha256"),
                         ("payload_sha256","payload_sha256")):
        expected=(req["path"][source] if field=="path_sha256" else
                  req["payload"][source] if field=="payload_sha256" else req[source])
        if res.get(field)!=expected:
            raise ValueError(field+" mismatch in external retimer response")
    if res.get("retimer_actual")!=req["retimer_requested"]:
        raise ValueError("worker silently substituted an alternate retimer")
    if res.get("endpoint_contract")!=req["endpoint_contract"]:
        raise ValueError("endpoint contract differs")
    if res.get("active_families")!=list(CONSTRAINT_FAMILIES):
        raise ValueError("constraint-family mismatch")
    if res.get("torque_rate_constraint")!=req["torque_rate_constraint"]:
        raise ValueError("torque rate contract differs")
    status=res.get("status")
    if status not in ("FEASIBLE_SCREENED","INFEASIBLE_SCREENED",
                      "NUMERICAL_FAILURE","UNSUPPORTED","ERROR"):
        raise ValueError("unknown worker status")
    for key in ("worker_wall_ms","model_setup_ms","retiming_ms","audit_ms"):
        v=res.get(key)
        if v is None or not np.isfinite(float(v)) or float(v)<0:
            raise ValueError("invalid worker timing "+key)
    if status=="FEASIBLE_SCREENED":
        duration=res.get("duration_s")
        if duration is None or not np.isfinite(float(duration)) or duration<=0:
            raise ValueError("feasible worker returned no finite duration")
        if res.get("independent_scaled95_qualified") is True:
            raise ValueError("online retimer cannot self-certify independent feasibility")
    if status=="INFEASIBLE_SCREENED" and res.get("proof_of_path_infeasibility") is True:
        if not isinstance(res.get("independent_lp_dual_receipt"),dict):
            raise ValueError("infeasibility proof needs explicit independently checked LP")
    if res.get("diagnostics") is not None:
        d=res["diagnostics"]
        if not isinstance(d,dict):
            raise ValueError("diagnostics must be an object")
        if "bottleneck_s" in d and not (0<=float(d["bottleneck_s"])<=1):
            raise ValueError("bottleneck progress outside path")
    return res


def run_worker(req:dict,command:str,output_dir:Path,*,timeout_s:float=MAX_WORKER_SECONDS)->dict:
    if not command or timeout_s<=0:
        raise ValueError("explicit real K107 adapter command is mandatory")
    output_dir.mkdir(parents=True,exist_ok=True)
    path=output_dir/(req["request_id"]+".request.json")
    response=output_dir/(req["request_id"]+".response.json")
    if path.exists() or response.exists():
        raise ValueError("per-request evidence already exists; use a unique ID")
    path.write_text(json.dumps(req,indent=2,allow_nan=False)+"\n")
    started=time.perf_counter()
    try:
        argv=shlex.split(command)+["--request",str(path),"--response",str(response)]
        completed=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                 text=True,timeout=timeout_s,check=False)
        subprocess_wall_ms=(time.perf_counter()-started)*1000.
        if completed.returncode!=0:
            return {"status":"ERROR","reason":"K107_WORKER_EXIT_NONZERO",
                    "returncode":completed.returncode,
                    "stderr_tail":completed.stderr[-2000:],
                    "subprocess_wall_ms":subprocess_wall_ms,
                    "request_sha256":req["request_sha256"]}
        if not response.exists():
            return {"status":"ERROR","reason":"WORKER_DID_NOT_WRITE_RESPONSE",
                    "subprocess_wall_ms":subprocess_wall_ms,
                    "request_sha256":req["request_sha256"]}
        document=json.loads(response.read_text())
        result=check_response(req,document)
        return {**result,"subprocess_wall_ms":subprocess_wall_ms,
                "raw_response_sha256":hashlib.sha256(response.read_bytes()).hexdigest()}
    except subprocess.TimeoutExpired:
        return {"status":"ERROR","reason":"WORKER_TIMEOUT",
                "subprocess_wall_ms":(time.perf_counter()-started)*1000.,
                "request_sha256":req["request_sha256"]}
    except (ValueError,KeyError,json.JSONDecodeError) as exc:
        return {"status":"ERROR","reason":"WORKER_RESPONSE_CONTRACT_ERROR",
                "details":repr(exc),
                "subprocess_wall_ms":(time.perf_counter()-started)*1000.,
                "request_sha256":req["request_sha256"]}
