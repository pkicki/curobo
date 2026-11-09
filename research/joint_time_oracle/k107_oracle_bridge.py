# SPDX-License-Identifier: Apache-2.0
"""Cross-repository, fail-closed TOPP-RA+Newton K107 request/response.

This adapter NEVER treats a K107 online screen as an independent
certificate. It binds task, original-effort payload, controls, cuRobo
model files and exact TOPP-RA+Newton script revision; the independent
Pinocchio dense/adaptive audit runs in the *cuRobo* repository.

K107 consumes an exact clamped-uniform quintic path. The previous
qualified W paths are exact; native cubic paths must first be converted
with explicit P/V/A/J replay error. An approximate curve is a DIFFERENT
geometry and cannot silently be counted as rescue of the original path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .cohort_retime import ROOT
from .geometry_feasibility_core import sha256_file
from .payload_parity import Payload, nominal_effort_only
from .torque_active import read_profile

BRIDGE_REL="scripts/joint_time_oracle_k107_bridge.py"
BRIDGE_SHA_AT_IMPLEMENTATION="d968ea27bb5c5facb1316beb1ca21bac064d9175"


def require_repo(repo:Path,expected_sha:str,*,allow_descendant:bool=False):
    repo=repo.resolve()
    actual=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    if not allow_descendant and actual!=expected_sha:
        raise ValueError(f"retimer revision changed: {actual}; expected {expected_sha}")
    if not (repo/BRIDGE_REL).is_file():
        raise FileNotFoundError("K107 bridge script absent")
    return actual


def prepare(out:Path,case:dict,profile_file:Path,controls:np.ndarray,
            *,source_geometry_id:str,source_geometry_sha256:str,
            model_files:tuple[Path,...],retimer_commit:str)->dict:
    profile=read_profile(profile_file)
    if profile["source_task_sha256"]!=case["task_sha256"]:
        raise ValueError("K107 task/profile checksum mismatch")
    if profile.get("payload")!=case.get("payload"):
        raise ValueError("payload task/profile mismatch")
    bounds=np.stack((profile["effort_lower"],profile["effort_upper"]))
    nominal=np.stack((profile["nominal_effort_lower"],profile["nominal_effort_upper"]))
    source_kind=case.get("regime","legacy_synthetic_derate")
    if source_kind=="nominal_effort_actual_payload":
        nominal_effort_only(bounds,nominal)
        payload=Payload(**{k:v for k,v in case["payload"].items() if k!="payload_sha256"})
        if payload.record()!=case["payload"]:
            raise ValueError("unregistered/modified payload")
        payload_record=case["payload"]
    elif source_kind=="legacy_synthetic_derate":
        if case.get("payload") is not None:
            raise ValueError("synthetic derated source has a payload")
        payload_record=Payload(0.).record()
    else:
        raise ValueError("unsupported source type for K107 matching")
    ctrl=np.asarray(controls,dtype=np.float64)
    if ctrl.shape!=(22,7) or not np.isfinite(ctrl).all():
        raise ValueError("unexpected quintic control polygon")
    if not (np.max(abs(ctrl[:3]-ctrl[0]))<1e-10 and
            np.max(abs(ctrl[-3:]-ctrl[-1]))<1e-10):
        raise ValueError("endpoint position/velocity/acceleration contract mismatch")
    out.mkdir(parents=True,exist_ok=True)
    path=out/"geometry.npz"
    if path.exists():
        with np.load(path) as z:
            np.testing.assert_array_equal(z["controls"],ctrl)
    else:
        np.savez_compressed(path,controls=ctrl)
    fingerprint={str(p.resolve()):sha256_file(p) for p in model_files}
    request={"schema_version":1,"source_case_sha256":case["task_sha256"],
             "source_geometry_id":source_geometry_id,
             "source_geometry_sha256":source_geometry_sha256,
             "geometry_file":str(path.resolve()),"geometry_sha256":sha256_file(path),
             "torque_profile_file":str(profile_file.resolve()),
             "torque_profile_sha256":sha256_file(profile_file),
             "payload":payload_record,"source_kind":source_kind,
             "model_fingerprint":fingerprint,
             "retimer_commit_sha256":retimer_commit,
             "endpoint_contract":"joint_rest","constraint_mode":"C3_NO_TR",
             "limit_scale":.95,
             "torque_rate_enforced":False,
             "note":"K107 output is a candidate, not independently validated"}
    f=out/"k107_request.json"
    payloadtext=json.dumps(request,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if f.exists() and f.read_text()!=payloadtext:
        raise ValueError("frozen retimer request differs")
    if not f.exists(): f.write_text(payloadtext)
    return request


def run(out:Path,retimer_repo:Path,retimer_python:str,retimer_sha:str,
        *,iterations:int=16,grid:int=2049)->dict:
    require_repo(retimer_repo,retimer_sha)
    if iterations not in (8,12,16,24) or grid not in (2049,4097):
        raise ValueError("unregistered retimer budget/grid")
    requestfile=out/"k107_request.json"
    request=json.loads(requestfile.read_text())
    if request["retimer_commit_sha256"]!=retimer_sha:
        raise ValueError("K107 source revision differs from request")
    output=out/"k107_response.json"
    if output.exists():
        response=json.loads(output.read_text())
        if response.get("request_sha256")!=sha256_file(requestfile):
            raise ValueError("existing response came from a different request")
        return response
    cmd=[retimer_python,str((retimer_repo/BRIDGE_REL).resolve()),
         "--request",str(requestfile.resolve()),"--output",str(output.resolve()),
         "--iterations",str(iterations),"--grid",str(grid)]
    started=time.perf_counter()
    completed=subprocess.run(cmd,cwd=retimer_repo,check=False,
                             text=True,capture_output=True,timeout=600)
    diagnostic={"schema_version":1,"cmd":cmd,"elapsed_process_wall_s":time.perf_counter()-started,
                "returncode":completed.returncode,
                "stdout_tail":completed.stdout[-2500:],
                "stderr_tail":completed.stderr[-3500:]}
    (out/"k107_process.json").write_text(json.dumps(diagnostic,indent=2,allow_nan=False)+"\n")
    if completed.returncode!=0 or not output.is_file():
        raise RuntimeError("K107 subprocess failed; see k107_process.json; not infeasibility")
    response=json.loads(output.read_text())
    if response.get("request_sha256")!=sha256_file(requestfile) or (
        response.get("source_case_sha256")!=request["source_case_sha256"]):
        raise ValueError("K107 response lacks exact source provenance")
    if response.get("constraint_mode") not in ("C3_NO_TR",None):
        raise ValueError("different K107 constraint contract")
    return response


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request-dir",type=Path,required=True)
    p.add_argument("--retimer-repo",type=Path,required=True)
    p.add_argument("--retimer-python",required=True)
    p.add_argument("--retimer-commit",required=True)
    p.add_argument("--grid",type=int,default=2049)
    p.add_argument("--iterations",type=int,default=16)
    a=p.parse_args()
    print(json.dumps({k:v for k,v in run(a.request_dir,a.retimer_repo,
                                         a.retimer_python,a.retimer_commit,
                                         iterations=a.iterations,grid=a.grid).items()
                      if k not in ("coefficients","knots")},indent=2))


if __name__=="__main__":main()
