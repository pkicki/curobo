# SPDX-License-Identifier: Apache-2.0
"""Process-isolated exact-path interface to TOPP-RA + K107 Newton.

The TOPP checkout is a SEPARATE repository with a pinned SHA, CUDA/COPP
environment and native dynamics. Never silently fall back to IPOPT, fit
the native cubic geometry, alter signed actuator limits, or claim the
K107 reference solve's process overhead is captured production latency.

A returned K107 timing law is PRELIMINARY until independently audited on
the *same* payload model, original joint effort limits, collision model
and exact geometric spline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from .cohort_geometry import NativeGeometry, QuinticGeometry
from .cohort_retime import ROOT, states
from .payload_timeopt_contract import Payload,canonical_sha,immutable_json,source_sha
from .payload_timeopt_contract import SOURCE_TOPP_SHA

FOREIGN_SCRIPT="scripts/retime_foreign_curobo_path.py"
REQUEST_SCHEMA="foreign_quintic_or_native_cubic_v1"


def geometry_from_npz(path:Path,*,lane:int|None=None)->tuple[dict,Any, np.ndarray]:
    """Return EXACT native cubic or existing 22-control quintic; no fitting."""
    with np.load(path) as src:
        fields=set(src.files)
        if "controls" not in fields:
            raise ValueError("no geometric control archive")
        if {"start_boundary","goal"}.issubset(fields):
            if lane is None:
                raise ValueError("explicit native candidate lane required")
            buffers={k:np.asarray(src[k]) for k in ("controls","start_boundary","goal")}
            if not 0<=lane<len(buffers["controls"]):
                raise ValueError("native lane out of range")
            native=NativeGeometry(buffers,lane)
            data={"kind":"native_uniform_cubic",
                  "start_boundary":buffers["start_boundary"][lane].tolist(),
                  "controls":buffers["controls"][lane].tolist(),
                  "goal":buffers["goal"][lane].tolist()}
            interval=native.nonstationary_interval()
            normalized=np.array([(interval[0]),(interval[1])])
            geometry=native
        else:
            controls=np.asarray(src["controls"],dtype=np.float64)
            if controls.shape!=(22,7):
                raise ValueError("expected frozen 22x7 quintic")
            data={"kind":"quintic_open_uniform","controls":controls.tolist()}
            geometry=QuinticGeometry(controls)
            normalized=np.unique(geometry.spline.t)
    return data,geometry,normalized


def parity_request(*,source_path:Path,lane:int|None,payload:Payload,
                   signed_effort:np.ndarray,case_id:str)->dict:
    signed=np.asarray(signed_effort,dtype=np.float64)
    if signed.shape!=(2,7) or not np.isfinite(signed).all():
        raise ValueError("signed nominal effort shape mismatch")
    if not np.allclose(signed[0],-signed[1],atol=1e-6,rtol=1e-6):
        raise ValueError("K107 symmetric torque interface cannot represent signed asymmetric bounds")
    geometry,_,_=geometry_from_npz(source_path,lane=lane)
    return {"schema":REQUEST_SCHEMA,"case_id":case_id,
            "payload":{"mass_kg":payload.mass_kg,
                       "com_xyz_m":list(payload.com_xyz_m)},
            "payload_collision_qualified":False,
            "effort_bounds_signed":signed.tolist(),
            "path_input_file":str(source_path.resolve()),
            "path_input_sha256":source_sha(source_path),
            "geometry":geometry,"geometry_sha256":canonical_sha(geometry),
            "native_lane":lane,
            "source_curobo_sha":"1597f1292ae45e9d2fe3ed2f76ec15f5126d38e2",
            "solver_constraint_contract":"C3_NO_TORQUE_RATE_SCALED95",
            "endpoint_contract":"ZERO_PATH_SPEED_FREE_ACCELERATION",
            "status":"REQUIRES_EXTERNAL_NATIVE_PINOCCHIO_DYNAMICS_PARITY_AND_INDEPENDENT_AUDIT"}


def solve_foreign(*,source_path:Path,lane:int|None,payload:Payload,effort:np.ndarray,
                  case_id:str,topp_repo:Path,topp_python:Path,output:Path,
                  expected_topp_sha:str)->dict:
    if not expected_topp_sha or len(expected_topp_sha)<12:
        raise ValueError("must pin full external TOPP solver revision")
    top=Path(topp_repo).resolve()
    actual=subprocess.check_output(["git","-C",str(top),"rev-parse","HEAD"],
                                    text=True).strip()
    if actual!=expected_topp_sha:
        raise ValueError("external TOPP repository revision mismatch: "+actual)
    script=top/FOREIGN_SCRIPT
    if not script.is_file():
        raise FileNotFoundError("TOPP foreign-path runner not installed in pinned checkout")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    request=parity_request(source_path=source_path,lane=lane,payload=payload,
                          signed_effort=effort,case_id=case_id)
    frozen=output/"foreign_request.json"
    immutable_json(frozen,request)
    path=output/"foreign_result.json"
    if path.exists():
        result=json.loads(path.read_text())
        if result.get("source_request_sha256")!=source_sha(frozen):
            raise ValueError("stale TOPP result from another request")
        return result
    before=time.perf_counter()
    completed=subprocess.run([str(Path(topp_python).resolve()),str(script),
                              "--request",str(frozen),"--output",str(path)],
                             cwd=top,text=True,capture_output=True,timeout=1800,check=False)
    elapsed=time.perf_counter()-before
    if not path.exists():
        raise RuntimeError(f"TOPP subprocess did not deliver result (exit {completed.returncode}): "+
                           completed.stderr[-1200:])
    result=json.loads(path.read_text())
    if (result.get("source_request_sha256")!=source_sha(frozen) or
        result.get("source_solver_lineage")!=SOURCE_TOPP_SHA):
        raise ValueError("foreign result source/solver lineage mismatch")
    if completed.returncode:
        raise RuntimeError("TOPP external subprocess error: "+completed.stderr[-1200:])
    immutable_json(output/"foreign_invocation.json",{
        "case_id":case_id,"source_path_sha256":source_sha(source_path),
        "external_git_sha":actual,"external_interpreter":str(topp_python),
        "subprocess_roundtrip_wall_s":elapsed,
        "reported_inner_reference_solver_ms":result.get("numerical_elapsed_ms"),
        "not_production_latency":True,
        "status":result.get("status")})
    return result


def reconstruct_topp_result(*,source_path:Path,lane:int|None,result:dict,
                            output:Path)->dict:
    """Reconstruct one candidate from delivered K107 cubic timing controls.

    This is an independent NUMERICAL replay, NOT an independent torque
    or collision feasibility certificate.  For a native cubic, use the
    original native geometry and trimmed knots rather than a waypoint fit.
    """
    if result.get("status")!="CANDIDATE":
        return {"status":"NO_K107_CANDIDATE","reason":result.get("status")}
    controls=np.asarray(result["squared_speed_coefficients"],dtype=float)
    if controls.shape!=(107,) or not np.isfinite(controls).all():
        raise ValueError("wrong K107 output shape")
    _,geometry,knots=geometry_from_npz(source_path,lane=lane)
    if isinstance(geometry,NativeGeometry):
        interval=geometry.nonstationary_interval()
        # all native cubic support interval boundaries mapped to trimmed s.
        k=geometry.spans
        knots=np.unique(np.clip(
            (np.arange(k+1)/k-interval[0])/(interval[1]-interval[0]),0,1))
        evaluator=geometry.evaluate_trimmed
    else:
        evaluator=geometry.evaluate
    timeline=states(evaluator,controls,count=8193,knots=knots)
    output=Path(output)
    if output.exists():
        raise ValueError("refusing to overwrite K107 replay")
    np.savez_compressed(output,**timeline)
    return {"status":"PRELIMINARY_K107_SAMPLED_REPLAY",
            "duration_s":float(timeline["time"][-1]),
            "source_geometry_sha256":source_sha(source_path),
            "K107_output_sha256":canonical_sha(controls.tolist()),
            "replay_sha256":source_sha(output),
            "scope":"K107 producer and numerical states only, independent physical/collision qualification pending"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-path",required=True,type=Path)
    p.add_argument("--native-lane",type=int)
    p.add_argument("--case-id",required=True)
    p.add_argument("--payload-mass",type=float,default=0.)
    p.add_argument("--payload-com",type=float,nargs=3,default=[0,0,0])
    p.add_argument("--effort-bounds-npz",type=Path,required=True)
    p.add_argument("--topp-repo",type=Path,required=True)
    p.add_argument("--topp-python",type=Path,required=True)
    p.add_argument("--topp-revision",required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    with np.load(args.effort_bounds_npz) as f:
        effort=np.asarray(f["effort_bounds"])
    result=solve_foreign(
        source_path=args.source_path,lane=args.native_lane,
        payload=Payload(args.payload_mass,tuple(args.payload_com)),
        effort=effort,case_id=args.case_id,topp_repo=args.topp_repo,
        topp_python=args.topp_python,output=args.output_dir,
        expected_topp_sha=args.topp_revision)
    summary=reconstruct_topp_result(
        source_path=args.source_path,lane=args.native_lane,result=result,
        output=args.output_dir/"k107_dense.npz")
    immutable_json(args.output_dir/"k107_replay.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
