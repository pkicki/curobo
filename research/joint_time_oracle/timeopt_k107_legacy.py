# SPDX-License-Identifier: Apache-2.0
"""Legacy synthetic-torque fixed-path K107 diagnostic, NOT payload benchmark.

Compare genuine TOPP-RA+Newton on the three independently qualified
quintic W alternative geometries from the original 30% torque-derating
study. The original effort profiles are preserved EXACTLY in this
separate legacy diagnostic and never enter the nominal-payload source.

A native cubic fixed path MUST use the exact cuRobo cardinal spline
basis adapter, not a quintic waypoint fit. Native-original diagnostic
is therefore explicitly UNSUPPORTED until a cardinal exact-path
handoff is independently qualified. An original path with a certified
positive relaxed necessary-torque LP cannot be made feasible by any
retimer, including K107. Testing it as if a failure were a solver bug
would be scientifically invalid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cohort_retime import ROOT
from .timeopt_oracle_contract import (
    CONSTRAINT_FAMILIES,ENDPOINT_CONTRACT,K107_ROUTE,
    path_record,
)
from .timeopt_payload_model import Payload,digest,file_sha256
from .timeopt_k107_service import K107Service
from .torque_active import read_profile

SOURCE=ROOT/"research/joint_time_oracle/results/torque_global_lp_screening_20261109"
OUTPUT=ROOT/"research/joint_time_oracle/results/legacy_k107_toppra_newton_20261109"
CASES=("trap_cluttered_08","confirm_geom_table_07","confirm_geom_cluttered_02")


def legacy_request(case:str,source:Path)->dict:
    if case not in CASES:
        raise ValueError("unregistered historical synthetic-torque case")
    folder=source/"no_lp"/case
    geo=json.loads((folder/"w_geometry.json").read_text())
    controls=np.asarray(geo["controls"],dtype=float)
    profile=read_profile(folder/"torque_profile.json")
    if not (profile.get("regime")=="j1234_f0300" or
            profile.get("scenario_id")=="j1234_f0300" or
            profile.get("scenario")=="j1234_f0300"):
        raise ValueError("historical torque scenario not the registered 30% stress")
    file=ROOT/"research/joint_time_oracle/results/native_reconstruction_20261106/baseline_validation_states.npz"
    with np.load(file) as f:
        limits={k:np.asarray(f[k+"_bounds"]).tolist() for k in
                ("velocity","acceleration","jerk")}
    limits["torque"]=np.stack((profile["effort_lower"],
                              profile["effort_upper"])).tolist()
    payload=Payload(0.0).as_record()
    if geo["task_sha256"]!=profile["source_task_sha256"]:
        raise ValueError("legacy W task/profile provenance mismatch")
    path=path_record(controls)
    row={
        "schema_version":1,"request_id":case+"_legacy_alt_W",
        "task_id":case,"task_sha256":geo["task_sha256"],
        "path":path,"retimer_requested":K107_ROUTE,
        "payload":payload,"constraints":limits,
        "endpoint_contract":ENDPOINT_CONTRACT,
        "torque_rate_constraint":"DISABLED_IN_COMMON_FOUR_FAMILY_COMPARISON",
        "study_kind":"LEGACY_SYNTHETIC_TORQUE_30PCT_DIAGNOSTIC",
        "source_w_geometry_sha256":file_sha256(folder/"w_geometry.json"),
        "source_torque_profile_sha256":file_sha256(folder/"torque_profile.json"),
        "independent_source_W_qualification_required":True,
        "active_families":list(CONSTRAINT_FAMILIES),
    }
    row["request_sha256"]=digest(row)
    return row


def run(source:Path,output:Path,command:str)->dict:
    if not command:raise ValueError("genuine K107 worker command required")
    output.mkdir(parents=True,exist_ok=True)
    records={}
    with K107Service(command,output/"oracle") as worker:
        for case in CASES:
            req=legacy_request(case,source)
            result=worker.evaluate(req)
            records[case]={
                "status":result["status"],
                "duration_s":result.get("duration_s"),
                "request_sha256":req["request_sha256"],
                "source_w_geometry_sha256":req["source_w_geometry_sha256"],
                "original_native_cubic_status":
                    "NOT_RUN_EXACT_CUBIC_K107_ADAPTER_PENDING",
                "fully_independent_scaled95_qualification":False,
                "note":"worker online screen is NOT independent validation"}
    target=output/"legacy_k107_summary.json"
    if target.exists() and json.loads(target.read_text())!=records:
        raise ValueError("legacy K107 diagnostic previously frozen")
    if not target.exists():target.write_text(json.dumps(records,indent=2,allow_nan=False)+"\n")
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=SOURCE)
    p.add_argument("--output",type=Path,default=OUTPUT)
    p.add_argument("--worker-command",required=True)
    a=p.parse_args()
    print(json.dumps(run(a.source,a.output,a.worker_command),indent=2))


if __name__=="__main__":
    main()
