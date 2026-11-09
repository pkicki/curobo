# SPDX-License-Identifier: Apache-2.0
"""Freeze three independently qualified W and original IPOPT P sources.

Existing no_lp artifacts from the completed global exploration are READ
ONLY. No control fitting, geometry editing, or selection by new result.
Require frozen W physical+Scaled95 receipt, exact source profile hashes,
IPOPT P controls, source P states, and source P solver status. Preserve
all input SHA and a complete three-task inventory before new retiming.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cohort_retime import ROOT
from .torque_active import read_profile
from .torque_margin_core import CASE_IDS, DEFAULT_SOURCE_ARM, immutable_json, sha256_file

PREVIOUS=ROOT/"research/joint_time_oracle/results/torque_global_lp_screening_20261109"
OUTPUT=ROOT/"research/joint_time_oracle/results/torque_margin_retime_20261109"
INPUTS=("w_geometry.json","w_dense.npz","p_dense.npz",
        "p_refined_dense.npz","p_timing_controls.npz","ipopt_polish.json",
        "torque_profile.json")
MODEL_INPUTS=(
    "curobo/content/assets/robot/franka_description/franka_panda.urdf",
    "curobo/content/configs/robot/franka.yml",
    "research/joint_time_oracle/results/native_reconstruction_20261106/baseline_validation_states.npz",
)


def prepare(source:Path,output:Path,*,source_arm:str=DEFAULT_SOURCE_ARM)->dict:
    arm=source/source_arm
    if source_arm!="no_lp":
        raise ValueError("registered source is independent no_lp arm only")
    all_tasks=json.loads((arm/"tasks.json").read_text())
    by_id={t["case_id"]:t for t in all_tasks}
    receipts=json.loads((arm/"w_adaptive_receipts.json").read_text())
    cases=[]
    copied=[]
    for case in CASE_IDS:
        if case not in by_id:
            raise ValueError("missing source qualified case: "+case)
        task=by_id[case]
        folder=arm/case
        files={}
        for name in INPUTS:
            path=folder/name
            if not path.is_file():
                raise FileNotFoundError("required source evidence missing: "+str(path))
            files[name]=sha256_file(path)
        receipt=receipts.get(case+":W")
        if not isinstance(receipt,dict) or receipt.get("physical") is not True or (
            receipt.get("scaled95") is not True):
            raise ValueError("source W lacks independent qualified Scaled95 receipt")
        if receipt.get("base_sha256")!=files["w_dense.npz"]:
            raise ValueError("W receipt source hash mismatch")
        profile=read_profile(folder/"torque_profile.json")
        if profile["source_task_sha256"]!=task["task_sha256"]:
            raise ValueError("task/profile mismatch")
        polish=json.loads((folder/"ipopt_polish.json").read_text())
        if not polish.get("solver",{}).get("solver_converged"):
            raise ValueError("source P IPOPT did not converge: "+case)
        if polish.get("p_timing_controls_sha256")!=files["p_timing_controls.npz"]:
            raise ValueError("P spline controls SHA mismatch")
        if polish.get("p_refined_sha256")!=files["p_refined_dense.npz"]:
            raise ValueError("source P refined trajectory SHA mismatch")
        if polish.get("p_sha256")!=files["p_dense.npz"]:
            raise ValueError("source P dense SHA mismatch")
        if polish.get("source_w_sha256")!=files["w_dense.npz"]:
            raise ValueError("source P not derived from qualified W")
        cases.append({"case_id":case,"task_sha256":task["task_sha256"],
                      "profile_sha256":profile["profile_sha256"],
                      "source_files_sha256":files,
                      "source_W_duration_s":polish["source_W_duration_s"],
                      "source_P_duration_s":polish["preliminary_P_duration_s"],
                      "source_P_scaled95_qualified":False,
                      "source_W_receipt":receipt})
        dst=output/case
        dst.mkdir(parents=True,exist_ok=True)
        dest=dst/"torque_profile.json"
        if dest.exists() and dest.read_bytes()!=(folder/"torque_profile.json").read_bytes():
            raise ValueError("frozen output torque profile changed")
        if not dest.exists():
            dest.write_bytes((folder/"torque_profile.json").read_bytes())
        copied.append(task)
    model_shas={name:sha256_file(ROOT/name) for name in MODEL_INPUTS}
    manifest={"schema_version":1,"source_arm":source_arm,
              "source_root":str(source.resolve()),"sources":cases,
              "model_inputs_sha256":model_shas,
              "source_tasks_sha256":sha256_file(arm/"tasks.json"),
              "source_w_receipts_sha256":sha256_file(arm/"w_adaptive_receipts.json"),
              "scope":"three preselected synthetic derating tasks; source W qualified; source P NOT Scaled95 qualified"}
    immutable_json(output/"margin_registration.json",manifest)
    immutable_json(output/"source_tasks.json",copied)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=PREVIOUS)
    p.add_argument("--output",type=Path,default=OUTPUT)
    a=p.parse_args()
    out=prepare(a.source,a.output)
    print(json.dumps({"cases":len(out["sources"]),"registration":
                      str(a.output/"margin_registration.json")},indent=2))


if __name__=="__main__":
    main()
