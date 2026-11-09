# SPDX-License-Identifier: Apache-2.0
"""Freeze retiming-guided planner after source-disjoint DEVELOPMENT only.

The nominal-payload task generator creates disjoint base endpoint groups
BEFORE solver results. Scored holdout policies must not be tuned against
the previously frozen holdout oracle observations. Freeze source code,
K107 worker version, payload parity, hard-case selection and
completed DEVELOPMENT results of all three arms. Freeze is immutable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cohort_retime import ROOT
from .timeopt_payload_model import file_sha256
from .timeopt_feedback_search import ARMS
from .timeopt_payload_tasks import OUT

RESULTS=ROOT/"research/joint_time_oracle/results/timeopt_feedback_20261109"
MODULES=(
    "research/joint_time_oracle/timeopt_payload_model.py",
    "research/joint_time_oracle/timeopt_payload_tasks.py",
    "research/joint_time_oracle/timeopt_payload_collision.py",
    "research/joint_time_oracle/timeopt_oracle_contract.py",
    "research/joint_time_oracle/timeopt_k107_worker.py",
    "research/joint_time_oracle/timeopt_k107_service.py",
    "research/joint_time_oracle/timeopt_k107_independent_audit.py",
    "research/joint_time_oracle/timeopt_feedback_search.py",
    "research/joint_time_oracle/timeopt_feedback_run.py",
    "research/joint_time_oracle/timeopt_campaign_freeze.py",
    "research/joint_time_oracle/timeopt_hard_case_registry.py",
    "research/joint_time_oracle/TIMEOPT_GUIDED_NOMINAL_PAYLOAD_PROTOCOL.md",
)


def current(*,source:Path,results:Path,registry:Path,
            parity:Path,other_revision:str)->dict:
    if not other_revision or len(other_revision)!=40:
        raise ValueError("pin full time_optimal_curobo master SHA")
    tasks=json.loads((source/"tasks.json").read_text())
    registry_data=json.loads(registry.read_text())
    qualified={
        row["case_id"] for row in registry_data["cases"]
        if row["class"] in (
            "NATIVE_FAILED_ALTERNATIVE_QUALIFIED",
            "NATIVE_FEASIBLE_BUT_DURATION_SUBOPTIMAL")
        and row["source_group"]=="development"
    }
    if not qualified:
        raise ValueError("cannot freeze zero demonstrated development hard cases")
    ledger={}
    for case in sorted(qualified):
        for arm in ARMS:
            path=results/arm/case/"search_result.json"
            evidence=results/arm/case/"independent_k107_qualification.json"
            if not path.exists() or not evidence.exists():
                raise ValueError("development arm not fully evaluated: "+case+"/"+arm)
            summary=json.loads(path.read_text())
            if summary.get("pilot_only_not_scored") is True:
                raise ValueError("pilot-only result cannot be frozen as scored")
            ledger[case+"/"+arm]={
                "search_sha256":file_sha256(path),
                "independent_audit_sha256":file_sha256(evidence)}
    model=json.loads(parity.read_text())
    if not model.get("model_and_collision_parity_pass"):
        raise ValueError("unqualified native/Pinocchio payload model")
    code={p:file_sha256(ROOT/p) for p in MODULES}
    return {
        "schema_version":1,"source_task_sha256":file_sha256(source/"tasks.json"),
        "source_manifest_sha256":file_sha256(source/"source_manifest.json"),
        "hard_case_registry_sha256":file_sha256(registry),
        "model_parity_sha256":file_sha256(parity),
        "other_repo_master_sha":other_revision,
        "development_completed":ledger,
        "source_code_sha256":code,
        "holdout_policy":"no tuning of proposal radii, K107 budget, reranking, times or density after freeze",
    }


def freeze(*,source:Path,results:Path,registry:Path,
           parity:Path,other_revision:str)->dict:
    data=current(source=source,results=results,registry=registry,
                 parity=parity,other_revision=other_revision)
    path=results/"holdout_freeze.json"
    if path.exists() and json.loads(path.read_text())!=data:
        raise ValueError("cannot overwrite source-disjoint holdout freeze")
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    return data


def verify(*,source:Path,results:Path,registry:Path,
           parity:Path,other_revision:str)->dict:
    path=results/"holdout_freeze.json"
    if not path.exists():
        raise ValueError("source-disjoint holdout requires frozen development")
    frozen=json.loads(path.read_text())
    if frozen!=current(source=source,results=results,registry=registry,
                       parity=parity,other_revision=other_revision):
        raise ValueError("method, model, K107 revision or development evidence changed")
    return frozen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=OUT)
    p.add_argument("--results",type=Path,default=RESULTS)
    p.add_argument("--registry",type=Path,required=True)
    p.add_argument("--parity-receipt",type=Path,required=True)
    p.add_argument("--timeopt-sha",required=True)
    a=p.parse_args()
    record=freeze(source=a.source,results=a.results,registry=a.registry,
                  parity=a.parity_receipt,other_revision=a.timeopt_sha)
    print(json.dumps({"development_arms":len(record["development_completed"]),
                      "K107_commit_sha":record["other_repo_master_sha"]},indent=2))


if __name__=="__main__":
    main()
