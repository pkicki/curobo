# SPDX-License-Identifier: Apache-2.0
"""Freeze TimeOpt-feedback algorithm and source-grouped development results.

The registered source distribution may be large (>180 source groups,
paired masses and COMs). Freeze >=10 complete development SOURCE GROUPS,
all three matched policies, before any confirmation/holdout scored run.
No result-dependent retuning of proposals or oracle acceptance thresholds
after freeze. Changes require new branch/result root and fresh holdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cohort_retime import ROOT
from .payload_timeopt_contract import ARMS,OUT,immutable_json,source_sha

METHOD_FILES=(
    "research/joint_time_oracle/payload_timeopt_contract.py",
    "research/joint_time_oracle/payload_model_parity.py",
    "research/joint_time_oracle/payload_parity_qualify.py",
    "research/joint_time_oracle/payload_qualify_sources.py",
    "research/joint_time_oracle/payload_hardcase_source.py",
    "research/joint_time_oracle/foreign_toppra_newton.py",
    "research/joint_time_oracle/timeopt_feedback_search.py",
    "research/joint_time_oracle/payload_independent_validate.py",
    "research/joint_time_oracle/payload_joint_freeze.py",
    "research/joint_time_oracle/NOMINAL_PAYLOAD_TIMEOPT_GUIDED_PROTOCOL.md",
)
MIN_DEVELOPMENT_SOURCE_GROUPS=10


def state(root:Path,*,minimum_sources:int=MIN_DEVELOPMENT_SOURCE_GROUPS)->dict:
    qualified=root/"qualified"/"qualified_tasks.json"
    if not qualified.exists():
        raise FileNotFoundError("no parity-qualified frozen task inventory")
    rows=json.loads(qualified.read_text())["tasks"]
    sources={t["source_task_id"] for t in rows if t["split"]=="development"}
    evidence={}
    complete=set()
    for sid in sorted(sources):
        subset=[x for x in rows if x["source_task_id"]==sid]
        # At least one admitted payload scenario for the source group;
        # do not select only retrospectively successful candidate paths.
        for task in subset:
            if task["admission"]!="ENDPOINT_ADMITTED":
                continue
            case=task["case_id"]
            paths=[root/"search"/arm/case/"search_result.json" for arm in ARMS]
            if not all(p.exists() for p in paths):
                continue
            info=[json.loads(p.read_text()) for p in paths]
            if any(x.get("source_case")!=case or x.get("arm")!=arm
                   for x,arm in zip(info,ARMS)):
                raise ValueError("development evidence arm/case mismatch")
            banks={x["shared_bank_sha256"] for x in info}
            if len(banks)!=1:
                raise ValueError("matched development methods used different bank")
            complete.add(sid)
            evidence[case]={a:source_sha(p) for a,p in zip(ARMS,paths)}
    if len(complete)<minimum_sources:
        raise ValueError(f"need {minimum_sources} complete DEVELOPMENT source groups "
                         f"for method freeze, found {len(complete)}")
    file_shas={f:source_sha(ROOT/f) for f in METHOD_FILES}
    return {"schema":1,"complete_development_source_groups":sorted(complete),
            "completed_development_case_hashes":evidence,
            "qualification_task_registry_sha256":source_sha(qualified),
            "implementation_hashes":file_shas,
            "source_group_minimum":minimum_sources,
            "constraint_scope":"nominal effort, independently paired payload mass/COM/collision",
            "study":"feedback guided vs same-bank ranking-only vs no-feedback global"}


def freeze(root:Path,*,minimum_sources:int=MIN_DEVELOPMENT_SOURCE_GROUPS)->dict:
    document=state(root,minimum_sources=minimum_sources)
    immutable_json(root/"feedback_holdout_freeze.json",document)
    return document


def verify(root:Path)->dict:
    file=root/"feedback_holdout_freeze.json"
    if not file.exists():
        raise ValueError("confirmation/holdout forbidden before complete development freeze")
    frozen=json.loads(file.read_text())
    current=state(root,minimum_sources=frozen["source_group_minimum"])
    if frozen!=current:
        raise ValueError("source/method/development results changed since feedback freeze")
    return frozen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUT)
    p.add_argument("--min-source-groups",type=int,default=MIN_DEVELOPMENT_SOURCE_GROUPS)
    a=p.parse_args()
    data=freeze(a.root,minimum_sources=a.min_source_groups)
    print(json.dumps({"frozen_source_groups":len(data["complete_development_source_groups"]),
                      "frozen_cells":len(data["completed_development_case_hashes"])},indent=2))


if __name__=="__main__":
    main()
