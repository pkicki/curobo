# SPDX-License-Identifier: Apache-2.0
"""Freeze timeopt-guided method and all development outcomes before holdout.

The native-failure discovery cohort may be enriched based ONLY on the
development split. All original-effort task/mass combinations in the
holdout remain in the denominator, including unsolved or model-blocked
ones. Never tune on a heldout result, even if the native planner fails.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cohort_retime import ROOT
from .geometry_feasibility_core import sha256_file
from .timeopt_feedback_geometry import ARMS

MODULES=(
    "research/joint_time_oracle/payload_parity.py",
    "research/joint_time_oracle/nominal_payload_cohort.py",
    "research/joint_time_oracle/payload_model_parity_runner.py",
    "research/joint_time_oracle/k107_oracle_bridge.py",
    "research/joint_time_oracle/k107_replay_independent.py",
    "research/joint_time_oracle/timeopt_feedback_geometry.py",
    "research/joint_time_oracle/timeopt_feedback_execute.py",
    "research/joint_time_oracle/timeopt_feedback_qualification.py",
    "research/joint_time_oracle/nominal_hardcase_classify.py",
    "research/joint_time_oracle/native_uniform_seed_diagnostic.py",
    "research/joint_time_oracle/geometry_feasibility_scene.py",
    "research/joint_time_oracle/cohort_validate.py",
    "research/joint_time_oracle/torque_full_planner.py",
)


def current(cohort:Path,search:Path,*,retimer_sha:str)->dict:
    registration=cohort/"cohort_registration.json"
    if not registration.exists():
        raise FileNotFoundError("nominal-payload population not frozen")
    manifest=json.loads(registration.read_text())
    files={name:sha256_file(ROOT/name) for name in MODULES}
    source={x["case_id"]:x for x in json.loads(
        (cohort/"nominal_payload/tasks.json").read_text())}
    development=set(manifest["split_case_ids"]["development"])
    observations={}
    for case in sorted(development):
        arms={}
        for arm in ARMS:
            f=search/arm/case/"search_result.json"
            if f.exists():
                row=json.loads(f.read_text())
                if row["task_sha256"]!=source[case]["task_sha256"]:
                    raise ValueError("development task/source checksum mismatch")
                arms[arm]={"result_sha256":sha256_file(f),
                           "parity_sha256":row["independent_model_parity_receipt_sha256"]}
        if arms:observations[case]=arms
    if not observations:
        raise ValueError("cannot freeze empty development observations")
    if not all(set(x)==set(ARMS) for x in observations.values()):
        raise ValueError("all scored development cases require all three matched arms")
    return {"schema_version":1,"population_sha256":sha256_file(registration),
            "retimer_repository_sha256":retimer_sha,"source_file_sha256":files,
            "development_case_ids":sorted(observations),
            "development_results":observations,
            "holdout_cases_sha256":sha256_file(cohort/"nominal_payload/tasks.json"),
            "scope":"frozen algorithm before any confirmation/holdout results; physical payload on nominal effort"}


def freeze(cohort:Path,search:Path,*,retimer_sha:str)->dict:
    data=current(cohort,search,retimer_sha=retimer_sha)
    path=search/"timeopt_feedback_holdout_freeze.json"
    if path.exists():
        if json.loads(path.read_text())!=data:
            raise ValueError("cannot update frozen feedback development")
    else:
        path.write_text(json.dumps(data,indent=2)+"\n")
    return data


def verify(cohort:Path,search:Path,*,retimer_sha:str):
    path=search/"timeopt_feedback_holdout_freeze.json"
    if not path.is_file():
        raise ValueError("holdout forbidden without frozen feedback development")
    found=json.loads(path.read_text())
    if found!=current(cohort,search,retimer_sha=retimer_sha):
        raise ValueError("feedback method/development/retimer changed after freeze")
    return found


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort",required=True,type=Path)
    p.add_argument("--search-root",required=True,type=Path)
    p.add_argument("--retimer-commit",required=True)
    a=p.parse_args()
    result=freeze(a.cohort,a.search_root,retimer_sha=a.retimer_commit)
    print(json.dumps({"registered_development_cases":len(result["development_case_ids"]),
                      "source_sha_count":len(result["source_file_sha256"])},indent=2))


if __name__=="__main__":main()
