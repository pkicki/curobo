# SPDX-License-Identifier: Apache-2.0
"""Immutable code/source/development-evidence gate before payload holdouts.

All mass/COM scenarios of a task share one source-task partition.
Confirmation tasks must not influence proposal geometry, ranking,
TimeOpt failure scoring, budget or hyperparameters. This freeze
protects algorithm evaluation, not the statistical selection bias
from discovering hard cases in the development distribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .cohort_retime import ROOT
from .payload_hardcase_registry import population_split
from .payload_contract import sha256_file

FILES=(
"research/joint_time_oracle/payload_contract.py",
"research/joint_time_oracle/payload_native.py",
"research/joint_time_oracle/payload_geometry_audit.py",
"research/joint_time_oracle/payload_parity.py",
"research/joint_time_oracle/nominal_payload_task_bank.py",
"research/joint_time_oracle/native_uniform_seed_replay.py",
"research/joint_time_oracle/retimer_oracle.py",
"research/joint_time_oracle/timeopt_feedback_search.py",
"research/joint_time_oracle/run_timeopt_feedback.py",
"research/joint_time_oracle/payload_hardcase_registry.py",
"research/joint_time_oracle/payload_study_freeze.py",
"research/joint_time_oracle/TIMEOPT_GUIDED_NOMINAL_PAYLOAD_PROTOCOL_20261110.md",
)


def current(root:Path)->dict:
    root=Path(root)
    source=root/"prospective_task_manifest.json"
    profiles=root/"payload_profiles.json"
    closeout=root/"development_summary.json"
    if not all(x.exists() for x in (source,profiles,closeout)):
        raise FileNotFoundError("prospective source, payloads and development outcome not frozen")
    development=json.loads(closeout.read_text())
    if development.get("status")!="COMPLETE" or (
        development.get("independently_qualified_development_tasks",0)<20):
        raise ValueError("cannot freeze incomplete or tiny development campaign")
    for row in development.get("task_outcomes",[]):
        if population_split(row["task_id"])!="development":
            raise ValueError("confirmation task leaked into development outcome")
    return {"schema_version":1,
            "source_sha256":sha256_file(source),
            "payload_profiles_sha256":sha256_file(profiles),
            "development_sha256":sha256_file(closeout),
            "source_code_sha256":{name:sha256_file(ROOT/name) for name in FILES},
            "contract":"development task outcomes and method settings frozen before confirmation"}


def freeze(root:Path)->dict:
    p=Path(root)/"confirmation_freeze.json"
    value=current(root)
    if p.exists() and json.loads(p.read_text())!=value:
        raise ValueError("frozen confirmation protocol changed")
    if not p.exists():
        p.write_text(json.dumps(value,indent=2)+"\n")
    return value


def verify(root:Path)->dict:
    path=Path(root)/"confirmation_freeze.json"
    if not path.exists():
        raise ValueError("confirmation forbidden before development closeout freeze")
    previous=json.loads(path.read_text())
    if previous!=current(root):
        raise ValueError("confirmation forbidden: development, model or method changed")
    return previous


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(freeze(a.root),indent=2))


if __name__=="__main__":
    main()
