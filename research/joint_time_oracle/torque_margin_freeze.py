# SPDX-License-Identifier: Apache-2.0
"""Freeze retiming code and trap development outcomes before two holdout tasks.

Each development arm must be completed. An independently audited R
receipt is required for every development arm that constructed R,
including negative receipts; methods may not silently tune beta,
dilation grid, IPOPT numerical settings, or source trajectory after
learning holdout outcomes. Holdout sets were previously feasibility-
selected, so this prevents only *algorithm* evaluation leakage, not
population selection bias.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cohort_retime import ROOT
from .torque_margin_core import CASE_IDS, sha256_file, immutable_json
from .torque_margin_prepare import OUTPUT

MODULES=(
    "research/joint_time_oracle/torque_margin_core.py",
    "research/joint_time_oracle/torque_margin_prepare.py",
    "research/joint_time_oracle/torque_margin_retime.py",
    "research/joint_time_oracle/torque_margin_adaptive.py",
    "research/joint_time_oracle/torque_margin_receipts.py",
    "research/joint_time_oracle/torque_margin_freeze.py",
    "research/joint_time_oracle/torque_margin_report.py",
    "research/joint_time_oracle/cohort_retime.py",
    "research/joint_time_oracle/torque_global_retime.py",
)
ARMS=("dilation","tight_b980","tight_b960","tight_b940","tight_b920")
DEV=CASE_IDS[0]


def current(root:Path)->dict:
    registration=root/"margin_registration.json"
    if not registration.exists():
        raise FileNotFoundError("missing frozen W/P campaign registration")
    evidence={}
    for arm in ARMS:
        record=root/arm/DEV/"screen_result.json"
        if not record.exists():
            raise FileNotFoundError("incomplete development arm: "+arm)
        item=json.loads(record.read_text())
        rfile=root/arm/DEV/"r_dense.npz"
        receiptfile=root/arm/"r_adaptive_receipts.json"
        audited=(json.loads(receiptfile.read_text()).get(DEV+":R")
                 if receiptfile.exists() else None)
        if rfile.exists() and not isinstance(audited,dict):
            raise ValueError("constructed development R not independently audited: "+arm)
        evidence[arm]={"result_sha256":sha256_file(record),
                       "R_sha256":sha256_file(rfile) if rfile.exists() else None,
                       "audited_receipt":audited}
    implementations={}
    for name in MODULES:
        path=ROOT/name
        if not path.exists():
            raise FileNotFoundError("implementation file missing "+name)
        implementations[name]=sha256_file(path)
    return {"schema_version":1,"development_case":DEV,
            "holdout_case_ids":list(CASE_IDS[1:]),
            "source_registration_sha256":sha256_file(registration),
            "development_evidence":evidence,
            "implementation_sha256":implementations,
            "note":"development-only method freeze; no generalization claim from preselected holdouts"}


def freeze(root:Path)->dict:
    record=current(root)
    immutable_json(root/"retiming_holdout_freeze.json",record)
    return record


def verify(root:Path):
    file=root/"retiming_holdout_freeze.json"
    if not file.exists():
        raise ValueError("retiming holdout requires frozen development first")
    record=json.loads(file.read_text())
    if record!=current(root):
        raise ValueError("development evidence or retiming implementation changed after freeze")
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUTPUT)
    a=p.parse_args()
    info=freeze(a.root)
    print(json.dumps({"frozen_arms":list(info["development_evidence"]),
                      "holdouts":info["holdout_case_ids"]},indent=2))


if __name__=="__main__":
    main()
