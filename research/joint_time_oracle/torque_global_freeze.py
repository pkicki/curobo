# SPDX-License-Identifier: Apache-2.0
"""Immutable methods, task bank and independent dev receipts before holdout.

A three-task comparison cannot produce population-level statistical claims.
The two previously source-disjoint feasibility examples are frozen ALGORITHM
holdouts only. They were selected previously, not newly sampled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cohort_retime import ROOT
from .geometry_feasibility_core import sha256_file
from .torque_global_bank import DEVELOPMENT_CASE, HOLDOUT_CASES, MAX_PROPOSALS, OUTPUT, SEED
from .torque_global_screen import ARMS

SOURCE_FILES = (
    "research/joint_time_oracle/torque_global_bank.py",
    "research/joint_time_oracle/torque_global_screen.py",
    "research/joint_time_oracle/torque_global_freeze.py",
    "research/joint_time_oracle/torque_global_report.py",
    "research/joint_time_oracle/torque_dual_guidance.py",
    "research/joint_time_oracle/geometry_feasibility_discover.py",
    "research/joint_time_oracle/geometry_feasibility_core.py",
    "research/joint_time_oracle/geometry_feasibility_receipts.py",
)


def current(root:Path,*,seed:int=SEED,budget:int=MAX_PROPOSALS,
            registry_file:Path|None=None)->dict:
    if budget!=MAX_PROPOSALS:
        raise ValueError("unregistered bank proposal budget")
    bankmeta=root/"bank"/DEVELOPMENT_CASE/"bank.json"
    if not bankmeta.exists():
        raise FileNotFoundError("missing frozen development candidate bank")
    bank=json.loads(bankmeta.read_text())
    if bank["seed"]!=seed or bank["proposal_count"]!=budget:
        raise ValueError("frozen development bank settings changed")
    evidence={}
    for arm in ARMS:
        file=root/arm/DEVELOPMENT_CASE/"screen_result.json"
        if not file.exists():
            raise FileNotFoundError("missing completed development arm: "+arm)
        item=json.loads(file.read_text())
        if (item["bank_sha256"]!=bank["bank_sha256"] or
            item.get("seed")!=seed or item["budget"]!=budget):
            raise ValueError("development arm source/control mismatch")
        receipts_file=root/arm/"w_adaptive_receipts.json"
        receipt=(json.loads(receipts_file.read_text()).get(DEVELOPMENT_CASE+":W")
                 if receipts_file.exists() else None)
        if item["preliminary_witness"] and not isinstance(receipt,dict):
            raise ValueError("development witness not independently audited")
        evidence[arm]={"result_sha256":sha256_file(file),
                       "dev_receipt":receipt,
                       "first_preliminary_proposal":item.get("first_screen_proposal")}
    files={}
    for name in SOURCE_FILES:
        file=ROOT/name
        if not file.exists():
            raise FileNotFoundError("missing frozen implementation: "+name)
        files[name]=sha256_file(file)
    return {"schema_version":1,"seed":seed,"budget":budget,
            "development_case":DEVELOPMENT_CASE,
            "holdout_case_ids":list(HOLDOUT_CASES),
            "bank_registration_sha256":sha256_file(bankmeta),
            "bank_sha256":bank["bank_sha256"],
            "dev_evidence":evidence,
            "method_source_sha256":files,
            "registry_sha256":sha256_file(registry_file) if registry_file else None,
            "method_claim":"common proposal bank, no LP vs necessary-torque LP early reject",
            "scope":"two previously source-disjoint feasibility tasks, no independent population-level inference"}


def freeze(root:Path,*,seed:int=SEED,budget:int=MAX_PROPOSALS,
           registry_file:Path|None=None)->dict:
    data=current(root,seed=seed,budget=budget,registry_file=registry_file)
    path=root/"holdout_freeze.json"
    if path.exists() and json.loads(path.read_text())!=data:
        raise ValueError("cannot overwrite immutable method/development freeze")
    if not path.exists():
        path.write_text(json.dumps(data,indent=2)+"\n")
    return data


def verify(root:Path,*,seed:int=SEED,budget:int=MAX_PROPOSALS,
           registry_file:Path|None=None)->dict:
    path=root/"holdout_freeze.json"
    if not path.exists():
        raise ValueError("holdout run forbidden before development freeze")
    found=json.loads(path.read_text())
    expected=current(root,seed=seed,budget=budget,registry_file=registry_file)
    if found!=expected:
        raise ValueError("holdout method, source bank, or development evidence changed")
    return found


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUTPUT)
    p.add_argument("--seed",type=int,default=SEED)
    p.add_argument("--budget",type=int,default=MAX_PROPOSALS)
    p.add_argument("--registry",type=Path)
    a=p.parse_args()
    print(json.dumps(freeze(a.root,seed=a.seed,budget=a.budget,
                            registry_file=a.registry),indent=2))


if __name__=="__main__":
    main()
