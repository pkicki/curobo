# SPDX-License-Identifier: Apache-2.0
"""Freeze algorithm implementation and development-only outcomes before holdout.

The source-disjoint table/cluttered cases must not be used to tune
step radii, beam policy, LP tolerance, direction method, or candidate budget.
Close out the discovery task across ALL THREE arms first, then immutable
freeze the method and registered budgets, THEN evaluate heldout cases.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .geometry_feasibility_core import sha256_file
from .cohort_retime import ROOT

DEV="trap_cluttered_08"
ARMS=("dual","random_local","random_global")
MODULES=(
    "research/joint_time_oracle/torque_guided_rescue.py",
    "research/joint_time_oracle/torque_dual_guidance.py",
    "research/joint_time_oracle/geometry_feasibility_core.py",
    "research/joint_time_oracle/geometry_feasibility_scene.py",
    "research/joint_time_oracle/geometry_feasibility_discover.py",
)


def build(root:Path,*,budget:int,seed:int,registry_file:Path|None=None)->dict:
    implementations={p:sha256_file(ROOT/p) for p in MODULES}
    cases={}
    for arm in ARMS:
        result=root/arm/DEV/"search_result.json"
        cfg=root/arm/"experiment_config.json"
        if not result.exists() or not cfg.exists():
            raise FileNotFoundError("development arm missing: "+arm)
        c=json.loads(cfg.read_text())
        r=json.loads(result.read_text())["report"]
        if c.get("budget")!=budget or c.get("seed")!=seed:
            raise ValueError("development search setting mismatch")
        if r.get("budget")!=budget or r.get("seed")!=seed:
            raise ValueError("development result setting mismatch")
        receiptfile=root/arm/"w_adaptive_receipts.json"
        receipts=json.loads(receiptfile.read_text()) if receiptfile.exists() else {}
        if r.get("candidate_witness_prescreen") and (DEV+":W") not in receipts:
            raise ValueError("development witness must be independently audited before freeze")
        cases[arm]={"development_result_sha256":sha256_file(result),
                    "experiment_config_sha256":sha256_file(cfg),
                    "development_independent_receipt":receipts.get(DEV+":W")}
    return {"schema_version":1,"development_case":DEV,
            "holdout_case_ids":["confirm_geom_table_07","confirm_geom_cluttered_02"],
            "budget":budget,"seed":seed,"implementations":implementations,
            "development_evidence":cases,
            "source_registry_sha256":sha256_file(registry_file)
                 if registry_file is not None else None,
            "note":"freeze BEFORE ANY controlled-search run on heldout tasks"}


def freeze(root:Path,*,budget:int,seed:int,registry_file:Path|None=None)->dict:
    payload=build(root,budget=budget,seed=seed,registry_file=registry_file)
    path=root/"holdout_freeze.json"
    if path.exists():
        if json.loads(path.read_text())!=payload:
            raise ValueError("holdout registration differs from frozen development")
    else:
        path.write_text(json.dumps(payload,indent=2)+"\n")
    return payload


def verify_frozen_holdout(root:Path,*,budget:int,seed:int,
                          registry_file:Path|None=None):
    file=root/"holdout_freeze.json"
    if not file.exists():
        raise ValueError("holdout forbidden: freeze development first")
    frozen=json.loads(file.read_text())
    current=build(root,budget=budget,seed=seed,registry_file=registry_file)
    if frozen!=current:
        raise ValueError("holdout forbidden: method, registry or dev evidence changed")
    return frozen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--budget",required=True,type=int)
    p.add_argument("--seed",type=int,default=20261108)
    p.add_argument("--registry",type=Path)
    a=p.parse_args()
    print(json.dumps(freeze(a.root,budget=a.budget,seed=a.seed,
                            registry_file=a.registry),indent=2))


if __name__=="__main__":
    main()
