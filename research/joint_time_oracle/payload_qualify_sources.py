# SPDX-License-Identifier: Apache-2.0
"""Promote immutable mass-only generated sources after CUDA/Pinocchio parity.

Every (payload mass, COM, scene) receipt must pass BOTH native-vs-independent
RNEA AND exact four-sphere robot+payload collision geometry before any
task is promoted to the scored experimental population. No source
endpoint, split, mass, nominal effort, or case identity may change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .payload_parity_qualify import admit_task,verify_task_payload_receipt
from .payload_timeopt_contract import (
    OUT,Payload,immutable_json,source_sha,validate_fresh_split,
)


def receipt_name(task:dict)->str:
    payload=Payload(**task["payload"])
    return f"{task['population']}__{payload.id}.json"


def promote(source:Path,receipt_root:Path,out:Path)->dict:
    frozen=json.loads((source/"tasks.json").read_text())
    original=frozen["tasks"]
    promoted=[];missing=[]
    for item in original:
        receiptfile=receipt_root/receipt_name(item)
        if not receiptfile.exists():
            missing.append({"case_id":item["case_id"],
                            "reason":"payload RNEA/collision parity receipt not found",
                            "expected":str(receiptfile)})
            continue
        receipt=json.loads(receiptfile.read_text())
        altered=admit_task(item,receipt,receiptfile)
        verify_task_payload_receipt(altered)
        promoted.append(altered)
    if missing:
        raise ValueError(f"cannot promote only a selected subset: {len(missing)} "
                         "model/scene receipts missing or unqualified")
    summary=validate_fresh_split(promoted)
    registry={"schema":1,"input_tasks_sha256":source_sha(source/"tasks.json"),
              "source_generation_sha256":source_sha(source/"source_generation.json"),
              "model_receipts":{receipt_name(t):source_sha(receipt_root/receipt_name(t))
                                for t in original},
              "task_count":len(promoted),"split":summary,
              "all_payload_collision_qualified":True,
              "nominal_actuator_effort_unchanged":True,
              "outcomes_not_consulted":True}
    immutable_json(out/"qualified_tasks.json",{"tasks":promoted})
    immutable_json(out/"qualification_registry.json",registry)
    return registry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=OUT/"source")
    p.add_argument("--receipts",type=Path,required=True)
    p.add_argument("--out",type=Path,default=OUT/"qualified")
    a=p.parse_args()
    result=promote(a.source,a.receipts,a.out)
    print(json.dumps({"task_count":result["task_count"],"split":result["split"]},indent=2))


if __name__=="__main__":
    main()
