# SPDX-License-Identifier: Apache-2.0
"""Shared immutable protocol for nominal-effort, attached-payload experiments.

The payload changes robot *inertia*, never the original actuator efforts.
A mass-only attachment is a dynamics study, not a payload-collision-safe
execution; collision-geometry parity must be qualified separately.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cohort_retime import ROOT

SCHEMA=1
SOURCE_CUROBO_SHA="1597f1292ae45e9d2fe3ed2f76ec15f5126d38e2"
SOURCE_TOPP_SHA="94d25ecde95bc7f31851b0b9a78f61bd5f7cb65e"
SCENES=("free","table","cluttered")
MASS_KG=(0.,1.,2.,3.,4.,5.,6.)
COM_OFFSETS_M=((0.,0.,0.),(0.,0.,.08))
RESTART_BUDGETS=(1,4,16,32)
PROPOSAL_BUDGETS=(16,32,64,128)
ARMS=("global_unguided","timeopt_ranked","timeopt_guided")
SEED=20261110
OUT=ROOT/"research/joint_time_oracle/results/payload_timeopt_joint_20261110"


@dataclass(frozen=True)
class Payload:
    """Point-mass extra inertia at a *known* attached-object frame."""
    mass_kg:float
    com_xyz_m:tuple[float,float,float]
    collision_model:str="DYNAMICS_ONLY_UNQUALIFIED_PAYLOAD_COLLISION"

    def __post_init__(self):
        if not np.isfinite(self.mass_kg) or not 0<=self.mass_kg<=8:
            raise ValueError("invalid research attached mass")
        if len(self.com_xyz_m)!=3 or any(not np.isfinite(x) or abs(x)>.25 for x in self.com_xyz_m):
            raise ValueError("invalid payload COM")
        if self.collision_model!="DYNAMICS_ONLY_UNQUALIFIED_PAYLOAD_COLLISION":
            raise ValueError("physical payload collision requires independently qualified implementation")

    @property
    def id(self)->str:
        mm=int(round(self.mass_kg*1000))
        cm="_".join(f"{int(round((x+.25)*1000)):03d}" for x in self.com_xyz_m)
        return f"m{mm:05d}_com{cm}"


def source_sha(path:Path)->str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as inp:
        for block in iter(lambda:inp.read(1<<20),b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value:Any)->str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,
                                      separators=(",",":")).encode()).hexdigest()


def immutable_json(path:Path,data:dict):
    path=Path(path)
    new=json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if path.exists() and json.loads(path.read_text())!=data:
        raise ValueError("frozen research contract changed: "+str(path))
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(new)


def exact_effort_contract(native_bounds:np.ndarray,original_bounds:np.ndarray,
                          *,atol:float=1e-6)->dict:
    native=np.asarray(native_bounds,dtype=float)
    original=np.asarray(original_bounds,dtype=float)
    if native.shape!=(2,7) or original.shape!=(2,7) or not np.isfinite(native).all():
        raise ValueError("expected signed 2x7 effort bounds")
    if not np.allclose(native,original,atol=atol,rtol=1e-6):
        raise ValueError("payload research altered original actuator effort limits")
    if not np.all(original[0]<0) or not np.all(original[1]>0):
        raise ValueError("unexpected signed actuator convention")
    symmetric=np.allclose(-original[0],original[1],atol=atol,rtol=1e-6)
    return {"nominal_signed_effort":original.tolist(),
            "nominal_effort_sha256":canonical_sha(original.tolist()),
            "symmetric_signed_effort":bool(symmetric),
            "topp_symmetric_adapter_eligible":bool(symmetric),
            "caution":"TOPP RobotLimits is symmetric; asymmetric signed bounds REQUIRE new support/parity"}


def ensure_payload_contract(payload:Payload,source:dict):
    if source.get("payload")!=asdict(payload):
        raise ValueError("source task payload spec mismatch")
    if source.get("joint_limit_scale")!=1.0:
        raise ValueError("research must retain original actuator bounds")
    if source.get("payload_collision_qualified") is not False:
        raise ValueError("collision-qualified payload must use a separately audited adapter")


def classify_native_and_reference(*,native_status:str,topp_status:str,
                                  independent_qualified:bool)->str:
    """Avoid attributing path infeasibility to TimeOpt from finite-budget failures."""
    if native_status=="ERROR" or topp_status in ("ERROR","NOT_RUN","CONTRACT_MISMATCH"):
        return "INCONCLUSIVE_OR_UNQUALIFIED_REFERENCE"
    if native_status=="SUCCESS_QUALIFIED":
        return "NATIVE_QUALIFIED"
    if native_status=="BUDGET_EXHAUSTED" and independent_qualified:
        return "NATIVE_BUDGET_FAILURE_WITH_INDEPENDENT_FEASIBLE_WITNESS"
    if native_status=="BUDGET_EXHAUSTED" and topp_status=="CANDIDATE":
        return "REFERENCE_CANDIDATE_NOT_INDEPENDENTLY_QUALIFIED"
    return "NO_FEASIBLE_WITNESS_ESTABLISHED"


def validate_fresh_split(tasks:list[dict],*,minimum_separation:float=.25):
    """Group by source endpoint pair, NEVER split payload variants independently."""
    ids={}
    groups={}
    for task in tasks:
        sid=task["source_task_id"]
        group=task["split"]
        if group not in ("development","confirmation","holdout"):
            raise ValueError("unrecognized split")
        if sid in groups and groups[sid]!=group:
            raise ValueError("source task and payload variants leaked across splits")
        groups[sid]=group
        if task["case_id"] in ids:
            raise ValueError("duplicate research case")
        ids[task["case_id"]]=task
    by_source={}
    for task in tasks:
        by_source.setdefault(task["source_task_id"],task)
    values=list(by_source.values())
    for i,a in enumerate(values):
        a0=np.asarray(a["start"]);a1=np.asarray(a["goal"])
        for b in values[i+1:]:
            b0=np.asarray(b["start"]);b1=np.asarray(b["goal"])
            separation=min(np.linalg.norm(a0-b0)+np.linalg.norm(a1-b1),
                           np.linalg.norm(a0-b1)+np.linalg.norm(a1-b0))
            if separation<minimum_separation:
                raise ValueError("different source endpoint pairs overlap")
    return {"source_groups":len(groups),"cases":len(tasks),
            "splits":{group:sum(v==group for v in groups.values())
                      for group in ("development","confirmation","holdout")}}
