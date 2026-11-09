# SPDX-License-Identifier: Apache-2.0
"""Build independent K107 candidate-audit cohort and strict receipts.

All K107 ONLINE_SCREEN_PASS proposals from all three search arms are
registered, not just a cherry-picked best. An arm's success is the first
INDEPENDENTLY physical+Scaled95 qualified candidate, not TimeOpt's
online feasibility flag. Model mass, nominal effort, geometry and audit
trajectory hashes are frozen in both 8193 and event-aware S32768 grids.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .cohort_torque_report import sampled_acceptance
from .geometry_feasibility_core import sha256_file
from .k107_replay_independent import reconstruct, replay
from .torque_active import read_profile, canonical_hash

def atomic_json(path:Path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if path.exists() and path.read_text()!=text:
        raise ValueError("existing frozen evidence differs: "+str(path))
    if not path.exists():path.write_text(text)


def build(cohort:Path,search_root:Path,audit_root:Path,*,arm:str)->dict:
    source=cohort/"nominal_payload"
    tasks={t["case_id"]:t for t in json.loads((source/"tasks.json").read_text())}
    records=[]
    audit_tasks=[]
    pilot=[]
    for folder in sorted((search_root/arm).iterdir()):
        if not folder.is_dir() or folder.name not in tasks:
            continue
        original=tasks[folder.name]
        log=json.loads((folder/"search_result.json").read_text())
        if log["task_sha256"]!=original["task_sha256"]:
            raise ValueError("search/source task SHA mismatch")
        for trial in log["ledger"]:
            if not trial.get("oracle_online_feasible"):
                continue
            n=int(trial["proposal"])
            candidate=folder/f"c{n:04d}"
            report={"source_case_id":folder.name,"source_task_sha256":original["task_sha256"],
                    "arm":arm,"proposal":n,"status":"NOT_REPLAYED"}
            try:
                response=json.loads((candidate/"k107_response.json").read_text())
                if response.get("status")!="ONLINE_SCREEN_PASS":
                    raise ValueError("ledger and K107 response disagree")
                certificate=replay(candidate)
                new_id=f"{folder.name}_{arm}_p{n:04d}"
                new_task={k:v for k,v in original.items() if k!="task_sha256"}
                new_task["case_id"]=new_id
                new_task["source_parent_sha256"]=original["task_sha256"]
                new_task["candidate_proposal"]=n
                new_task["task_sha256"]=canonical_hash(new_task)
                old_profile=read_profile(source/folder.name/"torque_profile.json")
                p={k:v for k,v in old_profile.items() if k!="profile_sha256"}
                p.update({"case_id":new_id,"source_case_id":new_id,
                          "source_task_sha256":new_task["task_sha256"],
                          "parent_effort_profile_sha256":old_profile["profile_sha256"]})
                p["profile_sha256"]=canonical_hash(p)
                target=audit_root/arm/new_id
                target.mkdir(parents=True,exist_ok=True)
                frozen_file=target/"k107_dense.npz"
                if frozen_file.exists():
                    if sha256_file(frozen_file)!=certificate["dense_sha256"]:
                        raise ValueError("candidate independent replay modified")
                else:
                    shutil.copy2(candidate/"k107_dense.npz",frozen_file)
                for original_name,stored_name in (
                    ("k107_response.json","source_k107_response.json"),
                    ("geometry.npz","source_geometry.npz"),
                ):
                    srcfile=candidate/original_name
                    dstfile=target/stored_name
                    if dstfile.exists():
                        if sha256_file(dstfile)!=sha256_file(srcfile):
                            raise ValueError("candidate source geometry/response changed")
                    else:
                        shutil.copy2(srcfile,dstfile)
                atomic_json(target/"torque_profile.json",p)
                atomic_json(target/"k107_candidate.json",{
                    "case_id":new_id,"source_case_id":folder.name,
                    "source_task_sha256":original["task_sha256"],
                    "source_response_sha256":sha256_file(candidate/"k107_response.json"),
                    "source_controls_sha256":sha256_file(candidate/"geometry.npz"),
                    "dense_sha256":sha256_file(frozen_file),
                    "source_replay_sha256":sha256_file(candidate/"k107_replay.json"),
                    "profile_sha256":p["profile_sha256"],
                    "same_geometric_source":True})
                audit_tasks.append(new_task)
                pilot.append({"case_id":new_id,"population":original["population"],
                              "task_sha256":new_task["task_sha256"],
                              "methods":{"K107":{"status":"PRELIMINARY",
                                 "duration_s":certificate["elapsed_s"],"verification":{}}}})
                report.update({"status":"RECONSTRUCTED_FOR_INDEPENDENT_AUDIT",
                               "audit_case_id":new_id})
            except Exception as error:
                report.update({"status":"REPLAY_FAILED","reason":repr(error)})
            records.append(report)
    root=audit_root/arm
    atomic_json(root/"tasks.json",audit_tasks)
    pilotfile=root/"pilot.jsonl"
    lines="".join(json.dumps(x,sort_keys=True,allow_nan=False)+"\n" for x in pilot)
    if pilotfile.exists() and pilotfile.read_text()!=lines:
        raise ValueError("audit cohort changed after generation")
    if not pilotfile.exists():pilotfile.write_text(lines)
    atomic_json(root/"candidate_inventory.json",records)
    return {"candidates":len(records),
            "replayed":sum(x["status"]=="RECONSTRUCTED_FOR_INDEPENDENT_AUDIT" for x in records),
            "failures":sum(x["status"]=="REPLAY_FAILED" for x in records)}


def adaptive(root:Path,*,arm:str)->dict:
    source=root/arm
    tasks=json.loads((source/"tasks.json").read_text())
    done=[]
    for task in tasks:
        folder=source/task["case_id"]
        meta=json.loads((folder/"k107_candidate.json").read_text())
        if sha256_file(folder/"k107_dense.npz")!=meta["dense_sha256"]:
            raise ValueError("K107 dense changed before independent event-aware reconstruction")
        # Pull original exact timing coefficients from the linked source response
        # via an explicit SHA-addressed path supplied by the runner.
        provenance=folder/"source_k107_response.json"
        geometry_file=folder/"source_geometry.npz"
        if not provenance.exists() or not geometry_file.exists():
            raise FileNotFoundError("copy source response and geometry with verified SHA before adaptive replay")
        if sha256_file(provenance)!=meta["source_response_sha256"] or (
                sha256_file(geometry_file)!=meta["source_controls_sha256"]):
            raise ValueError("source K107 controls/geometry do not match frozen response")
        result=json.loads(provenance.read_text())
        with np.load(geometry_file) as z:
            c=np.asarray(z["controls"],dtype=float)
        x=np.asarray(result["coefficients"],dtype=float)
        knots=np.asarray(result["knots"],dtype=float)
        # Both K107 timing knot events and quintic geometry knot events.
        geom_knots=np.linspace(0,1,c.shape[0]-5+1)
        events=np.r_[knots,geom_knots]
        eps=1e-8
        events=np.unique(np.clip(np.r_[events-eps,events,events+eps],0,1))
        states=reconstruct(c,x,knots,samples=32769,extra_s=events)
        path=folder/"k107_adaptive_dense.npz"
        if path.exists():
            raise ValueError("do not overwrite frozen independent adaptive states")
        np.savez_compressed(path,**states)
        done.append({"case_id":task["case_id"],
                     "source_dense_sha256":meta["dense_sha256"],
                     "adaptive_sha256":sha256_file(path),
                     "samples":len(states["time"])})
    atomic_json(source/"adaptive_inventory.json",done)
    return {"adaptive_count":len(done)}


def receipts(root:Path,*,arm:str)->dict:
    source=root/arm
    tasks=json.loads((source/"tasks.json").read_text())
    base=json.loads((source/"independent_sampled.json").read_text())
    adaptive=json.loads((source/"independent_adaptive.json").read_text())
    results={}
    for task in tasks:
        case=task["case_id"]
        folder=source/case
        record={"case_id":case,"physical":False,"scaled95":False,"reasons":[]}
        try:
            profile=read_profile(folder/"torque_profile.json")
            meta=json.loads((folder/"k107_candidate.json").read_text())
            path=folder/"k107_dense.npz"
            refined=folder/"k107_adaptive_dense.npz"
            if (sha256_file(path)!=meta["dense_sha256"] or
                profile["profile_sha256"]!=meta["profile_sha256"] or
                task["task_sha256"]!=profile["source_task_sha256"]):
                raise ValueError("candidate task/profile/trajectory SHA mismatch")
            a=base.get(case+":K107")
            b=adaptive.get(case+":K107_adaptive")
            if not isinstance(a,dict) or not isinstance(b,dict):
                raise ValueError("missing independently sampled or adaptive K107 results")
            if (a.get("source_trajectory_sha256")!=sha256_file(path)
                    or b.get("source_trajectory_sha256")!=sha256_file(refined)):
                raise ValueError("stale independent audit; source NPZ SHA mismatch")
            p1,r1=sampled_acceptance(a,profile,scaled=False)
            p2,r2=sampled_acceptance(b,profile,scaled=False)
            s1,r3=sampled_acceptance(a,profile,scaled=True)
            s2,r4=sampled_acceptance(b,profile,scaled=True)
            record.update({"physical":bool(p1 and p2),
                           "scaled95":bool(p1 and p2 and s1 and s2),
                           "reasons":[r1,r2,r3,r4],
                           "dense_sha256":sha256_file(path),
                           "adaptive_sha256":sha256_file(refined),
                           "payload":task.get("payload"),"scope":"sampled, not continuous-time certificate"})
        except Exception as error:
            record["reasons"].append(repr(error))
        results[case+":K107"]=record
    atomic_json(source/"k107_independent_receipts.json",results)
    return {"attempted":len(results),
            "qualified_scaled95":sum(v["scaled95"] for v in results.values())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("build","adaptive","receipts"),required=True)
    p.add_argument("--cohort",type=Path)
    p.add_argument("--search-root",type=Path)
    p.add_argument("--audit-root",type=Path,required=True)
    p.add_argument("--arm",required=True)
    a=p.parse_args()
    if a.stage=="build":
        if a.cohort is None or a.search_root is None:p.error("build requires cohort and search-root")
        out=build(a.cohort,a.search_root,a.audit_root,arm=a.arm)
    elif a.stage=="adaptive":
        out=adaptive(a.audit_root,arm=a.arm)
    else:
        out=receipts(a.audit_root,arm=a.arm)
    print(json.dumps(out,indent=2))


if __name__=="__main__":main()
