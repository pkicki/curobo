# SPDX-License-Identifier: Apache-2.0
"""Opt-in pre-TimeOpt native cuRobo geometry capture, with no production edits.

Attach to an EXISTING MotionPlanner only after warmup. Wraps its research
TrajOptSolver._finalize_result method, copies the all-seed interpolation
input immediately before TimeOpt, and logs the returned seed-validation
masks after the original method. This includes rejected/failed attempts
that would otherwise not appear in the selected output.

No optimizer behavior, ordering or numerical buffers are intentionally
changed. Copying CUDA tensors to CPU synchronizes and adds diagnostic
overhead. This run must NOT be used for native planning latency comparisons.
The observer never changes a failure to success. Data unavailable before
the finalizer (e.g. graph seed generation) is NOT silently fabricated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MethodType
import time

import numpy as np


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda:stream.read(1024*1024),b""):
            h.update(part)
    return h.hexdigest()


def tensor_cpu(x):
    if x is None:
        return None
    return x.detach().cpu().numpy().copy()


class NativePreTimeOptProbe:
    def __init__(self,planner,folder:Path,tag:str):
        self.solver=planner.trajopt_solver
        self.folder=Path(folder)
        self.tag=tag
        self.original=None
        self.attempts=[]
        self._invocations=0

    def __enter__(self):
        if self.original is not None:
            raise RuntimeError("native pre-TimeOpt probe already installed")
        self.folder.mkdir(parents=True,exist_ok=True)
        self.original=self.solver._finalize_result
        probe=self

        def wrapped(solver,current_state,num_seeds,return_seeds,seed_valid_mask=None):
            index=probe._invocations
            probe._invocations+=1
            tag=f"{probe.tag}_pretimeopt_{index:03d}"
            record={"tag":tag,"stage":"before_TimeOpt","state":"UNAVAILABLE",
                    "timing_instrumented":True,"index":index}
            try:
                native=solver._result_processor.get_interpolation_input()
                data={}
                for name in ("position","velocity","acceleration","jerk",
                             "knot","knot_dt","dt"):
                    value=getattr(native,name,None)
                    if value is not None:
                        data[name]=tensor_cpu(value)
                if not all(x in data for x in ("position","knot","knot_dt")):
                    raise RuntimeError("incomplete geometric interpolation input")
                path=probe.folder/(tag+".npz")
                np.savez_compressed(path,**data)
                record.update({"state":"PRETIMEOPT_SNAPSHOT_SAVED",
                               "sha256":sha(path),"geometry_file":str(path),
                               "candidate_count":int(data["position"].shape[0]),
                               "knot_shape":list(data["knot"].shape)})
                # Only observed source masks, not inferred TimeOpt outcomes.
                snapshot=getattr(solver._result_processor,"_snapshot",None)
                if snapshot is not None:
                    for field in ("success","path_valid","seed_cost"):
                        if getattr(snapshot,field,None) is not None:
                            values=tensor_cpu(getattr(snapshot,field))
                            record["pre_"+field]=values.reshape(-1).tolist()
                if seed_valid_mask is not None:
                    record["seed_valid_mask"]=tensor_cpu(
                        seed_valid_mask).reshape(-1).tolist()
            except Exception as exc:
                record["capture_error"]=repr(exc)
            probe.attempts.append(record)
            # ALWAYS run the unchanged original, regardless of capture failure.
            try:
                out=probe.original(current_state,num_seeds,return_seeds,seed_valid_mask)
                record["finalizer_returned"]=True
                if getattr(out,"success",None) is not None:
                    record["result_success"]=tensor_cpu(out.success).reshape(-1).tolist()
                validator=getattr(getattr(solver,"_timeopt",None),"validation_masks",None)
                if isinstance(validator,dict):
                    record["timeopt_validation_masks"]={
                        key:tensor_cpu(value).reshape(-1).tolist()
                        for key,value in validator.items()}
                return out
            except Exception as exc:
                record["finalizer_returned"]=False
                record["finalizer_exception"]=repr(exc)
                raise
            finally:
                probe.flush()

        self.solver._finalize_result=MethodType(wrapped,self.solver)
        return self

    def flush(self):
        dest=self.folder/(self.tag+"_pretimeopt_inventory.json")
        dest.write_text(json.dumps(self.attempts,indent=2)+"\n")

    def __exit__(self,exc_type,exc,tb):
        self.solver._finalize_result=self.original
        self.original=None
        self.flush()
        return False
