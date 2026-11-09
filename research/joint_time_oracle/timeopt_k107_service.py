# SPDX-License-Identifier: Apache-2.0
"""Persistent isolated K107 JSONL worker with provenance and timeout.

This adapter measures the all-in warmed round trip for each proposed
geometric path, avoiding Python/CUDA startup for every candidate. Each
request/response is SHA-linked and never overwritten. No invalid or
nonmatching worker output can become a successful retiming.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import select
import shlex
import subprocess
import time

from .timeopt_oracle_contract import check_response


class K107Service:
    def __init__(self,command:str,root:Path,*,timeout_s:float=90.):
        if not command or timeout_s<=0:
            raise ValueError("real K107 service command and positive timeout required")
        self.argv=shlex.split(command)+["--jsonl-service"]
        self.root=Path(root)
        self.timeout_s=timeout_s
        self.process=None
        self.stderr=None
        self.counter=0
        self.started=None

    def __enter__(self):
        self.root.mkdir(parents=True,exist_ok=True)
        self.stderr=(self.root/"k107_service.stderr.log").open("a")
        self.started=time.perf_counter()
        self.process=subprocess.Popen(self.argv,stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,stderr=self.stderr,
                                       text=True,bufsize=1)
        return self

    def __exit__(self,exc_type,exc,tb):
        if self.process is not None:
            try:
                if self.process.stdin is not None:self.process.stdin.close()
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill();self.process.wait()
        if self.stderr is not None:
            self.stderr.close()
        self.process=None
        return False

    @staticmethod
    def _sha(file:Path)->str:
        return hashlib.sha256(file.read_bytes()).hexdigest()

    def evaluate(self,req:dict)->dict:
        if self.process is None or self.process.poll() is not None:
            return {"status":"ERROR","reason":"K107_SERVICE_NOT_RUNNING"}
        self.counter+=1
        tag=f"{self.counter:06d}_{req['request_id']}"
        request_file=self.root/(tag+".request.json")
        response_file=self.root/(tag+".response.json")
        if request_file.exists() or response_file.exists():
            raise ValueError("duplicate oracle request source artifact")
        request_file.write_text(json.dumps(req,allow_nan=False,indent=2)+"\n")
        begun=time.perf_counter()
        try:
            self.process.stdin.write(json.dumps(req,allow_nan=False)+"\n")
            self.process.stdin.flush()
            ready,_,_=select.select([self.process.stdout],[],[],self.timeout_s)
            if not ready:
                self.process.kill()
                return {"status":"ERROR","reason":"K107_SERVICE_TIMEOUT",
                        "subprocess_wall_ms":(time.perf_counter()-begun)*1000.}
            line=self.process.stdout.readline()
            if not line:
                return {"status":"ERROR","reason":"K107_SERVICE_CLOSED_PIPE"}
            response=json.loads(line)
            response_file.write_text(json.dumps(response,allow_nan=False,indent=2)+"\n")
            try:
                checked=check_response(req,response)
            except (ValueError,KeyError) as exc:
                return {"status":"ERROR","reason":"WORKER_CONTRACT_MISMATCH",
                        "details":repr(exc)}
            return {**checked,
                    "subprocess_wall_ms":(time.perf_counter()-begun)*1000.,
                    "raw_response_sha256":self._sha(response_file),
                    "raw_request_sha256":self._sha(request_file),
                    "service_first_request":self.counter==1}
        except (OSError,ValueError,json.JSONDecodeError) as exc:
            return {"status":"ERROR","reason":"K107_SERVICE_IO_OR_SCHEMA_ERROR",
                    "details":repr(exc),
                    "subprocess_wall_ms":(time.perf_counter()-begun)*1000.}
