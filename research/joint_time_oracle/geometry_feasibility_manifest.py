# SPDX-License-Identifier: Apache-2.0
"""Build a deterministic checksum inventory for frozen numerical evidence.

This does not upload large NPZ trajectory archives. It records their paths,
sizes and SHA256 to link the compact committed report to retained local
raw evidence. Excludes generated manifest itself to avoid self-reference.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .geometry_feasibility_core import sha256_file
from .geometry_feasibility_discover import OUT
from .geometry_feasibility_report import REPORT

ROOT=Path(__file__).resolve().parents[2]


def generate(root:Path,report:Path,*,source:Path|None=None)->dict:
    if not root.exists():
        raise FileNotFoundError("raw experiment root unavailable")
    sources=[("raw",root)]
    if source is not None:
        if not source.exists():raise FileNotFoundError("source inventory unavailable")
        sources.append(("nominal_source",source))
    items=[]
    for label,folder in sources:
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            rel=path.relative_to(folder)
            if path.is_symlink():
                raise ValueError("symlink inside evidence root: "+str(path))
            items.append({"source":label,"relative_path":str(rel),
                          "bytes":path.stat().st_size,"sha256":sha256_file(path)})
    try:
        commit=subprocess.check_output(["git","rev-parse","HEAD"],
                                       cwd=ROOT,text=True).strip()
    except Exception:
        commit="UNAVAILABLE"
    payload={"schema_version":1,"raw_root":str(root.resolve()),
             "source_root":str(source.resolve()) if source is not None else None,
             "implementation_commit":commit,
             "artifact_count":len(items),"items":items,
             "note":"local NPZ files are not embedded in Git; verify against this inventory"}
    report.mkdir(parents=True,exist_ok=True)
    (report/"artifact_manifest.json").write_text(json.dumps(payload,indent=2)+"\n")
    return {"artifact_count":len(items),"implementation_commit":commit,
            "manifest":str(report/"artifact_manifest.json")}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUT)
    p.add_argument("--source",type=Path)
    p.add_argument("--report",type=Path,default=REPORT)
    a=p.parse_args()
    print(json.dumps(generate(a.root,a.report,source=a.source),indent=2))


if __name__=="__main__":
    main()
