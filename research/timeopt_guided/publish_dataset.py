# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a self-contained, checksummed benchmark release (problems + best-known solutions).

Layout of the output directory:
  README.md                     dataset card: problem contract, conventions, splits, usage
  contract.json                 machine-readable constraint contract and limits
  data/problems_{main,hard}.jsonl
  solutions/{main,hard}/<problem_id>.npz   best-known trajectory (time,q,qd,qdd,jerk; float32)
  scenes/{free,table,cluttered}.yml
  robot/franka_panda.urdf, robot/franka.yml, robot/LICENSE*, robot/NOTICE
  tools/validate_core.py, tools/validate.py, tools/requirements.txt
  checksums.sha256
Everything a downstream repository needs to load the problems, certify its own
trajectories with the SAME independent checks, and compare durations with the BKS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .common import ROOT, read_json

R = ROOT / "research/timeopt_guided/results"
VERSION = "1.0.0"
NAME = "panda-payload-motion-benchmark"
SCENE_FILES = {"free": None, "table": "collision_test.yml", "cluttered": "collision_primitives_3d.yml"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_validator(out: Path) -> None:
    src = (ROOT / "research/timeopt_guided/independent_validate.py").read_text()
    head_old = src[src.index("ROOT = Path(__file__)"):src.index("PARITY_ABS_NM")]
    head_new = '''DATASET = Path(__file__).resolve().parents[1]
URDF = DATASET / "robot/franka_panda.urdf"
ROBOT_YML = DATASET / "robot/franka.yml"
SCENE_DIR = DATASET / "scenes"
SCENES = {"free": "free.yml", "table": "table.yml", "cluttered": "cluttered.yml"}
ROOT = DATASET
'''
    src = src.replace(head_old, head_new)
    src = src.replace('''        f = SCENES[scene]
        path = None if f is None else (ROOT / f if "/" in f else SCENE_DIR / f)''',
                      '''        f = SCENES[scene]
        path = SCENE_DIR / f''')
    src = src.replace('"""Independent qualification of exported trajectories (Pinocchio env, NO torch/cuRobo).',
                      '"""Benchmark validator core (Pinocchio + NumPy + SciPy + PyYAML; no cuRobo, no torch).')
    (out / "tools/validate_core.py").write_text(src)
    (out / "tools/__init__.py").write_text("")
    (out / "tools/validate.py").write_text('''# SPDX-License-Identifier: Apache-2.0
"""Certify trajectories against the benchmark contract and compare with best-known solutions.

usage: python tools/validate.py --problems data/problems_hard.jsonl --trajectories <dir> [--out report.json]
<dir>/<problem_id>.npz must contain time [N], q, qd, qdd, jerk [N, 7] (rest to rest, start->goal).
A trajectory is CERTIFIED iff the checks in README.md ("Contract") pass, including the
inter-sample collision gap check (sample densely: >= 4096 samples recommended).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_core import validate  # noqa: E402

DATASET = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problems", type=Path, required=True)
    ap.add_argument("--trajectories", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ids", nargs="*", default=None)
    a = ap.parse_args()
    contract = json.loads((DATASET / "contract.json").read_text())
    rows = []
    for line in a.problems.read_text().splitlines():
        p = json.loads(line)
        if a.ids and p["problem_id"] not in a.ids:
            continue
        f = a.trajectories / f"{p['problem_id']}.npz"
        if not f.exists():
            rows.append({"problem_id": p["problem_id"], "submitted": False})
            continue
        z = np.load(f)
        traj = {k: np.asarray(z[k], float) for k in ("time", "q", "qd", "qdd", "jerk")}
        meta = {"payload": p["payload"], "scene": p["scene"], "start": p["start"], "goal": p["goal"],
                "limits": contract["limits"], "gap_check": True}
        v = validate(traj, meta)
        d = float(traj["time"][-1] - traj["time"][0])
        rows.append({"problem_id": p["problem_id"], "submitted": True, "certified": bool(v.get("certified")),
                     "duration_s": d, "best_known_s": p["best_known"]["duration_s"],
                     "ratio_to_best_known": d / p["best_known"]["duration_s"], "checks": v})
        print(p["problem_id"], "CERTIFIED" if v.get("certified") else "REJECTED",
              "%.4fs (best known %.4fs)" % (d, p["best_known"]["duration_s"]), flush=True)
    sub = [r for r in rows if r["submitted"]]
    cert = [r for r in sub if r["certified"]]
    summary = {"problems": len(rows), "submitted": len(sub), "certified": len(cert),
               "median_ratio_to_best_known": float(np.median([r["ratio_to_best_known"] for r in cert])) if cert else None}
    print(json.dumps(summary, indent=2))
    if a.out:
        a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, default=float))


if __name__ == "__main__":
    main()
''')
    (out / "tools/requirements.txt").write_text("numpy>=1.24\nscipy>=1.10\npyyaml>=6\npin>=3.0  # Pinocchio (pip package name: pin)\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    out = a.output
    if out.exists():
        raise SystemExit(f"{out} exists")
    for d in ("data", "solutions/main", "solutions/hard", "scenes", "robot", "tools"):
        (out / d).mkdir(parents=True)
    # robot model, scenes, licences
    fd = ROOT / "curobo/content/assets/robot/franka_description"
    shutil.copy(fd / "franka_panda.urdf", out / "robot/franka_panda.urdf")
    shutil.copy(fd / "LICENSE", out / "robot/LICENSE.franka_ros")
    shutil.copy(ROOT / "curobo/content/configs/robot/franka.yml", out / "robot/franka.yml")
    shutil.copy(ROOT / "LICENSE", out / "robot/LICENSE.curobo")
    shutil.copy(ROOT / "LICENSE_ASSETS", out / "robot/NOTICE.assets")
    for name, f in SCENE_FILES.items():
        if f is None:
            (out / "scenes/free.yml").write_text("# Obstacle-free scene (no world obstacles).\n{}\n")
        else:
            shutil.copy(ROOT / "curobo/content/configs/scene" / f, out / f"scenes/{name}.yml")
    write_validator(out)
    # contract (limits from franka.yml as loaded by cuRobo; payload convention)
    bank_main = R / "benchmark_20261111/bank"
    any_traj = next((R / "bks_hard_20261115/cells").glob("*/*/bks_0.traj.json"))
    limits = read_json(any_traj)["limits"]
    contract = {
        "name": NAME, "version": VERSION, "robot": "Franka Panda (7 DoF), fingers locked at 0.04 m",
        "joint_names": [f"panda_joint{i}" for i in range(1, 8)],
        "limits": limits, "limit_scale": 0.95, "relative_tolerance": 1e-3,
        "rest_to_rest": {"position_error_rad": 1e-3, "end_speed_rad_s": 1e-2},
        "payload_convention": ("rigid body on 'attached_object', a fixed child of 'panda_hand' with identity "
                               "transform; com_m in the panda_hand frame; solid sphere of radius_m (inertia "
                               "0.4*m*r^2 about the COM); the same sphere is a collision sphere; the payload "
                               "replaces cuRobo's 0.01 kg attached_object placeholder"),
        "collision_model": ("robot/franka.yml collision_spheres + payload sphere vs scene primitives; "
                            "self-collision pairs from franka.yml self_collision_ignore/buffer; clearance >= 0; "
                            "inter-sample gap check"),
    }
    (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    # problems + solutions
    counts = {}
    for bench in ("main", "hard"):
        bks_m = read_json(R / f"bks_{bench}_20261115/best_known_manifest.json")
        bks = {(p["task_id"], p["payload"]): p for p in bks_m["problems"]}
        rows = []
        if bench == "main":
            man = read_json(R / "benchmark_20261111/manifest.json")
            tasks = {t["task_id"]: t for t in read_json(bank_main / "tasks.json")["tasks"]}
            payloads = {p["label"]: p for p in read_json(bank_main / "tasks.json")["payloads"]}
            items = [(c["task_id"], c["payload"], c) for c in man["cells"] if c["label"] == "CERTIFIED_SOLVABLE"]
        else:
            man = read_json(R / "hard_benchmark_20261114/manifest.json")
            payloads = {p["label"]: p for p in read_json(R / "eval_20261114/hard_bank/tasks.json")["payloads"]}
            items = [(p["cell_id"], p["payload"], p) for p in man["problems"]]
        for tid, pl, info in items:
            b = bks[(tid, pl)]
            pid = f"{tid}__{pl}" if bench == "main" else tid
            src = R / f"bks_{bench}_20261115/cells/{tid}/{pl}/{b['bks_candidate']}.traj.npz"
            z = np.load(src)
            dst = out / f"solutions/{bench}/{pid}.npz"
            np.savez_compressed(dst, **{k: z[k].astype(np.float32) for k in ("time", "q", "qd", "qdd", "jerk")})
            if bench == "main":
                t = tasks[tid]
                row = {"problem_id": pid, "benchmark": "main", "split": t["split"], "subsets": [t["stratum"]],
                       "scene": t["scene"], "start": t["start"], "goal": t["goal"],
                       "metrics": {"oracle_rho": info.get("rho"), "endpoint_gravity_util": info.get("endpoint_gravity_util_bank")}}
            else:
                row = {"problem_id": pid, "benchmark": "hard", "split": info["split"], "subsets": info["labels"],
                       "scene": info["scene"], "start": info["start"], "goal": info["goal"],
                       "metrics": {k: info.get(k) for k in ("rho_star_hat", "tightness_driver", "detour_ratio",
                                                            "peak_free", "L_free", "L_static", "swing_ratio",
                                                            "peak_swing", "segments", "length_ratio")
                                   if info.get(k) is not None}}
            pay = payloads[pl]
            row["payload"] = {"label": pl, "mass_kg": pay["mass_kg"], "com_m": pay["com_m"], "radius_m": pay["radius_m"]}
            kind = ("method_" + b["bks_candidate"].split("method_")[1].split("_R1")[0]
                    if b["bks_candidate"].startswith("method_") else "bks_" + b["bks_stage"])
            row["best_known"] = {"duration_s": float(np.load(dst)["time"][-1]), "file": f"solutions/{bench}/{pid}.npz",
                                 "sha256": sha(dst), "source": kind,
                                 "note": "upper bound on the optimum (best certified trajectory found)"}
            rows.append(row)
        rows.sort(key=lambda r: r["problem_id"])
        (out / f"data/problems_{bench}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        counts[bench] = len(rows)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (out / "data/provenance.json").write_text(json.dumps({
        "version": VERSION, "generated_from_commit": head,
        "repository": "pkicki/curobo branch exp/timeopt-guided-planner-20261110",
        "source_manifests": {
            "main_benchmark": read_json(R / "benchmark_20261111/manifest.json")["manifest_sha256"],
            "hard_benchmark": read_json(R / "hard_benchmark_20261114/manifest.json")["manifest_sha256"],
            "main_bks": read_json(R / "bks_main_20261115/best_known_manifest.json")["manifest_sha256"],
            "hard_bks": read_json(R / "bks_hard_20261115/best_known_manifest.json")["manifest_sha256"]},
        "counts": counts}, indent=2) + "\n")
    (out / "README.md").write_text(readme(counts))
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "checksums.sha256")
    (out / "checksums.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files))
    print(counts)


def readme(counts: dict) -> str:
    return f"""---
license: apache-2.0
pretty_name: Panda payload motion-planning benchmark (certified, with best-known solutions)
tags:
- robotics
- motion-planning
- trajectory-optimization
- time-optimal
---

# {NAME} v{VERSION}

Rest-to-rest motion-planning problems for a 7-DoF Franka Panda carrying a rigid payload
(0–5 kg) under its **original** actuator limits, in three scenes. Every problem is
**certified solvable** (an independently certified trajectory exists) and ships with a
**best-known solution** (BKS): the fastest independently certified trajectory found.
BKS durations are upper bounds on the optimum, not optimality claims.

| set | problems | development | confirmation |
|---|---|---|---|
| main | {counts['main']} | random start/goal pairs (strata: `broad`, gravity-`enriched`) | held out |
| hard | {counts['hard']} | `TIGHT_STATIC`, `DETOUR`, `SWING` (see below) | held out |

Use the `development` split for tuning and report results on `confirmation`.

## Contract (what counts as a valid solution)

A trajectory `time[N], q, qd, qdd, jerk [N, 7]` (joints `panda_joint1..7`, rad, s) is
**certified** if all of the following hold (`tools/validate.py` implements them):

1. It starts at `start` and ends at `goal` at rest: position error ≤ 1e-3 rad and
   |qd| ≤ 1e-2 rad/s at both ends.
2. Joint positions stay within the limits in `contract.json`.
3. |qd|, |qdd| and |jerk| ≤ 0.95 × the limits in `contract.json`.
4. |τ| ≤ 0.95 × effort limits. τ is inverse dynamics (Pinocchio RNEA on
   `robot/franka_panda.urdf`, fingers at 0.04 m), aligned to cuRobo's inertial
   conventions and including the payload. The relative tolerance is 1e-3.
5. It is collision free: the sphere model of `robot/franka.yml` plus the payload sphere,
   against the scene primitives (cuboids, spheres, capsules), and self-collision under
   franka.yml's ignore lists and buffers. Clearance must be ≥ 0, with an
   **inter-sample gap check**: each sample's clearance must exceed the largest sphere
   displacement to its neighbouring samples (×2 for self pairs). Sample densely
   (BKS trajectories use 8193 samples).

Payload: a rigid body on the link `attached_object`, a fixed child of `panda_hand` with
identity transform. `com_m` is in the panda_hand frame; the body is a solid sphere of
`radius_m` (inertia 0.4·m·r²), and the same sphere is a collision sphere. See
`contract.json`.

**cuRobo users:** cuRobo (at the commit used here) silently ignores `sphere` obstacles in
scene configs. Give spheres to cuRobo as zero-length capsules (base = tip) to plan in the
same world the validator checks.

## Hard subsets

* `TIGHT_STATIC`: the minimax static gravity utilisation over collision-free routes is
  ρ\\* ∈ [0.90, 0.97] (ratio to 0.95·τmax). The straight joint-space line is statically
  infeasible, and the solution needs ≥ 3 segments or ≥ 1.3× the straight length.
* `DETOUR`: the shortest collision-free route is statically overloaded (peak ≥ 1.05).
  The shortest route with peak ≤ 0.97 is ≥ 2× longer.
* `SWING`: a DETOUR problem that also has a ≥ 1.5× shorter route through a static
  overload (1.01–1.3), traversed with a dynamically feasible "swing" (certified).
  `SWING_MARGINAL` marks overloads < 1%.

## Files

* `data/problems_{{main,hard}}.jsonl`: one problem per line: `problem_id`, `split`,
  `subsets`, `scene`, `payload`, `start`, `goal`, `metrics`, and `best_known`
  (`duration_s`, `file`, `sha256`, `source`).
* `solutions/<set>/<problem_id>.npz`: BKS trajectory (`time, q, qd, qdd, jerk`, float32).
* `scenes/*.yml`, `robot/*`: world and robot models, with licences (Apache-2.0;
  Franka URDF from franka_ros 0.7.0).
* `tools/validate.py`: certification and comparison with the BKS
  (`pip install -r tools/requirements.txt`).
* `data/provenance.json`, `checksums.sha256`: provenance and integrity.

## Usage

```python
from huggingface_hub import snapshot_download
import json, numpy as np
root = snapshot_download("<namespace>/{NAME}", repo_type="dataset")
problems = [json.loads(l) for l in open(f"{{root}}/data/problems_hard.jsonl")]
p = problems[0]
bks = np.load(f"{{root}}/{{p['best_known']['file']}}")
```

```bash
python tools/validate.py --problems data/problems_hard.jsonl --trajectories my_trajs/ --out report.json
```

## Provenance

Problems were selected before any planner was evaluated on them: random task banks for
main; roadmap-based selection for hard. Solvability was certified with independent
witnesses. BKS were obtained offline from diverse roadmap routes, planner outputs and
witnesses, retimed by a TOPP-RA + Newton retimer and refined by retimer-guided path
optimisation, then certified with the validator above. See `data/provenance.json`.
"""


if __name__ == "__main__":
    main()
