# SPDX-License-Identifier: Apache-2.0
"""Prospective, LP-blind bank of globally diverse quintic path proposals.

The SAME immutable bank is used by lp_gate and no_lp. No torque LP,
retimer, or previously qualified W path affects proposal construction.
Only nominal native A10 geometry and a seeded global random walk are
allowed. Historical W and full-planner failures are consulted solely for
frozen task admission. This is an adversarial three-task pilot, NOT
statistical evidence for generic robot workloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .cohort_geometry import QuinticGeometry
from .cohort_retime import ROOT
from .geometry_feasibility_core import sha256_file
from .geometry_feasibility_discover import source_native, source_tasks
from .torque_active import read_profile
from .torque_guided_rescue import fit_nominal_quintic, resolve_registry
from .torque_guided_rescue import frozen_research_sources

OUTPUT = ROOT / "research/joint_time_oracle/results/torque_global_lp_screening_20261109"
SEED = 20261109
MAX_PROPOSALS = 128
RADII = (0.25, 0.5, 0.8, 1.2, 1.6, 2.4)
BANK_MODES = ("independent", "walk", "two_lobe")
DEVELOPMENT_CASE = "trap_cluttered_08"
HOLDOUT_CASES = ("confirm_geom_table_07", "confirm_geom_cluttered_02")


def bank_seed(case: str, seed: int) -> int:
    """Deterministic case-specific sequence, never using a source W path."""
    return int(hashlib.sha256(f"global-bank:{case}:{seed}".encode()).hexdigest()[:16], 16)


def proposal_controls(base: np.ndarray, *, count: int = MAX_PROPOSALS,
                      seed: int) -> tuple[np.ndarray, list[dict]]:
    """Global perturbations and random walks; clamp endpoint derivatives."""
    baseline = np.asarray(base, dtype=np.float64)
    if baseline.shape != (22, 7) or count < 1:
        raise ValueError("expected 22-by-7 quintic reference and positive proposal budget")
    rng = np.random.default_rng(seed)
    grid = np.linspace(0., 1., len(baseline))
    proposals = np.empty((count, *baseline.shape), dtype=np.float64)
    rows = []
    walk = baseline.copy()
    for k in range(count):
        mode = BANK_MODES[k % len(BANK_MODES)]
        radius = RADII[(k // len(BANK_MODES)) % len(RADII)]
        center = float(rng.uniform(0.28, 0.72))
        sigma = float(rng.choice((0.15, 0.24, 0.36)))
        # Broad configuration-space changes; not localized by torque LP.
        direction = rng.normal(size=7)
        direction /= max(np.linalg.norm(direction), 1e-12)
        envelope = np.exp(-0.5 * ((grid - center) / sigma) ** 2)
        envelope[:3] = 0.; envelope[-3:] = 0.
        perturb = radius * envelope[:, None] * direction[None, :]
        if mode == "two_lobe":
            d2 = rng.normal(size=7)
            d2 /= max(np.linalg.norm(d2), 1e-12)
            center2 = float(rng.uniform(0.22, 0.78))
            envelope2 = np.exp(-0.5 * ((grid - center2) / 0.24) ** 2)
            envelope2[:3] = 0.; envelope2[-3:] = 0.
            perturb += 0.65 * radius * envelope2[:, None] * d2[None, :]
        if mode == "walk":
            if (k // len(BANK_MODES)) % 8 == 0:
                walk = baseline.copy()
            walk = 0.75 * walk + 0.25 * baseline + perturb
            candidate = walk.copy()
        else:
            candidate = baseline + perturb
        candidate[:3] = baseline[:3]
        candidate[-3:] = baseline[-3:]
        proposals[k] = candidate
        rows.append({"proposal":k + 1, "mode":mode, "radius_rad":radius,
                     "center_s":center,"spread_s":sigma})
    return proposals, rows


def make_bank(out: Path, case: str, *, seed: int = SEED,
              count: int = MAX_PROPOSALS, registry_file: Path | None = None) -> dict:
    stamp = time.perf_counter()
    registry = resolve_registry(registry_file)
    if case not in registry:
        raise ValueError("unknown source task")
    sources = frozen_research_sources({case:registry[case]})
    source = registry[case]["source_root"]
    discovery = registry[case]["discovery_root"]
    scenario = registry[case]["scenario_id"]
    tasks, rows = source_tasks(source)
    task = next((x for x in tasks if x["case_id"] == case), None)
    if task is None:
        raise ValueError("task missing from nominal source inventory")
    profile = read_profile(discovery / scenario / case / "torque_profile.json")
    if profile["source_task_sha256"] != task["task_sha256"]:
        raise ValueError("source task/profile digest mismatch")
    native = source_native(source,task,rows[case])
    fitted,fit_info = fit_nominal_quintic(native,task["start"],task["goal"])
    proposals,proposal_log = proposal_controls(fitted,count=count,seed=bank_seed(case,seed))
    # Pure sanity: spline construction; no LP or feasibility evaluation.
    for item in (fitted,proposals[0],proposals[-1]):
        if not np.isfinite(QuinticGeometry(item).evaluate(np.array([0.,1.]))[0]).all():
            raise ValueError("nonfinite generated geometric candidate")
    folder = out / "bank" / case
    folder.mkdir(parents=True,exist_ok=True)
    file = folder / "proposals.npz"
    metadata = folder / "bank.json"
    contract = {
        "schema_version":1,"case_id":case,"seed":seed,"proposal_count":count,
        "scenario":scenario,"task_sha256":task["task_sha256"],
        "torque_profile_sha256":profile["profile_sha256"],
        "source_hashes":sources[case],
        "registry_sha256":sha256_file(registry_file) if registry_file else None,
        "fitted_nominal_cubic_error":fit_info,
        "proposal_generator":"LP-blind global independent/walk/two_lobe; no source W controls",
        "radii_rad":list(RADII),"modes":list(BANK_MODES),
        "proposal_log":proposal_log,
    }
    if file.exists() or metadata.exists():
        if not (file.exists() and metadata.exists()):
            raise ValueError("incomplete existing bank")
        previous=json.loads(metadata.read_text())
        if any(previous.get(k)!=v for k,v in contract.items()):
            raise ValueError("existing bank registration differs")
        if sha256_file(file)!=previous["bank_sha256"]:
            raise ValueError("previous bank SHA mismatch")
        with np.load(file) as z:
            np.testing.assert_array_equal(z["proposals"],proposals)
            np.testing.assert_array_equal(z["base_controls"],fitted)
        return previous
    np.savez_compressed(file,proposals=proposals,base_controls=fitted)
    contract["bank_sha256"]=sha256_file(file)
    contract["bank_generation_wall_s"]=time.perf_counter()-stamp
    metadata.write_text(json.dumps(contract,indent=2)+"\n")
    return contract


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUTPUT)
    p.add_argument("--case",required=True)
    p.add_argument("--seed",type=int,default=SEED)
    p.add_argument("--budget",type=int,default=MAX_PROPOSALS)
    p.add_argument("--registry",type=Path)
    a=p.parse_args()
    print(json.dumps(make_bank(a.root,a.case,seed=a.seed,count=a.budget,
                               registry_file=a.registry),indent=2))


if __name__=="__main__":
    main()
