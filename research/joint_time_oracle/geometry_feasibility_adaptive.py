# SPDX-License-Identifier: Apache-2.0
"""Exact analytic witness replay around independent dynamic/collision extrema.

The witness follows a QuinticGeometry with the analytic smoothstep timing
s(t/T). The preliminary 8193-point scan is NOT proof. Reconstruct the
same exact curve at additional times, then repeat the independent
cohort_validate with --methods W_adaptive --extra ARM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cohort_geometry import QuinticGeometry
from .geometry_feasibility_core import sha256_file, timed_states
from .geometry_feasibility_discover import OUT
from .torque_active import read_profile


def refined_phases(diag: dict, T: float, *, base_count: int = 8193):
    u = np.linspace(0, 1, base_count)
    extra = []
    for family in diag.get("families", {}).values():
        for level in ("physical", "scaled95"):
            value = family.get(level, {}).get("time_s")
            if value is not None:
                center = float(value) / T
                extra.extend(
                    np.clip(
                        center + np.array([-4, -2, -1, -0.25, 0, 0.25, 1, 2, 4]) / 8192,
                        0,
                        1,
                    )
                )
    for key in ("world_min_time_s", "self_min_time_s"):
        value = diag.get(key)
        if value is not None:
            center = float(value) / T
            extra.extend(
                np.clip(
                    center + np.array([-4, -2, -1, -0.25, 0, 0.25, 1, 2, 4]) / 8192,
                    0,
                    1,
                )
            )
    return np.unique(np.r_[u, extra])


def replay(arm: Path, *, case_filter: str | None = None) -> dict:
    diagfile = arm / "independent_sampled.json"
    if not diagfile.exists():
        raise ValueError("missing independent base sampled audit")
    data = json.loads(diagfile.read_text())
    tasks = json.loads((arm / "tasks.json").read_text())
    outcomes = {}
    for task in tasks:
        case = task["case_id"]
        if case_filter and case_filter != case:
            continue
        folder = arm / case
        result = {"status": "ERROR"}
        try:
            prof = read_profile(folder / "torque_profile.json")
            geomdoc = json.loads((folder / "w_geometry.json").read_text())
            if geomdoc["profile_sha256"] != prof["profile_sha256"]:
                raise ValueError("candidate geometry/profile mismatch")
            basefile = folder / "w_dense.npz"
            with np.load(basefile) as datafile:
                base = {name: np.asarray(datafile[name]) for name in datafile.files}
            T = float(base["time"][-1])
            diag = data.get(case + ":W")
            if (
                not isinstance(diag, dict)
                or diag.get("torque_profile_sha256") != prof["profile_sha256"]
            ):
                raise ValueError("independent base audit profile mismatch")
            phases = refined_phases(diag, T)
            geometry = QuinticGeometry(np.asarray(geomdoc["controls"], dtype=float))
            states = timed_states(geometry, T, n=8193, extra_u=phases)
            indices = np.searchsorted(states["phase"], base["phase"])
            if (
                np.any(indices >= len(states["phase"]))
                or np.max(abs(states["phase"][indices] - base["phase"])) > 1e-12
            ):
                raise ValueError("source phase grid not recovered")
            errors = {
                key: float(np.max(abs(states[key][indices] - base[key])))
                for key in ("q", "qd", "qdd", "jerk", "time")
            }
            if any(value > 1e-9 for value in errors.values()):
                raise ValueError(
                    "exact analytic witness replay mismatched frozen samples"
                )
            dest = folder / "w_adaptive_dense.npz"
            np.savez_compressed(dest, **states)
            result = {
                "status": "REPLAYED",
                "source_witness_sha256": sha256_file(basefile),
                "adaptive_sha256": sha256_file(dest),
                "samples": len(states["time"]),
                "added_samples": len(states["time"]) - len(base["time"]),
                "max_replay_errors": errors,
                "profile_sha256": prof["profile_sha256"],
            }
        except Exception as exc:
            result["reason"] = repr(exc)
        outcomes[case] = result
        (folder / "w_adaptive_replay.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(case, result["status"], flush=True)
    (arm / "w_adaptive_replay_summary.json").write_text(
        json.dumps(outcomes, indent=2) + "\n"
    )
    return outcomes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=OUT)
    p.add_argument("--scenario", required=True)
    p.add_argument("--case")
    a = p.parse_args()
    print(json.dumps(replay(a.root / a.scenario, case_filter=a.case), indent=2))


if __name__ == "__main__":
    main()
