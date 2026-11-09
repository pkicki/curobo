# SPDX-License-Identifier: Apache-2.0
"""Opt-in observational snapshot of TimeOpt's numerical phase boundaries.

A finalization failure can arise BEFORE integration (invalid input or
timing controls), at integration, or later in path collision/dynamics
validation. Capture output of CUDA solver seed feasibility/numerical
flags, 32 squared-speed controls, TimeOpt finish status/timing and
integrated times on identical lanes. Classify the EARLIEST OBSERVED
invalid buffer, not its exclusive causal stage. The probe synchronizes
GPU-to-CPU and invalidates timing comparisons; never changes solver
inputs, return values, kernels, or production defaults.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def numeric_summary(values) -> dict:
    a = np.asarray(values)
    if a.size == 0:
        return {"status": "EMPTY", "shape": list(a.shape)}
    if a.dtype.kind == "b":
        return {
            "status": "BOOL",
            "shape": list(a.shape),
            "true": int(a.sum()),
            "false": int(a.size - a.sum()),
        }
    if a.dtype.kind not in "ifu":
        return {"status": "UNSUPPORTED_DTYPE", "shape": list(a.shape)}
    finite = np.isfinite(a)
    vals = a[finite]
    result = {
        "status": "FINITE" if bool(finite.all()) else "NONFINITE",
        "shape": list(a.shape),
        "finite": int(finite.sum()),
        "nan": int(np.isnan(a).sum()),
        "posinf": int(np.isposinf(a).sum()),
        "neginf": int(np.isneginf(a).sum()),
        "zero": int(np.sum(a == 0)),
    }
    if vals.size:
        result.update(
            {"finite_min": float(np.min(vals)), "finite_max": float(np.max(vals))}
        )
    return result


def per_lane_diagnosis(snapshot: dict) -> list[dict]:
    names = (
        "input_valid",
        "solver_seed_feasible",
        "solver_numerical_ok",
        "solver_output_timing",
        "finish_optimization_ok",
        "finished_squared_speed_controls",
        "integrated_times",
        "native_timeopt_success",
    )
    arrays = {k: np.asarray(snapshot[k]) for k in names if k in snapshot}
    size = None
    for v in arrays.values():
        if v.ndim >= 1:
            size = len(v)
            break
    if size is None:
        return []
    result = []
    for i in range(size):
        flags = {}
        problems = []
        for k, v in arrays.items():
            if v.ndim < 1 or len(v) != size:
                flags[k] = "UNAVAILABLE_SHAPE"
                continue
            row = np.asarray(v[i])
            if row.dtype.kind == "b":
                value = bool(row.all())
                flags[k] = value
                if not value:
                    problems.append(k + "_FALSE")
            elif row.dtype.kind in "if":
                valid = bool(np.isfinite(row).all())
                flags[k] = "FINITE" if valid else "NONFINITE"
                if not valid:
                    problems.append(k + "_NONFINITE")
            else:
                flags[k] = "OTHER"
        result.append(
            {
                "lane": i,
                "flags": flags,
                "first_observed_problem": problems[0] if problems else "NONE_OBSERVED",
                "all_observed_problems": problems,
                "interpretation": "buffer ordering only; NOT proof of exclusive root cause",
            }
        )
    return result


def observe_optimizer(stage, *, read_tensor) -> dict:
    """Called AFTER the original native TimeOpt has returned."""
    optimizer = getattr(stage, "optimizer", None)
    if optimizer is None:
        return {"status": "NO_NATIVE_OPTIMIZER"}
    solver = getattr(optimizer, "_solver", None)
    tensors = {
        "input_valid": getattr(optimizer, "_input_valid", None),
        "solver_seed_feasible": getattr(solver, "feasible", None),
        "solver_numerical_ok": getattr(solver, "numerical_ok", None),
        "solver_output_timing": getattr(solver, "output", None),
        "solver_uniform_duration": getattr(solver, "uniform_duration", None),
        "solver_seed_bounds": getattr(solver, "seed_bounds", None),
        "solver_output_bounds": getattr(solver, "output_bounds", None),
        "finish_optimization_ok": getattr(optimizer, "_optimization_ok", None),
        "finished_squared_speed_controls": getattr(optimizer, "_timing", None),
        "finished_uniform_duration": getattr(optimizer, "_uniform_duration", None),
        "integrated_times": getattr(optimizer, "_times", None),
        "native_timeopt_success": getattr(optimizer, "_success", None),
        "finish_status": getattr(optimizer, "_status", None),
    }
    captured = {}
    summary = {}
    for name, tensor in tensors.items():
        if tensor is None:
            summary[name] = {"status": "UNAVAILABLE"}
            continue
        try:
            array = np.asarray(read_tensor(tensor))
            captured[name] = array
            summary[name] = numeric_summary(array)
        except Exception as exc:
            summary[name] = {"status": "CAPTURE_ERROR", "reason": repr(exc)}
    per_lane = per_lane_diagnosis(captured)
    # Persist selected raw arrays in a separate compressed research archive.
    # Per-lane summaries alone are insufficient to reproduce a nonfinite
    # squared-speed control or invalid integrated-time pattern.
    keep = (
        "input_valid",
        "solver_seed_feasible",
        "solver_numerical_ok",
        "solver_output_timing",
        "solver_uniform_duration",
        "solver_seed_bounds",
        "solver_output_bounds",
        "finish_optimization_ok",
        "finished_squared_speed_controls",
        "finished_uniform_duration",
        "integrated_times",
        "native_timeopt_success",
        "finish_status",
    )
    return {
        "status": "OBSERVED_AFTER_NATIVE_FINALIZATION",
        "summary": summary,
        "per_lane": per_lane,
        "raw_buffers": {k: captured[k].tolist() for k in keep if k in captured},
        "limitations": "observational post-run buffers; CUDA graph, same-lane overwrite, and absent pre-integration state may limit causal attribution",
    }


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_trace(folder: Path, tag: str, info: dict) -> dict:
    """Seal raw numeric CUDA buffers and metadata as immutable research evidence."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if "/" in tag or ".." in tag or not tag:
        raise ValueError("invalid native trace tag")
    path = folder / (tag + "_timeopt_phase_trace.json")
    metadata = {k: v for k, v in info.items() if k != "raw_buffers"}
    raw = info.get("raw_buffers", {})
    archive = folder / (tag + "_timeopt_phase_buffers.npz")
    if raw:
        arrays = {k: np.asarray(v) for k, v in raw.items()}
        if archive.exists():
            with np.load(archive) as saved:
                if set(saved.files) != set(arrays) or any(
                    not np.array_equal(saved[key], value, equal_nan=True)
                    for key, value in arrays.items()
                ):
                    raise ValueError("previous numeric buffer snapshot changed")
        else:
            np.savez_compressed(archive, **arrays)
        metadata["raw_buffers_sha256"] = _sha(archive)
        metadata["raw_buffers_file"] = str(archive)
    payload = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != payload:
        raise ValueError("phase trace cannot silently be overwritten")
    if not path.exists():
        path.write_text(payload)
    return {
        "timeopt_phase_trace_file": str(path),
        "timeopt_phase_trace_sha256": _sha(path),
        "timeopt_phase_buffers_sha256": metadata.get("raw_buffers_sha256"),
        "stage_observed": info.get("status"),
        "first_observed_issues": [
            x["first_observed_problem"] for x in info.get("per_lane", [])
        ],
    }
