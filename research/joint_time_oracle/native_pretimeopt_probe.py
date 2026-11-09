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

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def tensor_cpu(x):
    if x is None:
        return None
    return x.detach().cpu().numpy().copy()


class NativePreTimeOptProbe:
    def __init__(self, planner, folder: Path, tag: str, *, post_samples: int = 0):
        self.solver = planner.trajopt_solver
        self.folder = Path(folder)
        self.tag = tag
        self.original = None
        self.attempts = []
        self._invocations = 0
        if post_samples and (post_samples < 33 or post_samples > 2049):
            raise ValueError("post sample count must be 0 or 33..2049")
        self.post_samples = post_samples
        self._metric_owner = None
        self._metric_method_original = None
        self._metric_captures = []

    def _capture_metrics(self, rollout, state, *args, **kwargs):
        """Record final constraint residuals and return original metrics unchanged."""
        metrics = self._metric_method_original(state, *args, **kwargs)
        captured = {"constraint_groups": [], "convergence": []}
        collections = (
            ("constraints", metrics.costs_and_constraints.constraints),
            (
                "hybrid_costs_constraints",
                metrics.costs_and_constraints.hybrid_costs_constraints,
            ),
        )
        for group_name, collection in collections:
            for name, values in zip(collection.names, collection.values):
                if values.ndim == 0:
                    maxima = minima = values.reshape(1)
                else:
                    lanes = values.reshape(values.shape[0], -1)
                    maxima = lanes.max(dim=1).values
                    minima = lanes.min(dim=1).values
                captured["constraint_groups"].append(
                    {
                        "group": group_name,
                        "name": name,
                        "max_by_lane": tensor_cpu(maxima).reshape(-1).tolist(),
                        "min_by_lane": tensor_cpu(minima).reshape(-1).tolist(),
                    }
                )
        for name, values in zip(metrics.convergence.names, metrics.convergence.values):
            captured["convergence"].append(
                {"name": name, "by_lane": tensor_cpu(values).reshape(-1).tolist()}
            )
        self._metric_captures.append(captured)
        return metrics

    def __enter__(self):
        if self.original is not None:
            raise RuntimeError("native pre-TimeOpt probe already installed")
        self.folder.mkdir(parents=True, exist_ok=True)
        self.original = self.solver._finalize_result
        probe = self

        def wrapped(
            solver, current_state, num_seeds, return_seeds, seed_valid_mask=None
        ):
            index = probe._invocations
            probe._invocations += 1
            tag = f"{probe.tag}_pretimeopt_{index:03d}"
            metric_capture_start = len(probe._metric_captures)
            record = {
                "tag": tag,
                "stage": "before_TimeOpt",
                "state": "UNAVAILABLE",
                "timing_instrumented": True,
                "index": index,
            }
            try:
                native = solver._result_processor.get_interpolation_input()
                data = {}
                for name in (
                    "position",
                    "velocity",
                    "acceleration",
                    "jerk",
                    "knot",
                    "knot_dt",
                    "dt",
                ):
                    value = getattr(native, name, None)
                    if value is not None:
                        data[name] = tensor_cpu(value)
                if not all(x in data for x in ("position", "knot", "knot_dt")):
                    raise RuntimeError("incomplete geometric interpolation input")
                path = probe.folder / (tag + ".npz")
                np.savez_compressed(path, **data)
                record.update(
                    {
                        "state": "PRETIMEOPT_SNAPSHOT_SAVED",
                        "sha256": sha(path),
                        "geometry_file": str(path),
                        "candidate_count": int(data["position"].shape[0]),
                        "knot_shape": list(data["knot"].shape),
                    }
                )
                # Only observed source masks, not inferred TimeOpt outcomes.
                snapshot = getattr(solver._result_processor, "_snapshot", None)
                if snapshot is not None:
                    for field in ("success", "path_valid", "seed_cost"):
                        if getattr(snapshot, field, None) is not None:
                            values = tensor_cpu(getattr(snapshot, field))
                            record["pre_" + field] = values.reshape(-1).tolist()
                if seed_valid_mask is not None:
                    record["seed_valid_mask"] = (
                        tensor_cpu(seed_valid_mask).reshape(-1).tolist()
                    )
            except Exception as exc:
                record["capture_error"] = repr(exc)
            probe.attempts.append(record)
            # ALWAYS run the unchanged original, regardless of capture failure.
            try:
                out = probe.original(
                    current_state, num_seeds, return_seeds, seed_valid_mask
                )
                record["finalizer_returned"] = True
                if getattr(out, "success", None) is not None:
                    record["result_success"] = (
                        tensor_cpu(out.success).reshape(-1).tolist()
                    )
                stage = getattr(solver, "_timeopt", None)
                validator = getattr(stage, "validation_masks", None)
                if isinstance(validator, dict):
                    record["timeopt_validation_masks"] = {
                        key: tensor_cpu(value).reshape(-1).tolist()
                        for key, value in validator.items()
                    }
                record["validation_metric_captures"] = probe._metric_captures[
                    metric_capture_start:
                ]
                # Preserve EXACT native cubic geometry even when the
                # TimeOpt solution mask is all false. Never mark these
                # candidate curves as feasible on capture alone.
                optimizer = getattr(stage, "optimizer", None)
                if optimizer is not None:
                    pathbuf = {}
                    for key, attr in (
                        ("controls", "_controls"),
                        ("start_boundary", "_start_boundary"),
                        ("goal", "_goal"),
                        ("stationary", "_stationary"),
                        ("success", "_success"),
                        ("timing", "_timing"),
                        ("integrated_times", "_times"),
                    ):
                        value = getattr(optimizer, attr, None)
                        if value is not None:
                            pathbuf[key] = tensor_cpu(value)
                    if all(
                        key in pathbuf for key in ("controls", "start_boundary", "goal")
                    ):
                        path = probe.folder / (tag + "_native_geometry.npz")
                        np.savez_compressed(path, **pathbuf)
                        record["native_cubic_geometry_sha256"] = sha(path)
                        record["native_cubic_geometry_file"] = str(path)
                        record["native_geometry_lane_count"] = int(
                            pathbuf["controls"].shape[0]
                        )
                if probe.post_samples and optimizer is not None:
                    reset = None
                    try:
                        # Diagnostic only; sampling may fail for paths that
                        # TimeOpt rejected. Never modify the native return.
                        import torch

                        durations = tensor_cpu(getattr(optimizer, "_times"))
                        durations = np.asarray(durations, dtype=float)
                        durations = (
                            durations[:, -1] if durations.ndim == 2 else durations
                        )
                        valid = np.isfinite(durations) & (durations > 1e-6)
                        # Invalid lanes receive a dummy 1-second sample grid
                        # and are explicitly NOT physically interpretable.
                        stable = np.where(valid, durations, 1.0)
                        device = getattr(optimizer, "_controls").device
                        unit = torch.linspace(0, 1, probe.post_samples, device=device)
                        grid = (
                            torch.as_tensor(stable, device=device, dtype=unit.dtype)[
                                :, None
                            ]
                            * unit[None, :]
                        )
                        # TimeOpt's sampler captures a graph for its first
                        # sample count. The diagnostic requests its own count
                        # after planning, so clear that graph before sampling.
                        reset = getattr(optimizer, "reset_cuda_graph", None)
                        if callable(reset):
                            reset()
                        state = optimizer.sample(grid.contiguous())
                        archive = {
                            "time": tensor_cpu(grid),
                            "q": tensor_cpu(state.position),
                            "qd": tensor_cpu(state.velocity),
                            "qdd": tensor_cpu(state.acceleration),
                            "jerk": tensor_cpu(state.jerk),
                            "valid_duration_mask": valid,
                        }
                        file = probe.folder / (tag + "_post_timeopt_states.npz")
                        np.savez_compressed(file, **archive)
                        record["post_timeopt_state_file"] = str(file)
                        record["post_timeopt_state_sha256"] = sha(file)
                        record["post_timeopt_valid_duration_mask"] = valid.tolist()
                        record["post_timeopt_samples"] = probe.post_samples
                    except Exception as exc:
                        record["post_timeopt_sampling_error"] = repr(exc)
                    finally:
                        # Restore the sampler's original lazy-graph state.
                        # A planner can finalize multiple internal candidates
                        # in one request; retaining N=post_samples would make
                        # the next native finalizer fail its own sampling call.
                        if callable(reset):
                            reset()
                return out
            except Exception as exc:
                record["finalizer_returned"] = False
                record["finalizer_exception"] = repr(exc)
                raise
            finally:
                probe.flush()

        self.solver._finalize_result = MethodType(wrapped, self.solver)
        timeopt = getattr(self.solver, "_timeopt", None)
        rollout = getattr(timeopt, "rollout", None)
        if rollout is not None and callable(
            getattr(rollout, "compute_metrics_from_state", None)
        ):
            self._metric_owner = rollout
            self._metric_method_original = rollout.compute_metrics_from_state
            probe = self

            def metric_wrapper(rollout, state, *args, **kwargs):
                return probe._capture_metrics(rollout, state, *args, **kwargs)

            self._metric_owner.compute_metrics_from_state = MethodType(
                metric_wrapper, self._metric_owner
            )
        return self

    def flush(self):
        dest = self.folder / (self.tag + "_pretimeopt_inventory.json")
        dest.write_text(json.dumps(self.attempts, indent=2) + "\n")

    def __exit__(self, exc_type, exc, tb):
        self.solver._finalize_result = self.original
        self.original = None
        if self._metric_owner is not None:
            self._metric_owner.compute_metrics_from_state = self._metric_method_original
        self._metric_owner = None
        self._metric_method_original = None
        self.flush()
        return False
