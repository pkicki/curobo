# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared contracts for the TimeOpt-guided planning study.

Everything here is pure NumPy so that both the CUDA environment (cuRobo, K107)
and the independent Pinocchio environment can import it.

Payload convention (identical in native cuRobo, K107 and Pinocchio):
  * rigid body attached to the ``attached_object`` link, which is a FIXED child
    of ``panda_hand`` with identity transform (``franka.yml``);
  * ``com_m`` is expressed in the ``panda_hand`` frame;
  * rotational inertia about the payload COM is a solid sphere,
    ``2/5 * mass * radius**2`` on the diagonal;
  * the same sphere (center ``com_m``, ``radius_m``) is the payload collision proxy.
  * the payload REPLACES the default 0.01 kg ``attached_object`` placeholder.
Actuator limits are the ORIGINAL ``franka.yml`` effort limits; nothing is derated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research/timeopt_guided/results"
JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
# Native cuRobo plans dynamic limits at 95% ("Scaled95"); K107 uses the same scale.
LIMIT_SCALE = 0.95
# Name, scene file (None = no obstacles), as in the earlier geometry study.
SCENES = {
    "free": None,
    "table": "collision_test.yml",
    "cluttered": "collision_primitives_3d.yml",
}


def curobo_scene(name: str):
    """Scene for cuRobo with sphere obstacles given as zero-length capsules.

    cuRobo's ``SceneData.load_from_scene_cfg`` (curobo/_src/geom/data/data_scene.py)
    loads cuboid/capsule/cylinder/mesh/voxel obstacles but silently SKIPS
    ``scene_cfg.sphere`` (verified at 299d7a6: a planner starting inside a sphere
    obstacle reports success). A capsule with coincident endpoints is exactly a
    sphere in cuRobo's capsule kernel, so this conversion is exact. Returns None
    for the obstacle-free scene and a dict otherwise.
    """
    file = SCENES[name]
    if file is None:
        return None
    import copy

    import yaml

    data = copy.deepcopy(yaml.safe_load((ROOT / "curobo/content/configs/scene" / file).read_text()))
    spheres = data.pop("sphere", None) or {}
    caps = data.setdefault("capsule", {}) or {}
    for key, sph in spheres.items():
        center = list(sph.get("position", (sph.get("pose") or [0, 0, 0])[:3]))
        caps[f"{key}__as_capsule"] = {"radius": float(sph["radius"]), "base": [0.0, 0.0, 0.0],
                                      "tip": [0.0, 0.0, 0.0], "pose": center + [1.0, 0.0, 0.0, 0.0]}
    data["capsule"] = caps
    return data


@dataclass(frozen=True)
class Payload:
    """Rigid tool payload on ``attached_object`` (see module docstring)."""

    mass_kg: float
    com_m: tuple[float, float, float]
    radius_m: float = 0.06
    label: str = ""

    def validate(self) -> "Payload":
        com = np.asarray(self.com_m, dtype=float)
        if not (0.0 <= self.mass_kg <= 10.0) or com.shape != (3,):
            raise ValueError(f"invalid payload {self}")
        if not np.isfinite(com).all() or np.any(np.abs(com) > 0.4):
            raise ValueError(f"payload COM outside research envelope: {self}")
        if not 0.01 <= self.radius_m <= 0.2:
            raise ValueError(f"payload radius outside research envelope: {self}")
        return self

    @property
    def inertia_diag(self) -> np.ndarray:
        """Solid-sphere principal inertia about the payload COM [kg m^2]."""
        return np.full(3, 0.4 * self.mass_kg * self.radius_m**2)

    def to_json(self) -> dict:
        d = asdict(self)
        d["com_m"] = list(self.com_m)
        return d

    @staticmethod
    def from_json(d: dict) -> "Payload":
        return Payload(
            float(d["mass_kg"]), tuple(float(x) for x in d["com_m"]),
            float(d.get("radius_m", 0.06)), str(d.get("label", "")),
        ).validate()

    def sha(self) -> str:
        return canonical_hash(self.to_json())


# Registered payload grid. Panda's rated payload is 3 kg; 4 kg and 5 kg are
# research-hypothetical loads (still NOMINAL actuator limits) used to find
# torque-active cases. COM offsets: on-axis (0,0,0.12) and lever-arm (0.08,0,0.16)
# which loads the 12 Nm wrist joints.
PAYLOAD_MASSES = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
PAYLOAD_COMS = {"axial": (0.0, 0.0, 0.12), "offset": (0.08, 0.0, 0.16)}


def payload_grid() -> list[Payload]:
    out = []
    for name, com in PAYLOAD_COMS.items():
        for m in PAYLOAD_MASSES:
            if m == 0.0 and name != "axial":
                continue  # zero payload is COM independent
            out.append(Payload(m, com, 0.06, f"m{m:g}_{name}").validate())
    return out


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=_default).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value).__name__}")


def write_json(path: Path, data, *, overwrite: bool = True) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=_default) + "\n")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def task_split(task_id: str, confirmation_fraction: float = 0.2) -> str:
    """Deterministic development/confirmation split by SOURCE task id."""
    h = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "confirmation" if h < confirmation_fraction else "development"


def quintic_knots(n_controls: int) -> np.ndarray:
    """Clamped uniform knot vector of degree 5 used by every quintic candidate."""
    return np.r_[np.zeros(5), np.linspace(0, 1, n_controls - 4), np.ones(5)]


def rest_controls(q0: np.ndarray, q1: np.ndarray, n: int = 22) -> np.ndarray:
    """Straight joint-space line with rest endpoints (3 repeated controls per end)."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    z = np.linspace(0, 1, n)
    smooth = z**3 * (10 - 15 * z + 6 * z * z)
    c = q0[None] + smooth[:, None] * (q1 - q0)[None]
    c[:3] = q0
    c[-3:] = q1
    return c


def enforce_rest(controls: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    c = np.asarray(controls, float).copy()
    c[:3] = q0
    c[-3:] = q1
    return c
