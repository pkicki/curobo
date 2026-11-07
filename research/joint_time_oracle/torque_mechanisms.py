# SPDX-License-Identifier: Apache-2.0
"""Independent Pinocchio torque decomposition for frozen B versus D geometry.

Explains which torque terms differ; does NOT establish a causal attribution.
Outputs are diagnostics, not continuous-time feasibility certificates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .torque_active import read_profile, torque_utilization

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "research/joint_time_oracle/results/torque_stress_final_20261107"
Rnea = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def decompose_rnea(
    positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray,
    rnea: Rnea,
) -> dict[str, np.ndarray]:
    """Separate M(q)qdd, C(q,qd)qd, and g(q) using RNEA linearity.

    rnea(q,v,a) returns the seven controlled-joint torques; external forces,
    friction and actuation saturation are NOT included. The zero-velocity
    inertia call removes velocity-dependent bias exactly for a rigid-body
    model. The velocity call isolates its velocity-dependent bias.
    """
    q, v, a = (np.asarray(t, dtype=np.float64)
               for t in (positions, velocities, accelerations))
    if q.ndim != 2 or q.shape != v.shape or q.shape != a.shape:
        raise ValueError("q, velocity and acceleration must be matching [N,dof] arrays")
    if not all(np.isfinite(x).all() for x in (q, v, a)):
        raise ValueError("nonfinite robot states")
    n, dof = q.shape
    zero = np.zeros(dof, dtype=np.float64)
    gravity = np.empty_like(q)
    inertia = np.empty_like(q)
    velocity_bias = np.empty_like(q)
    total = np.empty_like(q)
    for k in range(n):
        g = np.asarray(rnea(q[k], zero, zero), dtype=np.float64)
        mi = np.asarray(rnea(q[k], zero, a[k]), dtype=np.float64)
        cv = np.asarray(rnea(q[k], v[k], zero), dtype=np.float64)
        tau = np.asarray(rnea(q[k], v[k], a[k]), dtype=np.float64)
        if any(x.shape != (dof,) for x in (g, mi, cv, tau)):
            raise ValueError("RNEA callback returned incorrect torque shape")
        gravity[k], inertia[k], velocity_bias[k], total[k] = g, mi-g, cv-g, tau
    residual = total-(inertia+velocity_bias+gravity)
    if not np.isfinite(residual).all():
        raise ValueError("nonfinite inverse-dynamics decomposition")
    return dict(total=total, inertia=inertia, velocity_bias=velocity_bias,
                gravity=gravity, identity_residual=residual)


def common_physical_times(trajectory: dict, *, count: int=501) -> dict:
    """Sample both solutions by *normalized physical time*, not path progress."""
    t = np.asarray(trajectory["time"], dtype=np.float64)
    if t.ndim != 1 or len(t) < 3 or np.any(np.diff(t) <= 0):
        raise ValueError("trajectory timestamps must be strictly increasing")
    if count < 3:
        raise ValueError("need at least three samples")
    u = np.linspace(0., 1., count)
    t_query = t[0]+u*(t[-1]-t[0])
    result = {"normalized_time": u, "time": t_query}
    for name in ("q", "qd", "qdd"):
        x = np.asarray(trajectory[name], dtype=np.float64)
        if x.ndim != 2 or x.shape[0] != len(t):
            raise ValueError("invalid trajectory array " + name)
        result[name] = np.column_stack([
            np.interp(t_query, t, x[:, j]) for j in range(x.shape[1])
        ])
    return result


def metrics(components: dict[str,np.ndarray], bounds: np.ndarray,
            q: np.ndarray, duration: float) -> dict:
    total = components["total"]
    utilization = torque_utilization(total, bounds)
    k, j = np.unravel_index(int(np.argmax(utilization)),utilization.shape)
    family_names = ("inertia", "velocity_bias", "gravity")
    lim = float(bounds[1, j] if total[k,j]>=0 else -bounds[0,j])
    terms = {name: float(components[name][k,j]) for name in family_names}
    return {
        "duration_s": float(duration),
        "max_torque_utilization": float(utilization[k,j]),
        "max_torque_joint": int(j+1),
        "max_torque_normalized_time": float(k/(len(utilization)-1)),
        "max_torque_Nm": float(total[k,j]),
        "torque_limit_magnitude_Nm": lim,
        "peak_torque_components_Nm": terms,
        "peak_signed_component_utilization": {key:val/lim for key,val in terms.items()},
        "rms_components_Nm": {name:float(np.sqrt(np.mean(components[name]**2)))
                              for name in (*family_names, "total")},
        "mean_max_joint_utilization": float(np.mean(np.max(utilization,axis=1))),
        "fraction_of_samples_above_85pct": float(np.mean(np.any(utilization>=.85,axis=1))),
        "fraction_of_samples_above_95pct": float(np.mean(np.any(utilization>=.95,axis=1))),
        "joint_space_path_length_rad":float(np.sum(np.linalg.norm(np.diff(q,axis=0),axis=1))),
        "rnea_identity_max_error_Nm":float(np.max(np.abs(components["identity_residual"]))),
    }


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_case(arm: Path,case: str, *, count: int=501) -> dict:
    """Do not overwrite historic trajectories or validation receipts."""
    folder=arm/case
    profile_file=folder/"torque_profile.json"
    profile=read_profile(profile_file)
    if profile["source_case_id"]!=case:
        raise ValueError("torque profile/source case mismatch")
    bounds=np.array([profile["effort_lower"],profile["effort_upper"]])
    from .cohort_retime import dynamics
    model,data,state,qi,vi=dynamics()
    zero=np.zeros(model.nv)
    def rnea(q: np.ndarray,v: np.ndarray,a: np.ndarray) -> np.ndarray:
        state[qi]=q
        zero[vi]=v
        accel=np.zeros(model.nv)
        accel[vi]=a
        import pinocchio as pin
        return pin.rnea(model,data,state,zero,accel)[vi].copy()
    reports={}
    npz={}
    files={"torque_profile":sha(profile_file)}
    for method in ("B","D"):
        path=folder/(method.lower()+"_dense.npz")
        if not path.exists():
            reports[method]={"status":"MISSING_TRAJECTORY"}
            continue
        files[method+"_dense"]=sha(path)
        with np.load(path) as z:
            traj={key:z[key] for key in ("time","q","qd","qdd")}
        sampled=common_physical_times(traj,count=count)
        result=decompose_rnea(sampled["q"],sampled["qd"],sampled["qdd"],rnea)
        m=metrics(result,bounds,sampled["q"],traj["time"][-1]-traj["time"][0])
        if m["rnea_identity_max_error_Nm"]>1e-7:
            raise ValueError("Pinocchio decomposition identity failed")
        reports[method]={"status":"DECOMPOSED","metrics":m}
        for k,x in (*sampled.items(),*result.items()):
            npz[method.lower()+"_"+k]=x
    contrast={}
    if all(reports.get(m,{}).get("status")=="DECOMPOSED" for m in ("B","D")):
        b,d=reports["B"]["metrics"],reports["D"]["metrics"]
        contrast={
            "B_to_D_improvement_pct":100*(b["duration_s"]-d["duration_s"])/b["duration_s"],
            "path_length_increase_pct":100*(d["joint_space_path_length_rad"]/
                                            b["joint_space_path_length_rad"]-1),
            "peak_torque_utilization_delta_D_minus_B":
                d["max_torque_utilization"]-b["max_torque_utilization"],
            "gravity_rms_delta_Nm":d["rms_components_Nm"]["gravity"]-
                                   b["rms_components_Nm"]["gravity"],
            "inertia_rms_delta_Nm":d["rms_components_Nm"]["inertia"]-
                                   b["rms_components_Nm"]["inertia"],
            "velocity_bias_rms_delta_Nm":d["rms_components_Nm"]["velocity_bias"]-
                                         b["rms_components_Nm"]["velocity_bias"],
        }
    out=arm/"mechanisms"/case
    out.mkdir(parents=True,exist_ok=True)
    if npz:
        np.savez_compressed(out/"torque_components.npz",**npz)
    doc={"case_id":case,"regime":profile["regime"],"profile_sha256":profile["profile_sha256"],
         "input_sha256":files,"sampling":"uniform normalized physical time; interpolation diagnostics",
         "physical_model":"aligned Pinocchio rigid-body RNEA; no payload or joint friction",
         "methods":reports,"contrast":contrast,
         "interpretation":"descriptive physical decomposition; not a causal explanation or a certificate"}
    (out/"analysis.json").write_text(json.dumps(doc,indent=2)+"\n")
    return doc


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--regime",required=True,choices=("moderate","severe"))
    parser.add_argument("--cases",help="comma-separated source case IDs; default all admitted")
    parser.add_argument("--samples",type=int,default=501)
    args=parser.parse_args()
    arm=args.root/args.regime
    tasks=json.loads((arm/"tasks.json").read_text())
    selected=set(args.cases.split(",")) if args.cases else None
    rows=[]
    for task in tasks:
        case=task["case_id"]
        if selected is not None and case not in selected:
            continue
        try:
            item=evaluate_case(arm,case,count=args.samples)
            rows.append({"case_id":case,**item["contrast"],"status":"COMPLETE" if item["contrast"] else "INCOMPLETE"})
        except Exception as exc:
            rows.append({"case_id":case,"status":"ERROR","reason":str(exc)})
    output={"regime":args.regime,"cases":rows}
    path=arm/"mechanisms"/"summary.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({"cases":len(rows),"complete":sum(r["status"]=="COMPLETE" for r in rows),
                      "output":str(path)},indent=2))


if __name__=="__main__":
    main()
