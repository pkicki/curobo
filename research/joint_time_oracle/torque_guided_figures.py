# SPDX-License-Identifier: Apache-2.0
"""Paper-ready plots reconstructed only from frozen robot/path/torque evidence.

These are diagnostic plots, not rendered hypothetical robot scenes. Model-
aligned Pinocchio provides the end-effector locations and RNEA torque.
The original fixed-geometry LP obstruction is visualized independently
from any subsequently found W trajectory. Missing W output is labeled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin

from .cohort_geometry import QuinticGeometry
from .cohort_retime import ROOT
from .geometry_feasibility_discover import source_native, source_tasks
from .geometry_feasibility_core import sha256_file
from .torque_dual_guidance import AlignedPointDynamics, solve_point_lp
from .torque_guided_rescue import ROOTS, SCENARIO, OUTPUT
from .torque_active import read_profile


def ee_points(q:np.ndarray,dynamics:AlignedPointDynamics)->np.ndarray:
    model=dynamics.model
    data=dynamics.data
    frame=model.getFrameId("right_gripper",pin.BODY)
    xyz=np.zeros((len(q),3))
    for k,p in enumerate(q):
        dynamics.q[dynamics.qi]=p
        pin.framesForwardKinematics(model,data,dynamics.q)
        xyz[k]=data.oMf[frame].translation
    return xyz


def witness_torque(w:dict,dynamics:AlignedPointDynamics)->np.ndarray:
    out=np.empty_like(w["q"])
    v=np.zeros(dynamics.model.nv);a=np.zeros(dynamics.model.nv)
    for i in range(len(out)):
        dynamics.q[dynamics.qi]=w["q"][i]
        v.fill(0);a.fill(0)
        v[dynamics.vi]=w["qd"][i]
        a[dynamics.vi]=w["qdd"][i]
        out[i]=pin.rnea(dynamics.model,dynamics.data,dynamics.q,v,a)[dynamics.vi]
    return out


def make_figure(case:str,arm_name:str,output:Path,figroot:Path):
    if case not in ROOTS:
        raise ValueError("unknown frozen development source")
    src_name,disc_name=ROOTS[case]
    source=ROOT/"research/joint_time_oracle/results"/src_name
    discovery=ROOT/"research/joint_time_oracle/results"/disc_name
    tasks,records=source_tasks(source)
    task=next(t for t in tasks if t["case_id"]==case)
    profile=read_profile(discovery/SCENARIO/case/"torque_profile.json")
    bounds=np.stack((profile["effort_lower"],profile["effort_upper"]))
    native=source_native(source,task,records[case])
    dynamics=AlignedPointDynamics()
    s=np.linspace(0,1,257)
    original=np.asarray(native.evaluate_trimmed(s)[0])
    folder=output/arm_name/case
    reportfile=folder/"search_result.json"
    if not reportfile.exists():
        raise FileNotFoundError("controlled arm not executed")
    res=json.loads(reportfile.read_text())["report"]
    controls=np.asarray(np.load(folder/"best_controls.npz")["controls"])
    geom=QuinticGeometry(controls)
    alternative=np.asarray(geom.evaluate(s)[0])
    source_xyz=ee_points(original,dynamics)
    alt_xyz=ee_points(alternative,dynamics)
    original_lp=json.loads((discovery/SCENARIO/case/"refined_original_path_lp.json").read_text())
    progress=float(original_lp["worst_progress"])
    class Original:
        def evaluate(self,x):
            return native.evaluate_trimmed(x)
    a,m,g=dynamics.coefficients(Original(),progress)
    lp=solve_point_lp(a,m,g,bounds)
    if lp.status!="SOLVED":
        raise ValueError("LP witness unable to reconstruct")
    torque=g+m*lp.x+a*lp.y
    folder_out=figroot/case/arm_name
    folder_out.mkdir(parents=True,exist_ok=True)
    fig=plt.figure(figsize=(10,7))
    ax=fig.add_subplot(111,projection="3d")
    ax.plot(*source_xyz.T,label="Nominal native geometry")
    ax.plot(*alt_xyz.T,label=f"{arm_name} searched geometry")
    ax.scatter(*source_xyz[0],s=60,label="Start")
    ax.scatter(*source_xyz[-1],s=60,label="Goal")
    ax.set_xlabel("World x (m)");ax.set_ylabel("World y (m)")
    ax.set_zlabel("World z (m)")
    ax.set_title(f"{case}: actual Pinocchio right-gripper paths")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder_out/"end_effector_paths.png",dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5))
    joints=np.arange(1,8)
    ax.bar(joints-.18,torque,width=.36,label="Original path: LP-optimal instantaneous torque")
    ax.scatter(joints,bounds[0],label="Torque lower",marker="_",s=80)
    ax.scatter(joints,bounds[1],label="Torque upper",marker="_",s=80)
    ax.set_xticks(joints)
    ax.set_xlabel("Panda joint")
    ax.set_ylabel("Joint torque (Nm)")
    ax.set_title(f"Original path point s={progress:.4f}; required LP expansion={lp.expansion:.4f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder_out/"original_path_torque_lp.png",dpi=160)
    plt.close(fig)
    output_info={"case":case,"arm":arm_name,"scenario":SCENARIO,
                 "original_lp_expansion":float(lp.expansion),
                 "original_progress":progress,
                 "source_original_geometry_sha256":sha256_file(source/case/"a10_buffers.npz"),
                 "model":"aligned independent Pinocchio",
                 "alternative_controls_sha256":sha256_file(folder/"best_controls.npz"),
                 "witness_torque_plot":"NOT_AVAILABLE"}
    if (folder/"w_dense.npz").exists():
        with np.load(folder/"w_dense.npz") as z:
            witness={k:np.asarray(z[k]) for k in z.files}
        tau=witness_torque(witness,dynamics)
        fig,axes=plt.subplots(7,1,figsize=(10,15),sharex=True)
        for joint,ax in enumerate(axes):
            ax.plot(witness["time"],tau[:,joint],label="Independent Pinocchio RNEA")
            ax.axhline(bounds[0,joint],linestyle="--",label="lower" if joint==0 else None)
            ax.axhline(bounds[1,joint],linestyle="--",label="upper" if joint==0 else None)
            ax.set_ylabel(f"J{joint+1} Nm")
        axes[-1].set_xlabel("Physical execution time (s)")
        axes[0].legend()
        fig.suptitle(case+": candidate W torque profile (independent verification required)")
        fig.tight_layout()
        fig.savefig(folder_out/"witness_torques.png",dpi=150)
        plt.close(fig)
        output_info["witness_torque_plot"]="PRELIMINARY — must match independent sampled/adaptive receipt"
    (folder_out/"figure_provenance.json").write_text(json.dumps(output_info,indent=2)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUTPUT)
    p.add_argument("--arm",choices=("dual","random_local","random_global"),required=True)
    p.add_argument("--case",choices=tuple(ROOTS),required=True)
    p.add_argument("--figure-root",type=Path,default=ROOT/
                   "reports/joint_time_oracle/torque_lp_guided_rescue/figures")
    a=p.parse_args()
    make_figure(a.case,a.arm,a.root,a.figure_root)


if __name__=="__main__":
    main()
