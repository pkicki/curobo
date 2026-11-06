# SPDX-License-Identifier: Apache-2.0
"""Compare cuRoboV2-Fast TimeOpt with offline joint path+final-time optimization.

Run from repo root:
 python -m research.joint_time_oracle.run_study --output /tmp/joint_ceiling
CUDA robot, RNEA, collision, and a warmed planner are required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import torch

from .oracle import Boundary, JointTimeOracle, OracleConfig, normalized_worst


def as_numpy(x):
    a = x.detach().cpu().numpy().astype(np.float64)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 2 and a.shape[0] == 1:
        a = a[0]
    return a


def read_plan(result):
    trajectory = result.get_interpolated_plan()
    if trajectory is None:
        raise RuntimeError("missing interpolated TimeOpt plan")
    q = as_numpy(trajectory.position)
    if q.ndim != 2:
        raise RuntimeError(f"invalid joint plan shape: {q.shape}")
    dt = trajectory.dt
    dt = float(dt.reshape(-1)[0].item()) if isinstance(dt,torch.Tensor) else float(dt)
    if dt <= 0:
        raise RuntimeError("invalid uniform retimed timestep")
    out = dict(q=q,dt=dt)
    for attr, name in (("velocity","qd"),("acceleration","qdd"),("jerk","jerk")):
        tensor = getattr(trajectory,attr)
        if tensor is None:
            raise RuntimeError(f"retimed plan missing {attr}")
        out[name] = as_numpy(tensor)
    return out


def audit_baseline(model, plan, limits):
    q = torch.as_tensor(plan["q"],device=model.device,dtype=torch.float32)
    qd = torch.as_tensor(plan["qd"],device=model.device,dtype=torch.float32)
    qdd = torch.as_tensor(plan["qdd"],device=model.device,dtype=torch.float32)
    jerk = torch.as_tensor(plan["jerk"],device=model.device,dtype=torch.float32)
    with torch.no_grad():
        tau = model.torque(q,qd,qdd)
        outputs={}
        for name,val in (("position",q),("velocity",qd),("acceleration",qdd),
                         ("jerk",jerk),("torque",tau)):
            bound = torch.as_tensor(getattr(limits,name),device=model.device,dtype=torch.float32)
            outputs[name] = float((-normalized_worst(val,bound,1.0).min()).clamp_min(0).item())
        outputs["collision"] = float(model.collision(q).clamp_min(0).max().item())
    return outputs


def plot_profiles(output, base, best, task_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    q0 = base["q"]
    q1 = np.asarray(best["positions"])
    x0 = np.linspace(0,1,len(q0))
    x1 = np.linspace(0,1,len(q1))
    fig, ax = plt.subplots(figsize=(11,5))
    for joint in range(q0.shape[1]):
        line, = ax.plot(x0,q0[:,joint],alpha=.6,lw=1)
        ax.plot(x1,q1[:,joint],color=line.get_color(),ls="--",alpha=.9)
    ax.set(xlabel="Normalized elapsed time",ylabel="Joint angle [rad]",
           title=f"{task_id}: solid = cuRoboV2-Fast, dashed = joint free-time local solution")
    fig.tight_layout()
    fig.savefig(output/f"{task_id}.png",dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--robot",default="franka.yml")
    parser.add_argument("--scene",default="collision_table.yml")
    parser.add_argument("--tasks",type=Path,help='JSON list of {"start":[...],"goal":[...]}')
    parser.add_argument("--number",type=int,default=8)
    parser.add_argument("--seed",type=int,default=14)
    parser.add_argument("--controls",type=int,default=18)
    parser.add_argument("--attempts",type=int,default=4)
    parser.add_argument("--maxiter",type=int,default=250)
    parser.add_argument("--limit-scale",type=float,default=.95)
    parser.add_argument("--max-planner-attempts",type=int,default=5)
    parser.add_argument("--no-plot",action="store_true")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for native cuRobo benchmark")
    if args.number < 1 or args.max_planner_attempts < 1:
        parser.error("task count and planner attempts must be positive")
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import JointState
    from .curobo_adapter import CuroboOracleModel

    args.output.mkdir(parents=True,exist_ok=True)
    planner = MotionPlanner(MotionPlannerCfg.create(
        robot=args.robot,scene_model=args.scene,
        graph_planner_algorithm="rrt",enable_timeopt=True,use_cuda_graph=True))
    planner.warmup(enable_graph=True,num_warmup_iterations=5)
    model = CuroboOracleModel(planner)
    limits = model.limits()
    dof = len(model.joint_names)
    if args.tasks is not None:
        tasks = json.loads(args.tasks.read_text())
        if not isinstance(tasks,list):
            raise ValueError("tasks file must be a JSON list")
    else:
        rng = np.random.default_rng(args.seed)
        home = as_numpy(planner.default_joint_state.position)
        lo,hi = limits.position
        tasks = []
        for _ in range(args.number):
            goal = np.clip(home+rng.uniform(-.6,.6,size=dof),lo+.12,hi-.12)
            tasks.append(dict(start=home.tolist(),goal=goal.tolist()))

    manifest = dict(method="offline quintic spline joint-geometry/final-time SLSQP",
                    fork="pkicki/curobo",local_only=True,global_lower_bound=None,
                    constraints="position/velocity/acceleration/jerk/RNEA torque/world/self collision",
                    robot=args.robot,scene=args.scene,joint_names=model.joint_names,
                    gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__),
                    cuda=str(torch.version.cuda),python=platform.python_version(),
                    parameters={k:str(v) for k,v in vars(args).items()})
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    rows=[]
    for i,task in enumerate(tasks):
        name=f"task_{i:04d}"
        row=dict(task_id=name,start=task["start"],goal=task["goal"],status="UNATTEMPTED")
        try:
            q0 = np.asarray(task["start"],dtype=np.float64)
            q1 = np.asarray(task["goal"],dtype=np.float64)
            if q0.shape != (dof,) or q1.shape != (dof,):
                raise ValueError("start and goal must match active DoF")
            start = JointState.from_position(torch.as_tensor(q0,device=model.device,dtype=torch.float32)[None],
                                             joint_names=model.joint_names)
            goal = JointState.from_position(torch.as_tensor(q1,device=model.device,dtype=torch.float32)[None],
                                            joint_names=model.joint_names)
            torch.cuda.synchronize()
            stamp=time.perf_counter()
            result=planner.plan_cspace(goal,start,max_attempts=args.max_planner_attempts)
            torch.cuda.synchronize()
            row["planner_wall_s"]=time.perf_counter()-stamp
            if result is None or not bool(result.success.any()):
                row["status"]="PLANNER_FAILED"
            else:
                base=read_plan(result)
                t_fast=(len(base["q"])-1)*base["dt"]
                if result.timeopt_duration is not None:
                    t_fast=float(result.timeopt_duration.reshape(-1)[0].item())
                row["fast_duration_s"]=t_fast
                row["fast_validation"]=audit_baseline(model,base,limits)
                row["baseline_sampled_feasible"]=max(row["fast_validation"].values())<=.002
                row["fast_samples"]=len(base["q"])
                row["planner_reported_total_time_s"]=float(result.total_time)
                row["planner_reported_solve_time_s"]=float(result.solve_time)
                bc=Boundary(base["q"][0],base["q"][-1],base["qd"][0],base["qd"][-1],
                            base["qdd"][0],base["qdd"][-1])
                ocfg=OracleConfig(controls=args.controls,attempts=args.attempts,
                                  maxiter=args.maxiter,limit_scale=args.limit_scale,
                                  rng_seed=args.seed+i)
                solver=JointTimeOracle(limits,bc,model.torque,model.collision,
                                      config=ocfg,device=str(model.device))
                torch.cuda.synchronize()
                stamp=time.perf_counter()
                local=solver.solve(base["q"],t_fast)
                torch.cuda.synchronize()
                row["oracle_wall_s"]=time.perf_counter()-stamp
                row["oracle_status"]=local["status"]
                row["trials"]=local["trials"]
                row["best"]=local["best"]
                if local["best"] is None:
                    row["status"]="NO_VERIFIED_JOINT_SOLUTION"
                elif not row["baseline_sampled_feasible"]:
                    row["status"]="BASELINE_AUDIT_FAILED"
                else:
                    row["status"]="VERIFIED_LOCAL_SOLUTION"
                    row["improvement_percent"]=100*(t_fast-local["best"]["duration"])/t_fast
                    if not args.no_plot:
                        plot_profiles(args.output,base,local["best"],name)
            print(f'{name}: {row["status"]}',flush=True)
        except Exception as exc:
            row["status"]="ERROR"
            row["error"]=repr(exc)
            row["traceback"]=traceback.format_exc()
            print(f"{name}: ERROR {exc}",flush=True)
        rows.append(row)
        (args.output/"per_task.json").write_text(json.dumps(rows,indent=2))
    gains=[r["improvement_percent"] for r in rows if r["status"]=="VERIFIED_LOCAL_SOLUTION"]
    counts={status:sum(r["status"]==status for r in rows)
            for status in sorted({r["status"] for r in rows})}
    bootstrap_ci=None
    if len(gains)>=2:
        rng_ci=np.random.default_rng(args.seed+2001)
        arr=np.asarray(gains,dtype=np.float64)
        medians=np.median(rng_ci.choice(arr,size=(2000,len(arr)),replace=True),axis=1)
        bootstrap_ci=[float(x) for x in np.quantile(medians,[.025,.975])]
    summary=dict(tasks_attempted=len(rows),
                 baseline_succeeded=sum("fast_duration_s" in r for r in rows),
                 oracle_verified=len(gains),status_counts=counts,
                 median_improvement_percent=float(np.median(gains)) if gains else None,
                 median_improvement_bootstrap95_pct=bootstrap_ci,
                 mean_improvement_percent=float(np.mean(gains)) if gains else None,
                 negative_improvements=sum(v<0 for v in gains),
                 no_global_lower_bound=True,
                 interpretation="A shorter sampled-feasible trajectory establishes a lower bound on possible time savings, not their global ceiling. Bootstrap CI is over the verified subset only; selection bias remains.")
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    planner.destroy()


if __name__=="__main__":
    main()
