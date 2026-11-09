# SPDX-License-Identifier: Apache-2.0
"""Research adapter to the ACTUAL time_optimal_curobo K107 TOPP-RA+Newton.

Install/pin both repositories in one CUDA Python environment:
  PYTHONPATH=<time_optimal_curobo>/src:<time_optimal_curobo>/scripts:. \
    python -m research.joint_time_oracle.timeopt_k107_worker \
      --request /path/request.json --response /path/response.json

This imports benchmark_pdf_timing_polish_runtime.Request to invoke the
existing TOPP-RA seed + bounded primal-dual K107 polish and native
screen, rather than silently replacing it with TOPP-RA alone or IPOPT.
Native robot inertia comes from this repo's payload model. Strict
common contract: four families V/A/J/torque, torque rate disabled,
zero path-end speeds, free endpoint acceleration, exact quintic path.

Research-only code: preflight must verify model geometry/derivatives,
RNEA, limits, timing invariance, and independent S8193/S32768
physical+Scaled95. Default parameters are intentionally reference-
mode (not the optimized captured-graph latency configuration).
A returned FEASIBLE_SCREENED is NEVER independent certification.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time
import traceback

import numpy as np

from .timeopt_payload_model import Payload, native_robot
from .timeopt_oracle_contract import (
    CONSTRAINT_FAMILIES, ENDPOINT_CONTRACT, K107_ROUTE,
)
from .timeopt_payload_model import digest


def defaults()->SimpleNamespace:
    return SimpleNamespace(
        contract="free_acceleration", solver="primal_dual",
        polish_basis_backend="dense", grid=2049, screen_grid=8193,
        pdf_samples=8193, end_fraction=.005, end_density=16,
        layer_fraction=.05, layer_density=3, margin=.999,
        screen_accept=.9995, repair_target=.999,
        initial_gap=10., pd_gap=1e-4, pd_stop_tolerance=1e-4,
        pd_max_iterations=16, pd_centering=1., pd_max_increase=10.,
        pd_max_decrease=3., pd_reset_step=False,
        pd_linear_solver="dense", pd_sync_mode="legacy",
        pd_kernel_variant="reference", pd_restore_bisect=True,
        pd_persistent=False, pd_compiled_step=False, pd_fast_setup=False,
        pd_banded_assembly=False, pd_line_search_batch=4,
        compile=False, refine=False, preselect_rows=False,
        toppra_newton=True, toppra_newton_v2=True,
        toppra_only=True, toppra_newton_jerk_cap=True,
        toppra_newton_stride=3, toppra_newton_fit_k=29,
        toppra_newton_scale="batched", toppra_scale_backend="triton",
        toppra_overlap=False, fixed_timing_grid=False,
        zc_free_ends=False,
    )


def _symmetric_limits(request:dict,torch,device):
    from curobo_timeopt.types import RobotLimits

    out={}
    for family,bounds in request["constraints"].items():
        arr=np.asarray(bounds,dtype=np.float64)
        if arr.shape!=(2,7) or not np.isfinite(arr).all():
            raise ValueError("invalid "+family+" bounds")
        if not np.allclose(arr[0],-arr[1],atol=1e-5,rtol=1e-5):
            raise ValueError("UNSUPPORTED_ASYMMETRIC_SIGNED_LIMITS: "+family)
        out[family]=torch.as_tensor(arr[1],device=device,dtype=torch.float32)
    out["torque_rate"]=torch.full((7,),torch.inf,device=device)
    return RobotLimits(**out)


def _diagnostics(screen,limits,torch)->dict:
    # Select a measured limiting family/joint/progress, not an invented
    # gradient. These profiles are from the SAME physical K107 screen.
    profile=screen.profile
    values={"velocity":profile.qd[0],"acceleration":profile.qdd[0],
            "jerk":profile.qddd[0],"torque":screen.torque[0]}
    candidates=[]
    for family,observed in values.items():
        normalized=observed.abs()/getattr(limits,family)[None]
        v,idx=normalized.reshape(-1).max(dim=0)
        joint=int(idx%7)
        sample=int(idx//7)
        candidates.append((float(v),family,sample,joint))
    worst=max(candidates,key=lambda v:v[0])
    grid=screen.profile.x.shape[-1]
    # screen grid may be nonuniform; caller replaces using prepared.grid.
    return {"max_normalized_violation":max(0.,worst[0]-1.),
            "max_normalized_utilization":worst[0],
            "bottleneck_family":worst[1],
            "bottleneck_sample":worst[2],
            "bottleneck_joint":worst[3],
            "diagnostic_valid":math.isfinite(worst[0]),
            "sample_count":grid}


def evaluate(req:dict)->dict:
    import torch
    from curobo_timeopt.path.bspline_path import BSplinePath
    from curobo_timeopt.retiming.problem import RetimingProblem
    from curobo_timeopt.types import ConstraintMode
    from curobo_timeopt.curobo_adapter.dynamics import CuroboDynamicsAdapter
    import benchmark_pdf_timing_polish_runtime as benchmark

    start=time.perf_counter()
    if req.get("retimer_requested")!=K107_ROUTE:
        raise ValueError("this worker implements only the actual K107 route")
    if req.get("endpoint_contract")!=ENDPOINT_CONTRACT:
        raise ValueError("unsupported endpoint contract")
    if req.get("torque_rate_constraint")!="DISABLED_IN_COMMON_FOUR_FAMILY_COMPARISON":
        raise ValueError("torque rate not disabled in common matched contract")
    payload=Payload(**{k:req["payload"][k] for k in (
        "mass_kg","radius_m","native_link","pin_frame")})
    if payload.as_record()["payload_sha256"]!=req["payload"]["payload_sha256"]:
        raise ValueError("payload digest mismatch")
    if digest({k:v for k,v in req.items() if k!="request_sha256"})!=req.get("request_sha256"):
        raise ValueError("worker request checksum mismatch")
    device=torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    robot=native_robot(payload)
    limits=_symmetric_limits(req,torch,device)
    dynamics=CuroboDynamicsAdapter.from_kinematics_config(
        robot.kinematics.kinematics_config,robot.device_cfg)
    pathdata=req["path"]
    if pathdata.get("basis")!="clamped_uniform_bspline" or pathdata.get("degree")!=5:
        raise ValueError("unsupported path basis")
    controls=torch.as_tensor(pathdata["controls"],dtype=torch.float32,device=device)
    knots=torch.as_tensor(pathdata["knots"],dtype=torch.float32,device=device)
    if controls.shape!=(22,7) or knots.shape!=(28,):
        raise ValueError("invalid quintic worker path dimensions")
    path=BSplinePath(controls,degree=5,knots=knots)
    # Exact same geometric path and boundary data to within fp32
    q_ends=path.evaluate(torch.tensor([0.,1.],device=device))
    if float((q_ends-torch.tensor(pathdata["endpoint_q"],device=device)).abs().max())>2e-5:
        raise ValueError("exact path boundary replay mismatch")
    torch.cuda.synchronize()
    setup_ms=(time.perf_counter()-start)*1000.
    s=torch.linspace(0.,1.,1025,device=device)
    problem=RetimingProblem(path=path,limits=limits,s=s,dynamics=dynamics,
        mode=ConstraintMode.c3_no_torque_rate(),initial_speed=0.,
        final_speed=0.,feasibility_tolerance=0.)
    args=defaults()
    incumbent=torch.zeros(107,device=device,dtype=torch.float32)
    benchmark.FIXED_GRIDS=None
    job=benchmark.Request(problem,incumbent,args)
    torch.cuda.synchronize()
    t0=time.perf_counter()
    prepared=job.prepare()
    coeff,kind,repairs,info=job.consume(prepared)
    torch.cuda.synchronize()
    retiming_ms=(time.perf_counter()-t0)*1000.
    t1=time.perf_counter()
    screen=benchmark.screen_pdf_c3_candidates_fd5(
        prepared[1],coeff,torque_rate_backend="native_jvp")
    torch.cuda.synchronize()
    audit_ms=(time.perf_counter()-t1)*1000.
    utilization=max(float(getattr(screen,"rho_"+k).reshape(-1)[0])
                    for k in CONSTRAINT_FAMILIES)
    status=("FEASIBLE_SCREENED" if bool(screen.feasible[0]) and
            bool(screen.positive_timing[0]) and utilization<=args.screen_accept
            else "INFEASIBLE_SCREENED")
    duration=float(screen.profile.duration.reshape(-1)[0])
    if not math.isfinite(duration) or duration<=0:
        status="NUMERICAL_FAILURE"
        duration=None
    diagnostics=_diagnostics(screen,limits,torch)
    sample=int(diagnostics["bottleneck_sample"])
    diagnostics["bottleneck_s"]=float(prepared[1].grid[sample])
    result={
        "schema_version":1,"request_sha256":req["request_sha256"],
        "path_sha256":pathdata["path_sha256"],
        "task_sha256":req["task_sha256"],
        "payload_sha256":req["payload"]["payload_sha256"],
        "retimer_actual":K107_ROUTE,
        "endpoint_contract":ENDPOINT_CONTRACT,
        "active_families":list(CONSTRAINT_FAMILIES),
        "torque_rate_constraint":req["torque_rate_constraint"],
        "status":status,
        "duration_s":duration,
        "model_setup_ms":setup_ms,"retiming_ms":retiming_ms,"audit_ms":audit_ms,
        "worker_wall_ms":(time.perf_counter()-start)*1000.,
        "independent_scaled95_qualified":False,
        "newton_iterations":getattr(info,"newton_steps",None) if info is not None else None,
        "toppra_newton_kind":kind,"repair_count":repairs,
        "max_utilization_online":utilization,
        "diagnostics":diagnostics,
        "timing_coefficients":coeff.detach().cpu().tolist(),
        "screen_grid_s":prepared[1].grid.detach().cpu().tolist(),
        "note":"online native C3 no-rate screen only; independent Pinocchio S8193/adaptive required"
    }
    return result


def run_one(path:Path,response:Path)->dict:
    req=json.loads(path.read_text())
    try:
        result=evaluate(req)
    except Exception as exc:
        # A failed adapter is NOT proof of path infeasibility.
        result={
            "schema_version":1,"request_sha256":req.get("request_sha256"),
            "path_sha256":req.get("path",{}).get("path_sha256"),
            "task_sha256":req.get("task_sha256"),
            "payload_sha256":req.get("payload",{}).get("payload_sha256"),
            "retimer_actual":K107_ROUTE,
            "endpoint_contract":req.get("endpoint_contract"),
            "active_families":list(CONSTRAINT_FAMILIES),
            "torque_rate_constraint":req.get("torque_rate_constraint"),
            "status":"ERROR","duration_s":None,
            "worker_wall_ms":0.,"model_setup_ms":0.,"retiming_ms":0.,"audit_ms":0.,
            "diagnostics":{},"reason":repr(exc),
            "traceback":traceback.format_exc(limit=12)}
    response.parent.mkdir(parents=True,exist_ok=True)
    if response.exists():
        raise ValueError("refusing to overwrite K107 worker result")
    response.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request",required=True,type=Path)
    p.add_argument("--response",required=True,type=Path)
    a=p.parse_args()
    result=run_one(a.request,a.response)
    print(json.dumps({k:v for k,v in result.items() if k not in
                      ("screen_grid_s","timing_coefficients")},allow_nan=False))
    if result["status"]=="ERROR":
        raise SystemExit(1)


if __name__=="__main__":
    main()
