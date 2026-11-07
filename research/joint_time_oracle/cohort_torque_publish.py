# SPDX-License-Identifier: Apache-2.0
"""Publish torque-stress HTML summaries from frozen raw evidence.

Re-runnable offline. No solver runs, no hidden filtering, and no fabricated
durations. Run cohort_torque_report for each regime before publishing.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil

import numpy as np

from .torque_active import read_profile, torque_utilization

ROOT=Path(__file__).resolve().parents[2]
STRESS=ROOT/"research/joint_time_oracle/results/torque_stress_20261107"
REPORT=ROOT/"reports/joint_time_oracle/torque_active"


def fmt(x):
    return "N/A" if x is None else f"{x:.3f}%"


COLORS = {"A10":"#3765a8", "A40":"#49a0a7", "B":"#df8f2d",
          "C":"#b34d6d", "D":"#6956a5"}


def _downsample(x: np.ndarray, y: np.ndarray, max_points: int=100) -> tuple[np.ndarray,np.ndarray]:
    """Keep a deterministic representative subset while preserving both endpoints."""
    count=len(x)
    if count<=max_points:return x,y
    index=np.unique(np.linspace(0,count-1,max_points).astype(int))
    return x[index],y[index]


def _svg_panels(panels: list[tuple[str,list[tuple[str,np.ndarray,np.ndarray,str]]]],
                x_label: str, y_label: str, *, width: int=900, panel_height: int=185) -> str:
    """Render compact static SVG panels from saved trajectory samples."""
    height=panel_height*len(panels);parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        '<style>text{font:12px system-ui;fill:#233044}.axis{stroke:#8793a2;stroke-width:1}.grid{stroke:#e5e9ef;stroke-width:1}</style>']
    for pindex,(title,series) in enumerate(panels):
        top=pindex*panel_height;left=62;right=width-18;up=top+27;bottom=top+panel_height-30
        finite_x=np.concatenate([np.asarray(a[1],dtype=float) for a in series]) if series else np.array([0.,1.])
        finite_y=np.concatenate([np.asarray(a[2],dtype=float) for a in series]) if series else np.array([0.,1.])
        xmin,xmax=float(np.nanmin(finite_x)),float(np.nanmax(finite_x));ymin,ymax=float(np.nanmin(finite_y)),float(np.nanmax(finite_y))
        if xmax<=xmin:xmax=xmin+1.
        if ymax<=ymin:
            pad=max(abs(ymin)*.05,1e-3);ymin-=pad;ymax+=pad
        else:
            pad=.06*(ymax-ymin);ymin-=pad;ymax+=pad
        parts.append(f'<text x="{left}" y="{top+17}" font-weight="600">{html.escape(title)}</text>')
        for tick in range(1,4):
            y=up+(bottom-up)*tick/4
            parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>')
        parts.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/><line class="axis" x1="{left}" y1="{up}" x2="{left}" y2="{bottom}"/>')
        for name,x,y,color in series:
            x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);x,y=_downsample(x,y)
            good=np.isfinite(x)&np.isfinite(y);x=x[good];y=y[good]
            if len(x)<2:continue
            px=left+(x-xmin)/(xmax-xmin)*(right-left);py=bottom-(y-ymin)/(ymax-ymin)*(bottom-up)
            coords=" ".join(f'{a:.1f},{b:.1f}' for a,b in zip(px,py))
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.5" opacity=".88" points="{coords}"/>')
        parts.append(f'<text x="{(left+right)/2:.1f}" y="{top+panel_height-5}" text-anchor="middle">{html.escape(x_label)}</text>')
        parts.append(f'<text x="14" y="{(up+bottom)/2:.1f}" transform="rotate(-90 14 {(up+bottom)/2:.1f})" text-anchor="middle">{html.escape(y_label)}</text>')
        if pindex==0:
            lx=width-155
            for name,color in COLORS.items():
                parts.append(f'<line x1="{lx}" y1="{top+12}" x2="{lx+14}" y2="{top+12}" stroke="{color}" stroke-width="2"/><text x="{lx+18}" y="{top+16}">{name}</text>');lx+=43
    parts.append('</svg>')
    return ''.join(parts)


def _series_page(arm: Path, task: dict, output: Path) -> dict:
    """Write a trajectory detail page from independently validated artifacts."""
    case=task['case_id'];folder=arm/case;profile=read_profile(folder/'torque_profile.json')
    diag=json.loads((arm/'independent_adaptive.json').read_text())
    trajectories={};torques={};durations={}
    for method in COLORS:
        qpath=folder/f'{method.lower()}_adaptive_dense.npz';tpath=folder/f'{method.lower()}_adaptive_pin_torque.npz'
        if not qpath.exists() or not tpath.exists():continue
        with np.load(qpath) as z:trajectories[method]={k:np.asarray(z[k]) for k in ('time','q','qd','qdd','jerk')}
        with np.load(tpath) as z:torques[method]=np.asarray(z['torque'])
        durations[method]=float(trajectories[method]['time'][-1])
    limits=np.load(ROOT/'research/joint_time_oracle/results/native_reconstruction_20261106/baseline_validation_states.npz')
    receipts_path=arm/'adaptive_receipts.json'
    receipts=json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
    max_duration=max(durations.values(),default=1.)
    panels={name:[] for name in ('position','velocity','acceleration','jerk','torque')}
    keys={'position':'q','velocity':'qd','acceleration':'qdd','jerk':'jerk'}
    for joint in range(7):
        for family,state_key in keys.items():
            lines=[]
            for method,data in trajectories.items():lines.append((method,data['time'],data[state_key][:,joint],COLORS[method]))
            low,high=limits[family+'_bounds'];lines.extend([('lower',np.array([0.,max_duration]),np.full(2,low[joint]),'#777777'),('upper',np.array([0.,max_duration]),np.full(2,high[joint]),'#777777')])
            panels[family].append((f'Joint {joint+1}',lines))
        lines=[]
        for method in trajectories:
            lines.append((method,trajectories[method]['time'],torques[method][:,joint],COLORS[method]))
        lines.extend([('lower',np.array([0.,max_duration]),np.full(2,profile['effort_lower'][joint]),'#777777'),('upper',np.array([0.,max_duration]),np.full(2,profile['effort_upper'][joint]),'#777777')])
        panels['torque'].append((f'Joint {joint+1}',lines))
    sections=[]
    for family,title,ylabel in [('position','Joint positions','rad'),('velocity','Joint velocities','rad/s'),('acceleration','Joint accelerations','rad/s²'),('jerk','Joint jerks','rad/s³'),('torque','Pinocchio actuator torque','Nm')]:
        sections.append(f'<h3>{title}</h3>'+_svg_panels(panels[family],'physical time (s)',ylabel))
    path_panels=[];ee_xy=[];ee_xz=[];ee_yz=[];arc_progress={}
    for method,data in trajectories.items():
        q=data['q'];distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]
        if distance[-1]>0:distance/=distance[-1]
        arc_progress[method]=distance
    for joint in range(7):
        path_panels.append((f'Joint {joint+1}',[(method,arc_progress[method],data['q'][:,joint],COLORS[method])
                                                for method,data in trajectories.items()]))
    # End-effector samples are included in the independently computed torque archives.
    ee_data={}
    for method in trajectories:
        with np.load(folder/f'{method.lower()}_adaptive_pin_torque.npz') as z:ee_data[method]=np.asarray(z['end_effector'])
    for method,xyz in ee_data.items():
        ee_xy.append((method,xyz[:,0],xyz[:,1],COLORS[method]));ee_xz.append((method,xyz[:,0],xyz[:,2],COLORS[method]));ee_yz.append((method,xyz[:,1],xyz[:,2],COLORS[method]))
    sections += ['<h3>Geometric joint path by normalized joint-space arc length</h3>'+_svg_panels(path_panels,'normalized arc length','joint position (rad)'),
                 '<h3>End-effector path projections</h3>'+_svg_panels([('XY',ee_xy),('XZ',ee_xz),('YZ',ee_yz)],'position (m)','position (m)')]
    profile_diag={m:diag.get(f'{case}:{m}_adaptive',{}) for m in trajectories}
    rows=[]
    for m,duration in durations.items():
        item=profile_diag[m];torque_use=float(np.max(torque_utilization(torques[m],np.stack((profile['effort_lower'],profile['effort_upper'])))))
        receipt=receipts.get(f'{case}:{m}',{})
        status='physical + Scaled95' if receipt.get('scaled95') else ('physical' if receipt.get('physical') else 'unqualified')
        residuals={family:float(item.get('families',{}).get(family,{}).get('physical',{}).get('violation',0.)) for family in item.get('families',{})}
        worst_family=max(residuals,key=residuals.get) if residuals else 'n/a'
        worst=f'{worst_family}: {residuals[worst_family]:.3g}' if residuals else 'n/a'
        joint_length=float(np.linalg.norm(np.diff(trajectories[m]['q'],axis=0),axis=1).sum())
        rows.append(f'<tr><td>{m}</td><td>{duration:.6f}</td><td>{joint_length:.4f}</td><td>{100*torque_use:.2f}%</td><td>{item.get("world_min_clearance_m")}</td><td>{item.get("self_min_clearance_m")}</td><td>{item.get("samples")}</td><td>{worst}</td><td>{status}</td></tr>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>''' + html.escape(case) + ''' — torque stress</title><style>body{font:14px/1.45 system-ui;max-width:1050px;margin:2rem auto;padding:0 1rem;color:#1c2938}svg{width:100%;height:auto;margin:.7rem 0 1.5rem;background:white}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:.45rem;text-align:left}</style><a href="../../index.html">← report</a><h1>''' + html.escape(case) + '''</h1><p>''' + html.escape(task['population']) + ''' population; ''' + html.escape(profile['regime']) + ''' actuator derating (synthetic). Endpoint, Pinocchio torque, and primitive collision checks use the adaptive sampled replay. This is not a continuous-time certificate.</p><table><thead><tr><th>Method</th><th>Duration (s)</th><th>Joint path length (rad)</th><th>Torque utilization</th><th>World clearance (m)</th><th>Self clearance (m)</th><th>Samples</th><th>Worst physical residual</th><th>Independent audit</th></tr></thead><tbody>''' + ''.join(rows) + '''</tbody></table>''' + ''.join(sections) + '''</html>'''
    output.write_text(page,encoding='utf-8')
    return {'case_id':case,'population':task['population'],'regime':profile['regime'],'durations_s':durations}


def publish(source: Path, destination: Path) -> dict:
    """Publish summaries, trajectory detail pages, and checksummed provenance."""
    destination.mkdir(parents=True, exist_ok=True)
    summary={"status":"sampled adaptive verification; not continuous-time certification",
             "raw_results":str(source.resolve()),"regimes":{}}
    artifact_rows=[];case_rows=[];overview_rows=[];conclusions=[];population_rows=[];threshold_rows=[];manifests={}
    method_names=tuple(COLORS)
    for regime in ("moderate", "severe"):
        arm=source/regime;summary_path=arm/"torque_summary.json"
        if not summary_path.exists():
            summary["regimes"][regime]={"status":"NOT_EXECUTED"}
            overview_rows.append(f"<tr><td>{regime}</td><td colspan='6'>Not executed</td></tr>")
            continue
        record=json.loads(summary_path.read_text())
        stats=record["adaptive_qualified"]["groups"]["all"]["comparisons"]
        sampled=record["sampled"]["groups"]["all"]["comparisons"]
        primary=stats["physical:B->D"];exploratory=sampled["physical:B->D"]
        activity=[x for x in record["torque_utilization"] if x["method"]=="B"]
        tasks=json.loads((arm/"tasks.json").read_text())
        rows={r["case_id"]:r for r in (json.loads(line) for line in (arm/"pilot.jsonl").read_text().splitlines())}
        receipts_path=arm/"adaptive_receipts.json"
        receipts=json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
        target=destination/"cases"/regime;target.mkdir(parents=True,exist_ok=True)
        detail_by_case={x["case_id"]:x for x in record["case_details"]}
        b_util=[float(x["max_torque_utilization"]) for x in activity if x.get("max_torque_utilization") is not None]
        threshold_counts={}
        for method in method_names:
            values=[float(x["max_torque_utilization"]) for x in record["torque_utilization"]
                    if x["method"]==method and x.get("max_torque_utilization") is not None]
            threshold_counts[method]={str(t):sum(v>=t for v in values) for t in (.80,.85,.90,.95)}
            threshold_rows.append("<tr><td>"+regime+"</td><td>"+method+"</td>"+
                f"<td>{len(values)}</td>"+"".join(f"<td>{threshold_counts[method][str(t)]}</td>" for t in (.80,.85,.90,.95))+"</tr>")
        out={"attempted_source_tasks":record["source_case_count"],
             "A10_recorded":sum(rows.get(t["case_id"],{}).get("methods",{}).get("A10",{}).get("duration_s") is not None for t in tasks),
             "A40_recorded":sum(rows.get(t["case_id"],{}).get("methods",{}).get("A40",{}).get("duration_s") is not None for t in tasks),
             "physical_qualified_BD_pairs":primary["eligible_pairs"],
             "physical_BD_median_pct":primary["median_improvement_pct"],
             "physical_BD_bootstrap95_pct":primary["median_bootstrap95_pct"],
             "sampled_BD_pairs":exploratory["eligible_pairs"],
             "sampled_BD_median_pct":exploratory["median_improvement_pct"],
             "B_torque_utilization_median":float(np.median(b_util)) if b_util else None,
             "B_torque_utilization_max":float(max(b_util)) if b_util else None,
             "B_torque_ge_80pct":sum(x>=.80 for x in b_util),
             "B_torque_ge_85pct":sum(x>=.85 for x in b_util),
             "B_torque_ge_90pct":sum(x>=.90 for x in b_util),
             "B_torque_ge_95pct":sum(x>=.95 for x in b_util),
             "torque_threshold_counts_by_method":threshold_counts,
             "A10_A40_B_C_D_physical_receipts":{m:sum(receipts.get(f"{t['case_id']}:{m}",{}).get("physical") is True for t in tasks) for m in method_names},
             "A10_A40_B_C_D_scaled95_receipts":{m:sum(receipts.get(f"{t['case_id']}:{m}",{}).get("scaled95") is True for t in tasks) for m in method_names}}
        population_comparisons={}
        for population in ("free", "table", "cluttered"):
            group=record["adaptive_qualified"]["groups"].get("population:"+population,{})
            contrast=group.get("comparisons",{}).get("physical:B->D",{})
            ci=contrast.get("median_bootstrap95_pct")
            population_comparisons[population]={
                "attempted_tasks":contrast.get("attempted_tasks",0),
                "qualified_pairs":contrast.get("eligible_pairs",0),
                "median_improvement_pct":contrast.get("median_improvement_pct"),
                "mean_improvement_pct":contrast.get("mean_improvement_pct"),
                "median_bootstrap95_pct":ci,
                "improved_count":contrast.get("improved_count",0),
                "slower_count":contrast.get("slower_count",0)}
            ci_text="not available" if ci is None else f"[{ci[0]:.2f}%, {ci[1]:.2f}%]"
            population_rows.append(f"<tr><td>{regime}</td><td>{population}</td><td>{contrast.get('eligible_pairs',0)}/{contrast.get('attempted_tasks',0)}</td><td>{fmt(contrast.get('median_improvement_pct'))}</td><td>{fmt(contrast.get('mean_improvement_pct'))}</td><td>{ci_text}</td><td>{contrast.get('improved_count',0)}/{contrast.get('slower_count',0)}</td></tr>")
        out["population_BD_comparisons"]=population_comparisons
        summary["regimes"][regime]=out
        overview_rows.append("<tr><td>"+regime+"</td>"+
            f"<td>{out['attempted_source_tasks']}</td><td>{primary['eligible_pairs']}</td>"+
            f"<td>{fmt(primary['median_improvement_pct'])}</td>"+
            f"<td>{fmt(primary['median_bootstrap95_pct'][0] if primary['median_bootstrap95_pct'] else None)} – {fmt(primary['median_bootstrap95_pct'][1] if primary['median_bootstrap95_pct'] else None)}</td>"+
            f"<td>{out['B_torque_utilization_median']*100:.1f}%</td><td>{out['B_torque_ge_85pct']}/{len(b_util)}</td></tr>")
        for contrast in ("physical:A10->A40","physical:A40->B","physical:B->D","physical:B->C"):
            v=stats[contrast];ci=v["median_bootstrap95_pct"]
            ci_text="not available" if ci is None else f"[{ci[0]:.3f}%, {ci[1]:.3f}%]"
            conclusions.append(f"<tr><td>{regime}</td><td>{contrast.split(':')[1]}</td><td>{v['eligible_pairs']}/{v['attempted_tasks']}</td><td>{fmt(v['median_improvement_pct'])}</td><td>{fmt(v['mean_improvement_pct'])}</td><td>{ci_text}</td><td>{v['improved_count']}/{v['slower_count']}</td></tr>")
        for task in tasks:
            case=task["case_id"]
            page=target/(case+".html")
            page_meta=_series_page(arm,task,page)
            detail=detail_by_case[case]["methods"]
            native=rows.get(case,{}).get("methods",{})
            durations={m:(native.get(m,{}).get("duration_s") if m in ("A10","A40") else detail.get(m,{}).get("duration_s")) for m in method_names}
            b,d=durations.get("B"),durations.get("D")
            bd=100*(b-d)/b if b and d else None
            case_metrics={m:{k:detail.get(m,{}).get(k) for k in ("joint_path_length_rad","joint_path_curvature_p95","end_effector_path_length_m","max_utilization","world_clearance_m","self_clearance_m")} for m in method_names}
            case_rows.append({**page_meta,"detail":f"cases/{regime}/{case}.html", "metrics":case_metrics,
                "source_task_sha256":task["task_sha256"],"torque_profile_sha256":read_profile(arm/case/"torque_profile.json")["profile_sha256"],
                "durations_s":durations,"B_to_D_pct":bd,
                "physical":{m:receipts.get(f"{case}:{m}",{}).get("physical") is True for m in method_names},
                "scaled95":{m:receipts.get(f"{case}:{m}",{}).get("scaled95") is True for m in method_names}})
        manifests[regime]=json.loads((arm/"execution_manifest.json").read_text()) if (arm/"execution_manifest.json").exists() else None
        for file in arm.rglob("*"):
            if file.is_file() and file.suffix in (".json", ".jsonl", ".npz", ".log", ".yaml", ".yml"):
                artifact_rows.append({"path":str(file.relative_to(source)),"sha256":hashlib.sha256(file.read_bytes()).hexdigest(),"bytes":file.stat().st_size})
    # Static duration and improvement profiles are derived only from saved task rows.
    duration_panels=[];gain_panels=[]
    for regime in ("moderate","severe"):
        group=[r for r in case_rows if r["regime"]==regime]
        lines=[]
        for m in method_names:
            values=np.asarray([r["durations_s"][m] for r in group if r["durations_s"][m] is not None],dtype=float)
            if values.size:lines.append((m,np.arange(len(values)),values,COLORS[m]))
        duration_panels.append((regime+" — task durations by method",lines))
        gains=[r["B_to_D_pct"] for r in group if r["B_to_D_pct"] is not None]
        gain_panels.append((regime+" — B→D geometric duration improvement",[("paired tasks",np.arange(len(gains)),np.asarray(gains),COLORS["D"])]))
    duration_svg=_svg_panels(duration_panels,"task index within method","execution duration (s)")
    gain_svg=_svg_panels(gain_panels,"task index","improvement (%)")
    table_rows=[]
    for row in case_rows:
        d=row["durations_s"]
        table_rows.append(f"<tr data-search='{html.escape(row['case_id']+' '+row['population']+' '+row['regime'],quote=True)}'><td><a href='{row['detail']}'>{html.escape(row['case_id'])}</a></td><td>{row['regime']}</td><td>{row['population']}</td>"+
            "".join(f"<td>{'—' if d[m] is None else f'{d[m]:.6f}'} {'✓' if row['physical'][m] else '×'}</td>" for m in method_names)+
            f"<td>{fmt(row['B_to_D_pct'])}</td></tr>")
    c_attempts=[]
    for regime in ("moderate","severe"):
        for attempt_file in (source/regime).glob("*/c_attempts.json"):
            c_attempts.extend(json.loads(attempt_file.read_text()).get("attempts",[]))
    c_trials=[trial for attempt in c_attempts
              for trial in attempt.get("result", {}).get("trials", [])]
    c_converged=sum(trial.get("solver_success") is True for trial in c_trials)
    source_manifest=json.loads((source/"manifest.json").read_text()) if (source/"manifest.json").exists() else {}
    source_task_count=int(source_manifest.get("source_task_count", len(case_rows)))
    page="""<!doctype html><html lang='en'><meta charset='utf-8'><title>Torque-active joint geometry study</title>
<style>body{font:15px/1.5 system-ui;max-width:1250px;margin:2rem auto;padding:0 1rem;color:#1d2938}table{border-collapse:collapse;width:100%;margin:.8rem 0}td,th{border-bottom:1px solid #dce2e8;padding:.45rem;text-align:left}.note{background:#fff1e4;border-left:5px solid #bd6e14;padding:1rem}.scroll{overflow-x:auto}svg{width:100%;height:auto;background:white}</style>
<h1>Joint path-and-time optimization with active torque limits</h1>
<p class='note'><b>Scope and qualification.</b> Torque stress is synthetic uniform derating of asymmetric Panda actuator limits; it is not a physical payload experiment. This source inventory contains {source_task_count} tasks; moderate/severe deratings are correlated observations of shared source tasks, not independent tasks. Independent checks use aligned Pinocchio RNEA and analytic collision sphere primitives. Added samples surround observed extrema; no continuous-time collision or dynamic certificate is claimed. C is a local optimizer; only individually marked candidates pass sampled physical checks, and no global optimum is claimed.</p>
<p>Methods: A10/A40 native TimeOpt on the frozen cuRobo geometry and modified effort limits (10/40 iterations); B pointwise IPOPT timing on the original geometry; C SLSQP joint path/time with 14/22/34 controls and four starts per resolution; D the same IPOPT timing solver on C's exact quintic geometry. Primary geometry contrast: B versus D.</p>
<h2>Duration distributions and paired geometry effect</h2>"""+duration_svg+"<h2>B→D paired task improvements</h2>"+gain_svg+"<h2>Adaptive sampled qualification</h2><table><thead><tr><th>Regime</th><th>Attempted</th><th>Qualified B-D pairs</th><th>Median improvement</th><th>Paired bootstrap 95% CI</th><th>Median B torque utilization</th><th>B >=85%</th></tr></thead><tbody>"+"".join(overview_rows)+"</tbody></table><h2>Torque activity by method</h2><p>Counts are trajectory counts per admitted task and regime, using maximum Pinocchio torque divided by the derated asymmetric effort limit.</p><div class='scroll'><table><thead><tr><th>Regime</th><th>Method</th><th>Audited</th><th>>=80%</th><th>>=85%</th><th>>=90%</th><th>>=95%</th></tr></thead><tbody>"+"".join(threshold_rows)+"</tbody></table></div><h2>Paired effects</h2><div class='scroll'><table><thead><tr><th>Regime</th><th>Comparison</th><th>Qualified / attempted</th><th>Median (%)</th><th>Mean (%)</th><th>Paired bootstrap 95% CI</th><th>Improved / slower</th></tr></thead><tbody>"+"".join(conclusions)+"</tbody></table></div><h2>Searchable task results</h2><input id='filter' type='search' oninput='filterRows()' placeholder='Filter task, population, regime'><div class='scroll'><table><thead><tr><th>Task</th><th>Regime</th><th>Population</th><th>A10 (s)</th><th>A40 (s)</th><th>B (s)</th><th>C (s)</th><th>D (s)</th><th>B→D (%)</th></tr></thead><tbody>"+"".join(table_rows)+"</tbody></table></div><h2>Solver diagnostics</h2><p>C converged for "+str(c_converged)+" of "+str(len(c_trials))+" multistart attempts across "+str(len(c_attempts))+" task-regime solver reports. Nonconverged but independently feasible candidates remain visible; independent feasibility does not establish local stationarity. B/D use the same fixed-path IPOPT implementation and 513 collocation points with adaptive refinements. The per-task pages show all seven joint position, velocity, acceleration, jerk and torque profiles, normalized geometric path profiles, and end-effector path projections.</p><h2>Research interpretation</h2><p>Moderate and severe results are summarized separately. The severe regime is the critical test: a positive B→D effect means a changed C geometry retimed by the same fixed-path solver improved on B's original geometry. Gains do not establish that broad geometry search is exhausted. C convergence is incomplete despite multistart, homotopy exploration is limited, and finite sampling and omission of a full planner rerun under derated torque remain material limitations. No global optimality claim is made.</p><p>Raw results: <code>"+html.escape(str(source.resolve()))+"</code>. Frozen method manifests and full artifact SHA-256 inventory are linked below.</p><p><a href='summary.json'>aggregate summary</a> · <a href='case_index.json'>case index</a> · <a href='execution_manifests.json'>execution manifests</a> · <a href='artifact_manifest.json'>artifact checksums</a></p><script>function filterRows(){let q=document.getElementById('filter').value.toLowerCase();document.querySelectorAll('tbody tr[data-search]').forEach(r=>r.style.display=r.dataset.search.includes(q)?'':'none')}</script></html>"
    page=page.replace("{source_task_count}", str(source_task_count))
    implication_rows=[]
    for regime in ("moderate", "severe"):
        value=summary["regimes"].get(regime,{})
        if value.get("physical_qualified_BD_pairs") is None: continue
        ci=value.get("physical_BD_bootstrap95_pct")
        ci_text="not available" if ci is None else f"[{ci[0]:.2f}%, {ci[1]:.2f}%]"
        implication_rows.append(f"<li><b>{regime.title()}.</b> B→D median duration change was {value['physical_BD_median_pct']:.2f}% across {value['physical_qualified_BD_pairs']} verified pairs (paired bootstrap 95% CI {ci_text}); positive values mean the changed geometry retimed faster.</li>")
    implications=("<h2>Research Implications</h2><p>These local-search results show whether geometry added value after matched fixed-path timing; they do not establish a global minimum.</p><ul>"+"".join(implication_rows)+"</ul><p>Per-task geometry metrics and plots distinguish path length changes from duration. Where D is faster despite a longer joint-space path, shortening the path alone cannot explain the gain. Similar peak constraint utilization indicates a different feasible path can use the same active limits more effectively; the data do not isolate a unique dynamic mechanism.</p><p>Only A10/A40 fixed-geometry timing was rerun under derated effort. The geometric planner itself was not rerun with actuator limits, so superiority over a fully torque-aware cuRoboV2-Fast planner is not established.</p>")
    population_section="<h2>B→D geometry effect by population</h2><p>Paired, independently sampled physical-feasibility results; moderate and severe are shown separately because they reuse source tasks.</p><div class='scroll'><table><thead><tr><th>Regime</th><th>Population</th><th>Pairs / attempted</th><th>Median gain</th><th>Mean gain</th><th>Paired bootstrap 95% CI</th><th>Improved / slower</th></tr></thead><tbody>"+"".join(population_rows)+"</tbody></table></div>"
    page=page.replace("<h2>Research interpretation</h2>",population_section+implications+"<h2>Research interpretation</h2>")
    lane_reports=[]
    for regime in ("moderate", "severe"):
        lane_file=source/regime/"geometric_lane_search.json"
        if not lane_file.exists(): continue
        lane_data=json.loads(lane_file.read_text())
        for case,entry in lane_data.get("cases", {}).items():
            b_file=source/regime/case/"b_attempt.json"
            if not b_file.exists(): continue
            baseline=json.loads(b_file.read_text()).get("dense_duration_s")
            lanes=[x.get("duration_s") for x in entry.get("lanes", [])
                   if x.get("source_success") and x.get("duration_s") is not None]
            if baseline and lanes:
                improvement=100*(baseline-min(lanes))/baseline
                lane_reports.append((regime,case,len(lanes),improvement))
    if lane_reports:
        lane_rows="".join(f"<tr><td>{r}</td><td>{html.escape(c)}</td><td>{n}</td><td>{g:.4f}%</td></tr>" for r,c,n,g in lane_reports)
        lane_median=float(np.median([x[3] for x in lane_reports]))
        lane_section=("<h2>Additional cuRobo geometric candidates</h2><p>Four saved planner lanes were fixed-path retimed on "+str(len({x[1] for x in lane_reports}))+" distinct task geometries (up to three per regime), yielding "+str(len(lane_reports))+" successful task-regime checks. The median best-lane gain over B was "+f"{lane_median:.4f}%"+". These lanes provide limited route diversity and do not constitute exhaustive homotopy exploration or CEM/CMA-ES.</p><table><thead><tr><th>Regime</th><th>Task</th><th>Valid lanes</th><th>Best lane gain vs B</th></tr></thead><tbody>"+lane_rows+"</tbody></table>")
        page=page.replace("<h2>Research interpretation</h2>",lane_section+"<h2>Research interpretation</h2>")
    wider_index=destination.parent/"torque_active_wide"/"index.html"
    if destination.name != "torque_active_wide" and wider_index.exists():
        page=page.replace("<h2>Research interpretation</h2>","<p><a href='../torque_active_wide/index.html'>Open the separate source-disjoint wider-excursion cohort report</a>.</p><h2>Research interpretation</h2>")
    (destination/"index.html").write_text(page,encoding="utf-8")
    (destination/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (destination/"case_index.json").write_text(json.dumps(case_rows,indent=2)+"\n")
    (destination/"execution_manifests.json").write_text(json.dumps(manifests,indent=2)+"\n")
    (destination/"artifact_manifest.json").write_text(json.dumps({"source_root":str(source.resolve()),"artifacts":artifact_rows},indent=2)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=STRESS)
    p.add_argument("--report",type=Path,default=REPORT)
    a=p.parse_args()
    print(json.dumps(publish(a.source,a.report),indent=2))


if __name__=="__main__":
    main()
