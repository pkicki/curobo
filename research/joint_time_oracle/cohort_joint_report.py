# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the five-method exploratory pilot from stored sampled validations."""
from __future__ import annotations
import argparse
import html
import json
import os
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PILOT=ROOT/'research/joint_time_oracle/results/cohort_pilot_isolated_20261106'
RESULTS=Path(os.environ.get('ORACLE_COHORT_RESULTS', ROOT/'research/joint_time_oracle/results/cohort_joint_pilot_20261106'))
REPORT=ROOT/'reports/joint_time_oracle/cohort'
NAMES=('A10','A40','B','C','D')
PAIRS=(('A10','A40'),('A40','B'),('B','D'),('B','C'),('A10','C'))


def values(rows: list, first: str, second: str, gate: str) -> list:
    """Return signed, paired percentage changes only for sampled-feasible candidates."""
    result=[]
    for row in rows:
        a=row['methods'].get(first);b=row['methods'].get(second)
        if a and b and a['verification'].get(gate) and b['verification'].get(gate):
            result.append(100*(a['duration_s']-b['duration_s'])/a['duration_s'])
    return result


def progress(q: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Normalize geometric arc length and discard exactly repeated samples."""
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]
    if arc[-1]<=0:return np.array([0.,1.]),np.stack((q[0],q[-1]))
    indices=np.r_[True,np.diff(arc)>1e-12]
    return arc[indices]/arc[-1],q[indices]


def main() -> None:
    """Rebuild plots and HTML directly from frozen trajectories and diagnostics."""
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=REPORT)
    report=parser.parse_args().output
    report.mkdir(parents=True,exist_ok=True);assets=report/'assets';assets.mkdir(exist_ok=True)
    old=report/'index.html';previous=report/'native_exploratory.html'
    if old.exists() and not previous.exists():shutil.copyfile(old,previous)
    groups={mode:[json.loads(line) for line in (RESULTS/f'{mode}_sampled.jsonl').read_text().splitlines()] for mode in ('physical','scaled95')}
    stats={mode:json.loads((RESULTS/f'{mode}_statistics.json').read_text()) for mode in groups}
    rows=groups['physical'];config=json.loads((RESULTS/'frozen_config.json').read_text());escape=html.escape
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    method_counts=[]
    for i,name in enumerate(NAMES):
        data=[row['methods'][name]['duration_s'] for row in rows if name in row['methods'] and row['methods'][name]['verification']['physical']]
        if data:axes[0].boxplot(data,positions=[i],widths=.6)
        method_counts.append(len(data))
    axes[0].set_xticks(range(5),[f'{name}\n(n={count})' for name,count in zip(NAMES,method_counts)]);axes[0].set_ylabel('duration (s)');axes[0].set_title('Physically feasible sampled candidates')
    pair_counts=[]
    for i,(a,b) in enumerate(PAIRS):
        data=values(rows,a,b,'physical');axes[1].scatter(np.full(len(data),i),data,s=15,alpha=.8);pair_counts.append(len(data))
    axes[1].axhline(0,color='black',lw=.7);axes[1].set_xticks(range(len(PAIRS)),[a+'→'+b+f'\n(n={n})' for (a,b),n in zip(PAIRS,pair_counts)],rotation=30)
    axes[1].set_ylabel('duration improvement (%)');axes[1].set_title('Paired sampled changes')
    fig.tight_layout();fig.savefig(assets/'five_method_summary.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7));labels=('baseline duration (s)','joint path length (rad)',
        'joint total turning (rad)','torque utilization','jerk utilization','joint path deviation (rad)')
    population_color={'free':'#3478ad','table':'#d58a2e','cluttered':'#9a4d79'}
    for row in rows:
        b=row['methods'].get('B');d=row['methods'].get('D')
        if not b or not d or not b['verification']['physical'] or not d['verification']['physical']:continue
        gain=100*(b['duration_s']-d['duration_s'])/b['duration_s'];g=b['geometry_metrics'];u=b['constraint_utilization']
        xs=(b['duration_s'],g['joint_path_length_rad'],g['joint_total_turning_rad'],u.get('torque'),u.get('jerk'),
            d['geometry_metrics'].get('arc_aligned_deviation_rms_rad'))
        for ax,x in zip(axes.flat,xs):
            if x is not None:ax.scatter(x,gain,color=population_color.get(row['population'],'gray'),s=20,alpha=.8)
    for ax,label in zip(axes.flat,labels):ax.axhline(0,color='black',lw=.5);ax.set_xlabel(label);ax.set_ylabel('B→D improvement (%)')
    for name,color in population_color.items():axes.flat[0].scatter([],[],color=color,label=name)
    axes.flat[0].legend();fig.suptitle('Exploratory sampled comparisons');fig.tight_layout()
    fig.savefig(assets/'geometry_relationships.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9,7))
    for ax,family in zip(axes.flat,('velocity','acceleration','jerk','torque')):
        for i,name in enumerate(NAMES):
            data=[row['methods'][name]['constraint_utilization'].get(family) for row in rows
                  if name in row['methods'] and row['methods'][name]['verification']['physical']]
            data=[x for x in data if x is not None]
            if data:ax.boxplot(data,positions=[i])
        ax.set_xticks(range(len(NAMES)),NAMES);ax.axhline(1,color='red',lw=.6,ls='--');ax.set_title(family+' / physical limit')
    fig.suptitle('Sampled peak constraint utilization');fig.tight_layout()
    fig.savefig(assets/'five_method_utilization.png',dpi=130);plt.close(fig)
    sections=[]
    for mode,source in [('physical','physical'),('scaled95','scaled95')]:
        sections.append('<h2>'+('Physical limits' if mode=='physical' else '95%-scaled limits')+'</h2>')
        summary=stats[source]['groups'].get('pilot',stats[source]['groups']['all'])
        sections.append('<table><tr><th>Contrast</th><th>Paired / attempted</th><th>Median %</th><th>Mean %</th><th>Bootstrap 95% CI</th><th>Faster / slower</th><th>&gt;1%, &gt;5%, &gt;10%</th><th>Median |Δ| ms</th></tr>')
        for a,b in PAIRS:
            item=summary['comparisons'][mode+':'+a+'->'+b];fmt=lambda x:'—' if x is None else f'{x:.3f}'
            ci=item['median_bootstrap95_pct'];label='—' if ci is None else f'[{ci[0]:.3f}, {ci[1]:.3f}]'
            sections.append(f"<tr><td>{a}→{b}</td><td>{item['eligible_pairs']}/{summary['tasks']}</td><td>{fmt(item['median_improvement_pct'])}</td><td>{fmt(item['mean_improvement_pct'])}</td><td>{label}</td><td>{item['improved_count']} / {item['slower_count']}</td><td>{item['improvement_gt_1pct']}, {item['improvement_gt_5pct']}, {item['improvement_gt_10pct']}</td><td>{fmt(item['median_absolute_difference_ms'])}</td></tr>")
        sections.append('</table>')
    population_rows=[]
    for population in ('free','table','cluttered'):
        for mode in ('physical','scaled95'):
            key='pilot/'+population;summary=stats[mode]['groups'].get(key)
            if summary is None:continue
            cells=[]
            for a,b in PAIRS[:4]:
                item=summary['comparisons'][mode+':'+a+'->'+b];value=item['median_improvement_pct']
                cells.append('—' if value is None else f"{value:.3f}% (n={item['eligible_pairs']})")
            population_rows.append('<tr><td>'+population+'</td><td>'+mode+'</td>'+''.join('<td>'+x+'</td>' for x in cells)+'</tr>')
    population_table='<table><tr><th>Population</th><th>Limits</th><th>A10→A40</th><th>A40→B</th><th>B→D</th><th>B→C</th></tr>'+''.join(population_rows)+'</table>'
    fig,ax=plt.subplots(figsize=(7,3.5))
    for i,name in enumerate(('B','C','D')):
        runtime=[row['execution'].get(name,{}).get('wall_s') for row in rows]
        runtime=[v for v in runtime if v is not None]
        if runtime:ax.boxplot(runtime,positions=[i])
    ax.set_xticks(range(3),('B','C','D'));ax.set_ylabel('subprocess wall time (s)')
    ax.set_title('Offline computation incl. startup; not production latency')
    fig.tight_layout();fig.savefig(assets/'offline_runtime.png',dpi=130);plt.close(fig)
    table=[]
    for row in rows:
        case=row['case_id'];methods=row['methods'];fig,axes=plt.subplots(5,1,figsize=(10,11),sharex=True)
        traj={};ee={}
        for j,name in enumerate(NAMES):
            file=(PILOT if name in ('A10','A40') else RESULTS)/case/f'{name.lower()}_dense.npz'
            torque=RESULTS/case/f'{name.lower()}_pin_torque.npz'
            if not file.exists() or not torque.exists():continue
            a=np.load(file);data={k:a[k] for k in a.files};pin=np.load(torque);traj[name]=data
            ee[name]=pin['end_effector'] if 'end_effector' in pin.files else None
            for ax,arr in zip(axes,(data['q'],data['qd'],data['qdd'],data['jerk'],pin['torque'])):
                for k in range(7):ax.plot(data['time'],arr[:,k],color=plt.cm.tab10(k),ls=['-','--',':','-.',(0,(3,1,1,1))][j],lw=.6,alpha=.7)
        for ax,label in zip(axes,('q (rad)','v (rad/s)','a (rad/s²)','jerk (rad/s³)','torque (Nm)')):ax.set_ylabel(label);ax.grid(alpha=.15)
        axes[-1].set_xlabel('physical time (s)');axes[0].set_title(case+' — joints colored; A10 solid, A40 dashed, B dotted, C dash-dot, D long dash')
        fig.tight_layout();fig.savefig(assets/(case+'_joint.png'),dpi=100);plt.close(fig)
        fig=plt.figure(figsize=(10,4));ax=fig.add_subplot(121);ax3=fig.add_subplot(122,projection='3d')
        for name in ('A10','B','C','D'):
            if name not in traj:continue
            p,q=progress(traj[name]['q']);query=np.linspace(0,1,501)
            ax.plot(query,np.interp(query,p,q[:,3]),label=name)
            path=ee.get(name)
            if path is not None:ax3.plot(path[:,0],path[:,1],path[:,2],label=name)
        ax.set(xlabel='normalized joint arc length',ylabel='joint 4 (rad)',title='Geometric path, reparameterized');ax.legend()
        ax3.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title='End-effector paths');fig.tight_layout()
        fig.savefig(assets/(case+'_geometry.png'),dpi=110);plt.close(fig)
        info=escape(json.dumps(row,indent=2))
        (report/(case+'_joint.html')).write_text('<!doctype html><meta charset="utf-8"><title>'+escape(case)+'</title><h1>'+escape(case)+'</h1><p>Exploratory independently sampled results; no continuous-time certification.</p><img style="max-width:100%" src="assets/'+escape(case)+'_joint.png"><img style="max-width:100%" src="assets/'+escape(case)+'_geometry.png"><pre>'+info+'</pre>',encoding='utf-8')
        table.append('<tr><td><a href="'+escape(case)+'_joint.html">'+escape(case)+'</a></td><td>'+escape(row['population'])+'</td>'+''.join('<td>'+('—' if name not in methods else f"{methods[name]['duration_s']:.6f}"+(' ✓' if methods[name]['verification']['physical'] else ' !'))+'</td>' for name in NAMES)+'</tr>')
    geometry=stats['physical']['groups']['pilot']['comparisons']['physical:B->D']
    if geometry['eligible_pairs']:
        median=geometry['median_improvement_pct'];ci=geometry['median_bootstrap95_pct']
        interval='unavailable' if ci is None else f'[{ci[0]:.2f}%, {ci[1]:.2f}%]'
        conclusion=(f"Across {geometry['eligible_pairs']} physically sampled B–D pairs, the median geometric change is "
                    f"{median:.2f}% (paired bootstrap 95% interval {interval}); "
                    f"{geometry['improved_count']} are faster and {geometry['slower_count']} slower. "
                    'This small generated pilot cannot establish a general or global effect.')
    else:
        conclusion='There are no independently sampled feasible B–D pairs; geometric benefit remains unknown.'
    interesting=[]
    for row in rows:
        b=row['methods'].get('B');d=row['methods'].get('D')
        if b and d and b['verification']['physical'] and d['verification']['physical']:
            gain=100*(b['duration_s']-d['duration_s'])/b['duration_s']
            difference=d['geometry_metrics']['joint_path_length_rad']-b['geometry_metrics']['joint_path_length_rad']
            interesting.append((gain,difference,row['case_id']))
    examples='No valid sampled B–D examples.'
    if interesting:
        ordered=sorted(interesting)
        longer=[item[2] for item in interesting if item[0]>0 and item[1]>0]
        examples=(f"Largest sampled B→D change: {ordered[-1][2]} ({ordered[-1][0]:.3f}%); "
                  f"smallest: {ordered[0][2]} ({ordered[0][0]:.3f}%). "
                  f"Longer joint path but shorter duration: {', '.join(longer) if longer else 'none observed'}.")
    counts=stats['physical']['groups']['pilot']['methods']
    counts_html=''.join(f"<li>{name}: {counts[name]['with_duration']} outputs; {counts[name]['physical_qualified']} independently sampled physical passes; {counts[name]['solver_converged']} solver-converged.</li>" for name in NAMES)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Joint timing pilot</title>
<style>body{font:15px/1.5 system-ui;max-width:1200px;margin:2em auto;padding:0 1em}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #ccc;text-align:left;padding:.4em}.warn{background:#fff2dc;padding:1em}pre{overflow:auto}</style>
<h1>Five-method exploratory pilot</h1><p class="warn">Results are independently sampled local candidates, not global optima or continuous-time certificates. Collision checks include neighborhoods of detected sampled minima, not continuous swept certificates; full CUDA collision-gradient qualification remains incomplete. Physical and scaled results use different restored candidates where needed. No 100-task confirmation has been run.</p>
<p>Implementation frozen at commit '''+escape(config['implementation_commit'])+'''. B and D retain the exact original geometric curves with matched timing formulations. C includes feasibility-restored initial fits when solver iterates fail; see candidate origin per case. Timing comparisons are exploratory and conditional on sampled feasibility.</p>
<h2>Derivative and convergence qualification</h2><p>The pointwise fixed-path timing Jacobian has a finite-difference sweep test and the timing NLP used local-support sparse Jacobian structure. The historical CUDA joint-optimizer Jacobian audit still contains discrepancies associated with active-sample/contact switches; those cases are not all classified, so the joint optimizer's derivative path is not fully qualified. B and D converged to local discretized IPOPT solutions on 20/20 tasks each. C produced a solver-converged selected candidate on only '''+str(counts['C']['solver_converged'])+'''/20; feasible nonconverged candidates remain in the analysis. No result has a global-optimality certificate. See <a href="../../../research/joint_time_oracle/results/native_reconstruction_20261106/pointwise_jacobian.json">pointwise Jacobian evidence</a> and the <a href="../../index.html">historical derivative audit</a>.</p>
<img src="assets/five_method_summary.png" alt="Duration distribution and paired changes" style="max-width:100%"><h2>Method outcomes</h2><ul>'''+counts_html+'</ul>'+''.join(sections)+'''
<h2>Mechanisms and limits</h2><img src="assets/geometry_relationships.png" alt="B versus D improvement against duration, path, and dynamics" style="max-width:100%"><img src="assets/five_method_utilization.png" alt="Physical constraint utilization" style="max-width:100%"><h2>Populations and convergence</h2>'''+population_table+'''<img src="assets/offline_runtime.png" alt="Offline subprocess runtime distributions" style="max-width:100%"><p>Population-specific means, confidence intervals and failure exclusions are in the linked machine-readable statistics. A10 and A40 are fixed iteration budgets, not convergence certificates. B/D status refers to a local discretized NLP. See per-task pages for sampled constraint diagnostics, path lengths, and candidate origins.</p>
<p><a href="../../../research/joint_time_oracle/results/cohort_joint_final_20261106/physical_statistics.json">Physical statistics</a> · <a href="../../../research/joint_time_oracle/results/cohort_joint_final_20261106/scaled95_statistics.json">Scaled statistics</a> · <a href="native_exploratory.html">Earlier native-only exploratory report</a></p>
<h2>Per-task profiles</h2><label>Search <input id="query" oninput="filterRows()"></label><table><thead><tr><th>Task</th><th>Population</th><th>A10 (s)</th><th>A40 (s)</th><th>B (s)</th><th>C (s)</th><th>D (s)</th></tr></thead><tbody id="cases">'''+''.join(table)+'''</tbody></table><p>✓ = sampled physical pass; ! = sampled failure. Missing methods are never counted as zero improvement.</p>
<h2>Representative cases</h2><p>'''+escape(examples)+'''</p><h2>Research implications</h2><p>'''+escape(conclusion)+''' Fixed-path solver local minima and nonconverged joint optimization can mask possible improvements. Geometry approximation in C initialization, sampled constraints, model approximations and limited homotopy exploration prevent a global conclusion. Native iteration gains and solver differences are shown separately.</p>
<script>function filterRows(){let q=document.getElementById('query').value.toLowerCase();for(const r of document.querySelectorAll('#cases tr'))r.hidden=!r.textContent.toLowerCase().includes(q)}</script></html>'''
    (report/'index.html').write_text(page,encoding='utf-8')
    print('wrote',report/'index.html')


if __name__=='__main__':main()
