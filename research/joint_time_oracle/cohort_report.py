# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render exploratory pilot evidence and preserve a reproducible checksum manifest."""
from __future__ import annotations
import hashlib
import html
import json
import subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .cohort_statistics import analyze, html_report, read_records

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research/joint_time_oracle/results/cohort_pilot_isolated_20261106'
REPORT=ROOT/'reports/joint_time_oracle/cohort'


def main() -> None:
    """Rebuild pilot statistics and per-task figures without running optimization."""
    rows=read_records(OUT/'pilot.jsonl'); diagnostics=json.loads((OUT/'independent_sampled.json').read_text())
    tasks=json.loads((OUT/'tasks.json').read_text()); byid={t['case_id']:t for t in tasks}
    REPORT.mkdir(parents=True,exist_ok=True);assets=REPORT/'assets';assets.mkdir(exist_ok=True)
    summary=analyze(rows)
    (REPORT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (REPORT/'statistics.html').write_text(html_report(rows,summary),encoding='utf-8')
    colors=['#375c87','#c07429']
    fig,ax=plt.subplots(figsize=(9,4))
    for i,row in enumerate(rows):
        a=row['methods'].get('A10',{}).get('duration_s');b=row['methods'].get('A40',{}).get('duration_s')
        if a and b:
            ax.plot([i-.13,i+.13],[a,b],color='#aaa',lw=.7)
            ax.scatter([i-.13,i+.13],[a,b],color=colors,s=14)
    ax.set(xlabel='Exploratory task index',ylabel='Duration (s)',title='Unqualified native A10 and A40 durations')
    ax.scatter([],[],color=colors[0],label='A10');ax.scatter([],[],color=colors[1],label='A40');ax.legend()
    fig.tight_layout();fig.savefig(assets/'native_durations.png',dpi=130);plt.close(fig)
    table=[]
    for row in rows:
        case=row['case_id'];task=byid[case]; methods=row['methods']
        fig,axes=plt.subplots(5,1,figsize=(10,12),sharex=True)
        for j,method in enumerate(('A10','A40')):
            filename=OUT/case/f'{method.lower()}_dense.npz';torquefile=OUT/case/f'{method.lower()}_pin_torque.npz'
            if not filename.exists() or not torquefile.exists(): continue
            archive=np.load(filename);s={k:archive[k] for k in archive.files};tau=np.load(torquefile)['torque']
            for ax,value in zip(axes,(s['q'],s['qd'],s['qdd'],s['jerk'],tau)):
                for joint in range(7):
                    ax.plot(s['time'],value[:,joint],ls='-' if j==0 else '--',color=plt.cm.tab10(joint),lw=.7,alpha=.8)
        for ax,label in zip(axes,('Position (rad)','Velocity (rad/s)','Acceleration (rad/s²)','Jerk (rad/s³)','Torque (Nm)')):
            ax.set_ylabel(label);ax.grid(alpha=.2)
        axes[-1].set_xlabel('Physical time (s)');axes[0].set_title(case+': A10 solid, A40 dashed; sampled, unqualified')
        fig.tight_layout();fig.savefig(assets/(case+'.png'),dpi=110);plt.close(fig)
        detail={'task':task,'result':row,'independent_sampled':{m:diagnostics.get(case+':'+m) for m in ('A10','A40')}}
        (REPORT/(case+'.html')).write_text('<!doctype html><meta charset="utf-8"><title>'+html.escape(case)+
             '</title><h1>'+html.escape(case)+'</h1><p>Unqualified exploratory output. Solid A10; dashed A40.</p>'+
             '<img style="max-width:100%" src="assets/'+html.escape(case)+'.png"><pre>'+html.escape(json.dumps(detail,indent=2))+'</pre>')
        table.append('<tr><td><a href="'+html.escape(case)+'.html">'+html.escape(case)+'</a></td><td>'+html.escape(row['population'])+
                     '</td>'+''.join('<td>'+('—' if methods.get(m,{}).get('duration_s') is None else f"{methods[m]['duration_s']:.6f}")+'</td>' for m in ('A10','A40','B','C','D'))+
                     '<td>Unqualified</td></tr>')
    physical=sum(all(w['physical']['violation']<=1e-6 for w in v['families'].values()) and
                 (v['world_min_clearance_m'] is None or v['world_min_clearance_m']>=0) and v['self_min_clearance_m']>=0
                 for v in diagnostics.values())
    scaled=sum(all(w['scaled95']['violation']<=1e-6 for w in v['families'].values()) and
               (v['world_min_clearance_m'] is None or v['world_min_clearance_m']>=0) and v['self_min_clearance_m']>=0
               for v in diagnostics.values())
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Exploratory pilot</title>
<style>body{font:15px system-ui;margin:2em auto;max-width:1100px;line-height:1.5}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ccc;padding:.4em;text-align:left}.warn{background:#fff1dd;padding:1em}</style>
<h1>Exploratory 20-task pilot: blocked</h1><p class="warn">No qualified paired comparisons. B, C and D were not run:
the arbitrary speed-control bound, exact-geometry retiming executor, adaptive qualification and CUDA derivative classification remain unresolved.
No confirmation cohort was launched. Native durations below are unqualified diagnostics; sampled checks are not continuous-time certification.</p>
'''+f'<p>{len(rows)} tasks attempted; {len(diagnostics)} native trajectories independently sampled; {physical} pass sampled physical limits and collision, {scaled} pass sampled 95%-scaled limits and collision. Endpoint and adaptive qualification are not implied by those counts.</p>'+'''
<img src="assets/native_durations.png" alt="Unqualified duration distributions" style="max-width:100%">
<p><a href="statistics.html">Fail-closed paired statistics and exclusion counts</a> · <a href="summary.json">Machine-readable statistical summary</a></p>
<label>Search cases <input id="query" oninput="filterRows()"></label><table><thead><tr><th>Task</th><th>Population</th><th>A10 (s)</th><th>A40 (s)</th><th>B</th><th>C</th><th>D</th><th>Status</th></tr></thead><tbody id="cases">'''+''.join(table)+'''</tbody></table>
<script>function filterRows(){let q=document.getElementById('query').value.toLowerCase();for(const r of document.querySelectorAll('#cases tr'))r.hidden=!r.textContent.toLowerCase().includes(q)}</script></html>'''
    (REPORT/'index.html').write_text(page,encoding='utf-8')
    files=['curobo/content/configs/robot/franka.yml','curobo/content/assets/robot/franka_description/franka_panda.urdf',
           'curobo/content/configs/scene/collision_test.yml','curobo/content/configs/scene/collision_primitives_3d.yml',
           'research/joint_time_oracle/cohort_explore.py','research/joint_time_oracle/cohort_validate.py']
    previous = OUT/'study_manifest.json'
    base = (json.loads(previous.read_text())['git_parent'] if previous.exists() else
            subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    meta={'task_source':'deterministic generated endpoints; not a representative benchmark',
          'runner_provenance_note':'Execution used working-tree scripts; the current runner adds a post-run geometry assertion and the validator adds an input option, so their checksums are not exact executed-code checksums.',
          'git_parent':base,
          'planner_gpu':'NVIDIA GeForce RTX 3090','cuda_shim':'results/device_property_compat.py',
          'task_count':len(tasks),'input_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
          'qualification':'blocked: no adaptive checks, B/C/D executor or qualified paired comparisons',
          'sample_counts':{'independent_checked':len(diagnostics),'physical_and_collision':physical,'scaled95_and_collision':scaled}}
    (OUT/'study_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='checksums.json']
    (OUT/'checksums.json').write_text(json.dumps({str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
    print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
