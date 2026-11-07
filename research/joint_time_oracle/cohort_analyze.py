# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate independently sampled pilot evidence into separate feasibility contracts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .cohort_statistics import analyze
from .cohort_retime import PILOT, RESULTS
from .cohort_geometry import NativeGeometry, QuinticGeometry


def passed(diag: dict | None, mode: str, min_samples: int = 8193) -> bool:
    """Apply explicitly declared tolerances to independently sampled diagnostics."""
    if not diag:return False
    endpoint=diag['endpoints']; family=diag['families']
    return bool(all(family[name][mode]['violation']<=1e-5 for name in family)
        and (diag['world_min_clearance_m'] is None or diag['world_min_clearance_m']>=-1e-6)
        and diag['self_min_clearance_m']>=-1e-6
        and endpoint['position_start']<=1e-5 and endpoint['position_goal']<=1e-5
        and endpoint['velocity']<=1e-4 and endpoint['acceleration']<=1e-3
        and diag['samples']>=min_samples)


def geometry_metrics(q: np.ndarray, ee: np.ndarray | None) -> dict:
    """Calculate sampled joint-space length and a stable geometric turning measure."""
    dq=np.diff(q,axis=0);length=np.linalg.norm(dq,axis=1);valid=length>1e-9
    unit=dq[valid]/length[valid,None]
    turning=float(np.arccos(np.clip(np.sum(unit[:-1]*unit[1:],axis=1),-1,1)).sum()) if len(unit)>1 else 0.
    return {'joint_path_length_rad':float(length.sum()),'joint_total_turning_rad':turning,
            'end_effector_path_length_m':float(np.linalg.norm(np.diff(ee,axis=0),axis=1).sum()) if ee is not None else None}


def arc_samples(q: np.ndarray, count: int = 1001) -> np.ndarray:
    """Represent a path by normalized joint-space arc length for geometric comparison."""
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]
    if arc[-1] <= 0:return np.repeat(q[:1],count,axis=0)
    keep=np.r_[True,np.diff(arc)>1e-12];progress=arc[keep]/arc[-1]
    return np.stack([np.interp(np.linspace(0,1,count),progress,q[keep,j]) for j in range(q.shape[1])],axis=1)


def main() -> None:
    """Write source-preserving physical and restored-scaled JSONL and statistics."""
    tasks=json.loads((PILOT/'tasks.json').read_text());byid={t['case_id']:t for t in tasks}
    bounds_archive=np.load(Path(__file__).resolve().parent/'results/native_reconstruction_20261106/baseline_validation_states.npz')
    bounds={name:bounds_archive[name+'_bounds'] for name in ('velocity','acceleration','jerk','torque')}
    native={r['case_id']:r for r in (json.loads(line) for line in (PILOT/'pilot.jsonl').read_text().splitlines())}
    execution={r['case_id']:r for r in (json.loads(line) for line in (RESULTS/'execution.jsonl').read_text().splitlines())}
    original=json.loads((RESULTS/'independent_original.json').read_text())
    scaled=json.loads((RESULTS/'independent_scaled.json').read_text())
    native_diag=json.loads((RESULTS/'independent_native.json').read_text())
    scaled_meta=json.loads((RESULTS/'scaled_restoration.json').read_text())
    adaptive=json.loads((RESULTS/'independent_collision_adaptive.json').read_text())
    dynamic_adaptive=json.loads((RESULTS/'independent_dynamic_adaptive.json').read_text())
    scaled_dynamic_adaptive=json.loads((RESULTS/'independent_scaled_adaptive.json').read_text())
    allrows={'physical':[],'scaled95':[]}
    for task in tasks:
        case=task['case_id'];folder=RESULTS/case;baseline=native[case]
        baseline_buffers=np.load(PILOT/case/'a10_buffers.npz')
        native_geometry=NativeGeometry(baseline_buffers,baseline['methods']['A10']['lane'])
        exact_native=native_geometry.evaluate_trimmed(np.linspace(0,1,8193))[0]
        cfile=folder/'c_best.json'
        exact_joint=(QuinticGeometry(np.asarray(json.loads(cfile.read_text())['coefficients'])).evaluate(np.linspace(0,1,8193))[0]
                     if cfile.exists() else None)
        for gate in allrows:
            row={'case_id':case,'cohort':'pilot','population':task['population'],'task_sha256':task['task_sha256'],
                 'methods':{},'execution':execution.get(case,{}).get('stages',{}),
                 'qualification_note':'Exploratory independent sampled checks only; adaptive collision and continuous-time certification incomplete.'}
            for name in ('A10','A40','B','C','D'):
                source=(PILOT if name in ('A10','A40') else RESULTS)/case/f'{name.lower()}_dense.npz'
                if not source.exists():continue
                origdiag=(native_diag if name in ('A10','A40') else original).get(case+':'+name)
                selected=source;diag=origdiag;restored=False
                original_adaptive=dynamic_adaptive.get(case+':'+name+'_adaptive')
                needs_scaled=not passed(origdiag,'scaled95') or (original_adaptive is not None and not passed(original_adaptive,'scaled95',1))
                if gate=='scaled95' and needs_scaled:
                    candidate=folder/f'{name.lower()}_scaled_dense.npz'
                    if candidate.exists():selected=candidate;diag=scaled.get(case+':'+name+'_scaled');restored=True
                with np.load(selected) as archive:
                    time=archive['time'];q=archive['q'];duration=float(time[-1])
                    dynamic={name:archive[key] for name,key in [('velocity','qd'),('acceleration','qdd'),('jerk','jerk')]}
                ee_path=(folder/f'{name.lower()}_scaled_pin_torque.npz' if restored else
                         RESULTS/case/f'{name.lower()}_pin_torque.npz')
                ee=None
                if ee_path.exists():
                    with np.load(ee_path) as a:
                        if 'end_effector' in a.files:ee=a['end_effector']
                        dynamic['torque']=a['torque']
                adaptive_pin=folder/f'{name.lower()}_adaptive_pin_torque.npz'
                if adaptive_pin.exists():
                    with np.load(adaptive_pin) as a:
                        dynamic['torque_audit']=a['torque']
                collision_check=adaptive.get(case+':'+name)
                extra_name=name+('_scaled_adaptive' if restored else '_adaptive')
                dynamic_check=(scaled_dynamic_adaptive if restored else dynamic_adaptive).get(case+':'+extra_name)
                physical=passed(diag,'physical') and (collision_check is None or collision_check['qualified'])
                if dynamic_check is not None:physical=physical and passed(dynamic_check,'physical',1)
                scaled_ok=physical and passed(diag,'scaled95')
                if dynamic_check is not None:scaled_ok=scaled_ok and passed(dynamic_check,'scaled95',1)
                entry={'duration_s':duration,'verification':{'physical':physical,'scaled95':scaled_ok,
                       'endpoint':physical,'dense':bool(diag and diag['samples']>=8193),
                       'collision_independent':bool(diag),'dynamics_independent':bool(diag)},
                       'adaptive_collision_checked':adaptive.get(case+':'+name,{}).get('qualified',False),
                       'adaptive_collision_diagnostics':adaptive.get(case+':'+name),
                       'adaptive_dynamic_checked':bool(dynamic_check),
                       'adaptive_dynamic_diagnostics':dynamic_check,
                       'continuous_certified':False,'restored':restored,
                       'diagnostic_key':case+':'+name+('_scaled' if restored else ''),
                       'geometry_metrics':geometry_metrics(exact_native if name in ('A10','A40','B') else exact_joint,ee), 'constraint_diagnostics':diag,
                       'constraint_utilization':{}}
                for family,val in dynamic.items():
                    if family=='torque_audit':
                        family='torque'
                    lo,hi=bounds[family]
                    # Effort bounds may be asymmetric; positive and negative sides are handled separately.
                    ratio=np.maximum(np.where(val>=0,val/hi,0),np.where(val<0,val/lo,0))
                    utilization=float(np.max(ratio))
                    entry['constraint_utilization'][family]=max(
                        utilization,entry['constraint_utilization'].get(family,0.))
                if name in ('B','D'):
                    info=json.loads((folder/f'{name.lower()}_attempt.json').read_text())
                    entry.update({'solver_converged':info['solver_converged'],'control_bound_active':info['lower_control_bound_active'],
                                  'stationarity_inf':info['stationarity_inf'],'collocation_points':info['collocation_points'],
                                  'primal_min_margin':info['primal_min_margin']})
                elif name=='C':
                    best=json.loads((folder/'c_best.json').read_text());entry['candidate_origin']=best.get('candidate_origin')
                    attempts=json.loads((folder/'c_attempts.json').read_text())['attempts']
                    trials=[trial for attempt in attempts for trial in attempt['result']['trials']]
                    entry['solver_converged']=any(trial['solver_success'] and abs(trial['duration']-best['duration'])<1e-5
                                                  for trial in trials) and best.get('candidate_origin')!='feasibility_restored_initial_fit'
                    entry['attempt_count']=len(trials)
                    entry['successful_solver_statuses']=sum(bool(trial['solver_success']) for trial in trials)
                else:entry['solver_converged']=False
                if restored:entry['restoration']=scaled_meta.get(case+':'+name)
                row['methods'][name]=entry
            baseline_file=PILOT/case/'a10_dense.npz'
            if baseline_file.exists():
                base_path=arc_samples(exact_native)
                for name,entry in row['methods'].items():
                    candidate=(PILOT if name in ('A10','A40') else RESULTS)/case/f'{name.lower()}_dense.npz'
                    path=arc_samples(exact_native if name in ('A10','A40','B') else exact_joint)
                    distance=np.linalg.norm(path-base_path,axis=1)
                    entry['geometry_metrics']['arc_aligned_deviation_max_rad']=float(distance.max())
                    entry['geometry_metrics']['arc_aligned_deviation_rms_rad']=float(np.sqrt(np.mean(distance**2)))
                if 'B' in row['methods']:
                    buffers=np.load(PILOT/case/'a10_buffers.npz')
                    lane=native[case]['methods']['A10']['lane'];geometry=NativeGeometry(buffers,lane)
                    with np.load(RESULTS/case/'b_dense.npz') as bdata:
                        error=float(np.max(np.abs(bdata['q']-geometry.evaluate_trimmed(bdata['progress'])[0])))
                    row['methods']['B']['geometry_metrics']['exact_replay_max_error_rad']=error
                    row['methods']['B']['geometry_replay_sampled_ok']=error<=1e-8
                    if error>1e-8:
                        row['methods']['B']['verification']['physical']=False
                        row['methods']['B']['verification']['scaled95']=False
                if 'C' in row['methods'] and 'D' in row['methods']:
                    with np.load(RESULTS/case/'c_dense.npz') as cdata: cpath=arc_samples(cdata['q'])
                    with np.load(RESULTS/case/'d_dense.npz') as ddata:
                        dpath=arc_samples(ddata['q'])
                        controls=np.asarray(json.loads((RESULTS/case/'c_best.json').read_text())['coefficients'])
                        error=float(np.max(np.abs(ddata['q']-QuinticGeometry(controls).evaluate(ddata['progress'])[0])))
                    row['methods']['D']['geometry_metrics']['c_to_d_arc_deviation_max_rad']=float(np.linalg.norm(cpath-dpath,axis=1).max())
                    row['methods']['D']['geometry_metrics']['exact_replay_max_error_rad']=error
                    row['methods']['D']['geometry_replay_sampled_ok']=error<=1e-8
                    if error>1e-8:
                        row['methods']['D']['verification']['physical']=False
                        row['methods']['D']['verification']['scaled95']=False
            allrows[gate].append(row)
    for gate,rows in allrows.items():
        (RESULTS/f'{gate}_sampled.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        (RESULTS/f'{gate}_statistics.json').write_text(json.dumps(analyze(rows),indent=2)+'\n')
    manifest={str(p.relative_to(RESULTS)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in RESULTS.rglob('*') if p.is_file() and p.name!='checksums.json'}
    (RESULTS/'checksums.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('wrote exploratory paired statistics for',len(tasks),'tasks')


if __name__=='__main__':main()
