# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize the frozen single-task pointwise timing experiment and checksums."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main() -> None:
    """Generate comparison data, charts and a manifest from recorded attempts."""
    root=Path(__file__).resolve().parents[2]
    out=root/'research/joint_time_oracle/results/native_reconstruction_20261106'
    old=json.loads((out/'qualification_summary.json').read_text())['cases']
    dense=json.loads((out/'pointwise_dense_restored.json').read_text())
    solver=json.loads((out/'pointwise_ipopt_native40_stride1_grid2049_refine1_tol1e-07.json').read_text())
    collision=json.loads((out/'independent_collision.json').read_text())
    restorations=json.loads((out/'scaled_restorations.json').read_text())
    native=json.loads((out/'native_iterations.json').read_text())
    timings={'A_native':old['baseline']['duration_s'],'B_restored':dense['duration_s'],
             'C_joint':old['candidate']['duration_s']}
    source=np.load(out/'fixed_path_attempt.npz'); scale=np.median(source['initial_timing'])
    controls=np.load(out/'pointwise_ipopt_native40_stride1_grid2049_refine1_tol1e-07.npz')['controls'] / scale
    summary={'pilot':'BLOCKED', 'durations_s':timings,
             'A_to_B_s':timings['A_native']-timings['B_restored'],
             'A_to_B_percent':100*(timings['A_native']-timings['B_restored'])/timings['A_native'],
             'B_to_C_s':timings['B_restored']-timings['C_joint'],
             'B_to_C_percent':100*(timings['B_restored']-timings['C_joint'])/timings['B_restored'],
             'scaled_restorations': restorations, 'fixed_path':dense,'solver':solver,
             'solver_crosscheck': json.loads((out/'pointwise_ipopt_native_stride1_grid1025_refine0_tol1e-09_upper10000.json').read_text()),'maximum_scaled_control':float(controls.max()),
             'control_upper_bound':10000.,'native_iterations':native,'collision':collision,
             'qualifications':['IPOPT convergence is for a bounded sampled NLP, and its upper bound is active; unbounded fixed-path optimum is not established.',
                 'B is uniformly slowed after the converged bounded solve; restoration is not a stationary solution.',
                 'A and C satisfy sampled physical bounds but violate 95% scaled bounds; a common scaled comparison is not qualified.',
                 'The complete joint-oracle CUDA Jacobian discrepancy classification is still open.',
                 'Independent collision checks are sampled sphere checks, not continuous certificates.']}
    (out/'timing_research_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    assets=root/'reports/joint_time_oracle/assets'; assets.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(7,3.3)); bars=ax.bar(['A native','B fixed path\nrestored','C joint'],list(timings.values()),color=['#667b91','#298577','#c3743d'])
    ax.set_ylabel('Duration (s)');ax.set_ylim(0.39,0.415);ax.set_title('Single task: duration (axis starts at 0.39 s)')
    ax.bar_label(bars,fmt='%.6f',padding=3);fig.tight_layout();fig.savefig(assets/'timing_decomposition.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.3));ax.plot([r['iterations'] for r in native],[r['duration_s'] for r in native],marker='o')
    ax.axhline(timings['B_restored'],color='#298577',ls='--',label='B restored (independent)')
    ax.set(xlabel='Native TimeOpt iterations',ylabel='Duration (s)',title='Frozen native TimeOpt iteration sweep');ax.legend();fig.tight_layout();fig.savefig(assets/'native_iteration_sweep.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.3))
    for label, name in [('2170 points, native 40 seed', 'pointwise_ipopt_native40_stride1_grid2049_refine1_tol1e-07.npz'),
                        ('1079 points, native 10 seed', 'pointwise_ipopt_native_stride1_grid1025_refine0_tol1e-09_upper10000.npz')]:
        history=np.load(out/name)['history']; ax.plot(history[:,1],history[:,2],label=label)
    ax.set(xlabel='IPOPT iteration', ylabel='Duration objective (s)', title='Bounded pointwise solver convergence')
    ax.legend(); ax.grid(alpha=.2);fig.tight_layout();fig.savefig(assets/'ipopt_convergence.png',dpi=140);plt.close(fig)
    fixed=np.load(out/'pointwise_dense_restored.npz')
    fig, axes=plt.subplots(4,1,figsize=(9,10),sharex=False)
    for label,file,style in [('A',out/'baseline_validation_states.npz','-'),('B',out/'pointwise_dense_restored.npz','--'),('C',out/'candidate_validation_states.npz',':')]:
        states=np.load(file)
        for ax,key in zip(axes,('q','qd','qdd','jerk')):
            ax.plot(states['time'],states[key][:,3],style,label=label,linewidth=1)
    for ax,name in zip(axes,('joint 4 position (rad)','velocity (rad/s)','acceleration (rad/s²)','jerk (rad/s³)')):
        ax.set_ylabel(name); ax.grid(alpha=.2)
    axes[0].legend(ncol=3); axes[-1].set_xlabel('physical time (s)');fig.tight_layout()
    fig.savefig(assets/'pointwise_profiles.png',dpi=140);plt.close(fig)
    names=['pointwise_dense.json','pointwise_dense.npz','pointwise_dense_restored.json','pointwise_dense_restored.npz',
           'pointwise_jacobian.json','pointwise_jacobian_native.npz','pointwise_jacobian_optimized.npz',
           'native_iterations.json','native_iteration_controls.npz','native_40_timing_attempt.npz','independent_collision.json',
           'pointwise_direct_pinocchio.json','scaled_restorations.json','baseline_scaled_restored.npz','candidate_scaled_restored.npz',
           'refinement_input_2049.json','refinement_input_2049.npz',
           'timing_research_summary.json']
    names += [p.name for p in sorted(out.glob('pointwise_ipopt_*.json'))]+[p.name for p in sorted(out.glob('pointwise_ipopt_*.npz'))]
    names += [p.name for p in sorted(out.glob('independent_collision_*.npz'))]
    (out/'timing_research_checksums.json').write_text(json.dumps({name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in names},indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
