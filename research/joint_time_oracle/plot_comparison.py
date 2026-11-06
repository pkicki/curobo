# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate comparison figures from the stored, unqualified trajectories."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root=Path(__file__).resolve().parents[2]
    source=root/'research/joint_time_oracle/results/native_reconstruction_20261106'
    output=root/'reports/joint_time_oracle/assets'; output.mkdir(parents=True,exist_ok=True)
    names=('baseline','fixed_path','candidate')
    labels=('A: native TimeOpt','B: independent timing','C: joint candidate')
    colors=('tab:blue','tab:orange','tab:green')
    data=[np.load(source/f'{name}_validation_states.npz') for name in names]
    torques=[np.load(source/f'{name}_pinocchio_dense.npz')['tau'] for name in names]
    fig,axs=plt.subplots(5,7,figsize=(22,13),sharex='col')
    for row,(field,bound) in enumerate([('q','position'),('qd','velocity'),('qdd','acceleration'),
                                        ('jerk','jerk'),('tau','torque')]):
        for joint in range(7):
            ax=axs[row,joint]
            for i,d in enumerate(data):
                values=torques[i] if field=='tau' else d[field]
                ax.plot(d['time'],values[:,joint],color=colors[i],label=labels[i],lw=0.8)
            lo,hi=data[0][bound+'_bounds'][:,joint]
            ax.axhline(lo,c='gray',ls=':',lw=.6);ax.axhline(hi,c='gray',ls=':',lw=.6)
            if row==0: ax.set_title(f'Joint {joint+1}')
            if joint==0: ax.set_ylabel(field)
            if row==4: ax.set_xlabel('time (s)')
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=3)
    fig.suptitle('Unqualified single-task comparison; dotted lines are physical limits',y=.995)
    fig.tight_layout(rect=(0,0,1,.96)); fig.savefig(output/'comparison_profiles.png',dpi=120);plt.close(fig)
    fig,axs=plt.subplots(4,1,figsize=(10,9),sharex=True)
    for ax,(field,bound) in zip(axs,[('qd','velocity'),('qdd','acceleration'),('jerk','jerk'),('tau','torque')]):
        for i,d in enumerate(data):
            value=torques[i] if field=='tau' else d[field]
            low,high=d[bound+'_bounds']
            utilization=np.maximum(value/high,value/low).max(axis=1)
            ax.plot(d['time'],utilization,color=colors[i],label=labels[i],lw=.8)
        ax.axhline(1,c='gray',ls=':',lw=.8);ax.set_ylabel(bound)
    axs[-1].set_xlabel('physical time (s)');axs[0].legend();
    fig.suptitle('Maximum per-joint physical limit utilization');fig.tight_layout()
    fig.savefig(output/'constraint_utilization.png',dpi=130);plt.close(fig)
    ee=np.load(source/'ee_paths.npz');fig=plt.figure(figsize=(12,5));ax=fig.add_subplot(121,projection='3d'); geom=fig.add_subplot(122)
    for name,label,color in zip(names,labels,colors):
        xyz=ee[name];ax.plot(xyz[:,0],xyz[:,1],xyz[:,2],label=label,color=color)
        q=ee[name+'_q'];arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]
        if arc[-1]>0:arc/=arc[-1]
        for j in range(q.shape[1]): geom.plot(arc,q[:,j],color=color,alpha=.5,lw=.8)
    ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title='Panda hand path');ax.legend(fontsize=7)
    geom.set(xlabel='normalized joint-space arc length',ylabel='joint angle (rad)',title='Joint paths (all joints)')
    fig.tight_layout();fig.savefig(output/'comparison_geometry.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    main()
