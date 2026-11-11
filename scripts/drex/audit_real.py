"""Post-hoc CPU audit of already frozen real-scene Drex decisions. No inference."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

import real_data as data
import real_pilot as pilot
from runtime import read, write
from transfer import sha, verify


def session_summary(gains):
    values=np.asarray([np.mean(v) for _,v in sorted(gains.items())],dtype=np.float64)
    rng=np.random.default_rng(data.SEED)
    means=values[rng.integers(0,len(values),(2000,len(values)))].mean(axis=1)
    by_session={s:float(np.mean(v)) for s,v in sorted(gains.items())}
    return dict(mean_pp=float(values.mean()),cluster_ci95_pp=np.quantile(means,[.025,.975]).tolist(),
        by_session=by_session,
        leave_one_session_out_pp={s:float(np.mean([v for sid,v in by_session.items() if sid!=s]))
                                  for s in by_session} if len(values)>1 else {})


def paired_contrast(rows,a,b):
    pairs=defaultdict(dict)
    for r in rows: pairs[r['token']][r['method']]=r
    groups=defaultdict(list)
    for group in pairs.values():
        x,y=group[a],group[b]
        if x['session_id']!=y['session_id']: raise ValueError('Unmatched scene/session')
        groups[x['session_id']].append(x['raw_gain_pp']-y['raw_gain_pp'])
    return session_summary(groups)


def geometric_order_audit(records,samples,results,labels):
    groups=defaultdict(dict)
    for r in results: groups[r['token']][(r['mode'],r['ordering'])]=r
    output={}
    for kind in ('reversed','permuted'):
        events=[]
        for record in records:
            token=record['token']; s=samples[token]; group=groups[token]
            def selected(ordering):
                r=group[('camera',ordering)]; p=np.empty(16)
                p[r['pool_positions']]=r['probabilities']
                return s['pool_local'].index(pilot.choice(p,s))
            a,b=selected('canonical'),selected(kind)
            displacement=np.linalg.norm(s['trajectories'][a,:,:2]-s['trajectories'][b,:,:2],axis=1)
            delta=float(100*(labels[token][b,0]-labels[token][a,0]))
            events.append(dict(token=token,session_id=record['session_id'],choice_changed=a!=b,
                mean_displacement_m=float(displacement.mean()),max_displacement_m=float(displacement.max()),
                delta_pdm_pp=delta))
        changed=[r for r in events if r['choice_changed']]
        output[kind]=dict(scenes=len(events),choice_flips=len(changed),
            changed_choices_with_mean_distance_over_1m=sum(r['mean_displacement_m']>1 for r in changed),
            changed_choices_with_pdm_delta_over_half_pp=sum(abs(r['delta_pdm_pp'])>.5 for r in changed),
            max_mean_displacement_m=max(r['mean_displacement_m'] for r in events),
            largest_pdm_changes=sorted(events,key=lambda r:abs(r['delta_pdm_pp']),reverse=True)[:5])
    return output


def run(package,out):
    verify(package)
    manifest,index,scenes,vocab=data.package_metadata(package)
    identity=read(out/'identity.json')
    if sha(package/'package_manifest.json')!=identity['package_manifest_sha256']:
        raise RuntimeError('Package differs from frozen pilot')
    records=identity['records']
    if {r['token'] for r in records}-{*index}: raise ValueError('Unexpected cohort scenes')
    samples={r['token']:data.load_inputs(package,r['token'],manifest,vocab) for r in records}
    decisions,results=pilot.committed_decisions(out,records,samples)
    labels={r['token']:pilot.load_labels(package,r['token'],manifest,samples[r['token']]) for r in records}
    rows=pilot.score_decisions(decisions,labels)
    primary=[r for r in rows if r['method']=='drex_mean3']
    groups=defaultdict(list)
    for r in primary: groups[r['session_id']].append(r['raw_gain_pp'])
    result=dict(status='complete',type='posthoc_cpu_audit',decision_commit_sha256=sha(out/'decision_commit.json'),
        primary_concentration=session_summary(groups),
        camera_single_minus_blank_single=paired_contrast(rows,'drex_single','blank_single'),
        mean3_minus_blank_single_policy_comparison=paired_contrast(rows,'drex_mean3','blank_single'),
        note=('Only single-camera vs single-blank is the matched photo ablation. Mean3 vs single-blank '
              'also changes aggregation. Leave-one-session-out is a sensitivity analysis, not exclusion '
              'of difficult scenes or model tuning. No new decisions, training, inference or holdout.'),
        order_geometry=geometric_order_audit(records,samples,results,labels),
        primary_largest_losses=sorted(primary,key=lambda r:r['raw_gain_pp'])[:5],
        primary_largest_gains=sorted(primary,key=lambda r:r['raw_gain_pp'],reverse=True)[:5])
    write(out/'audit_v1.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args(); root=a.root.expanduser().resolve()
    run(root/'data/navsim_train_v1/package',a.run.expanduser().resolve())
