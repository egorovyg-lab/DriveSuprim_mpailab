"""V2 central-domain projected candidate images; isolated scene-parallel inference."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import time
import traceback

import numpy as np
import overlay_v2_data as view
import audit_overlay_projection as projection_audit
import real_data as base
import real_pilot as old
from real_parallel import assignment
from runtime import read,write,environment,remaining_bytes
from transfer import sha
from night import load_result,save_result,validate_probabilities
from audit_real import paired_contrast


def name(token,mode,order):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',token): raise ValueError('Unsafe token')
    return f'{token}-{mode}-{order}.json'


def prepare(root,package,out,replicas):
    if replicas not in (1,2,4,8): raise ValueError('Use 1,2,4 or 8 replicas')
    manifest,records,scenes,vocab=base.prepare(package,root,out/'base')
    plan=dict(config=view.CONFIG,replicas=replicas,assignments=assignment(records,replicas),records=records,
        base_identity_sha256=sha(out/'base/identity.json'),
        sources={n:sha(Path(__file__).with_name(n)) for n in ('overlay_v2_data.py','overlay_v2_pilot.py','run_overlay_v2.sh',
            'overlay_data.py','overlay_pilot.py','audit_overlay_projection.py','audit_real.py')},note=view.NOTE)
    base.freeze(out/'plan.json',plan)
    return plan,manifest,records,scenes,vocab


def check_plan(out):
    plan=read(out/'plan.json')
    for n,h in plan['sources'].items():
        if sha(Path(__file__).with_name(n))!=h: raise RuntimeError('Overlay code changed; use a new output name')
    if sha(out/'base/identity.json')!=plan['base_identity_sha256']: raise RuntimeError('Base identity changed')
    commit=read(out/'projection_commit.json')
    if commit['plan_sha256']!=sha(out/'plan.json'):
        raise RuntimeError('V2 projection contract belongs to another plan')
    for filename,expected in commit['files'].items():
        if sha(out/filename)!=expected: raise RuntimeError('V2 projection diagnostics changed')
    report=read(out/'projection_report.json')
    if report['status']!='render_contract_passed' or report['retained_legacy_suspect_segments']!=0:
        raise RuntimeError('V2 projection contract did not pass')
    return plan


def samples_for(package,manifest,records,vocab):
    return {r['token']:base.load_inputs(package,r['token'],manifest,vocab) for r in records}


def projection_preflight(package,out,manifest,records,scenes,vocab):
    from PIL import Image
    samples=samples_for(package,manifest,records,vocab)
    diagnostics=[]; previews={}
    for i,record in enumerate(records):
        token=record['token']; sample=samples[token]
        try:
            row,_,_=view.payload(package,scenes[token],manifest,sample,'overlay','canonical')
            path=base.camera_paths(package,scenes[token],manifest)[4]
            with Image.open(path) as image:
                calib=view.calibration(scenes[token],image.size)
                radius=view.lens_domain(tuple(calib[3]))
                candidates=[]
                for pid,poses in enumerate(sample['trajectories'][:4]):
                    legacy,_,trace=projection_audit.inspect_path(poses,calib,image.size)
                    segments,valid=view.segment_details(poses,calib,image.size)
                    if (len(valid)!=len(trace['suspect']) or not np.array_equal(
                            view.dense_path(poses),projection_audit.dense_path(poses)[0])):
                        raise RuntimeError('Dense point alignment changed')
                    retained=sum(any(trace['suspect'][j] for j in segment['point_indices']) for segment in segments)
                    if retained: raise RuntimeError('V2 retained a detected distortion foldback')
                    candidates.append(dict(candidate_name=view.ids(token)[pid],teacher_rank=pid+1,
                        candidate_vocab=sample['pool_vocab'][pid],path_length_m=legacy['path_length_m'],
                        endpoint_xy_m=legacy['endpoint_xy_m'],
                        legacy_visible_segments=legacy['original_visible_segments'],
                        legacy_suspect_visible_segments=legacy['suspect_visible_segments'],
                        corrected_visible_segments=len(segments),retained_legacy_suspect_segments=retained,
                        rejected_dense_points=int((~valid).sum())))
        except Exception as exc:
            write(out/'calibration_error.json',dict(token=token,error=repr(exc),
                camera=scenes[token].get('current_camera_calibrations',{}).get('cam_f0')))
            raise RuntimeError(f'V2 projection preflight token={token}: {exc}') from exc
        diagnostics.append(dict(token=token,central_domain_radius=None if np.isinf(radius) else float(radius),
            translation=calib[1].tolist(),distortion=calib[3].tolist(),candidates=candidates))
        if i<4:
            for j,image in enumerate(row['images']):
                filename=f'preview-{token}-{j}.png'
                buffer=io.BytesIO(); image.save(buffer,format='PNG'); content=buffer.getvalue()
                target=out/filename
                if target.exists() and target.read_bytes()!=content:
                    raise RuntimeError('V2 frozen preview changed: '+filename)
                if not target.exists():
                    tmp=target.with_suffix('.png.tmp'); tmp.write_bytes(content); tmp.replace(target)
                previews[filename]=hashlib.sha256(content).hexdigest()
    total=lambda key:sum(c[key] for r in diagnostics for c in r['candidates'])
    report=dict(status='render_contract_passed',scenes=len(records),candidates=4*len(records),
        legacy_scenes_with_suspect_segments=sum(any(c['legacy_suspect_visible_segments'] for c in r['candidates']) for r in diagnostics),
        legacy_suspect_visible_segments=total('legacy_suspect_visible_segments'),
        retained_legacy_suspect_segments=total('retained_legacy_suspect_segments'),
        corrected_visible_segments=total('corrected_visible_segments'),
        scenes_without_corrected_projection=sum(all(c['corrected_visible_segments']==0 for c in r['candidates']) for r in diagnostics),
        outcomes_read=False,visual_alignment_verified=False,
        note='Numerical render checks passed; physical road alignment still requires preview inspection. '+view.NOTE)
    base.freeze(out/'projection_diagnostics.json',diagnostics)
    base.freeze(out/'projection_report.json',report)
    files={n:sha(out/n) for n in ('projection_diagnostics.json','projection_report.json')}
    base.freeze(out/'projection_commit.json',dict(plan_sha256=sha(out/'plan.json'),files=dict(files,**previews)))
    return report


def infer_subset(package,out,manifest,records,scenes,samples,infer,max_new=None):
    expected={name(r['token'],m,o) for r in records for m,o in view.TASKS}; done={}; new=0; started=time.monotonic()
    if {p.name for p in (out/'results').glob('*.json')}-expected: raise RuntimeError('Unexpected overlay results')
    for r in records:
        token=r['token']
        for mode,kind in view.TASKS:
            path=out/'results'/name(token,mode,kind)
            if not path.exists() and ((max_new is not None and new>=max_new) or time.monotonic()-started>3*3600): continue
            row,order,h=view.payload(package,scenes[token],manifest,samples[token],mode,kind)
            if path.exists():
                result=load_result(path,h)
                if (result['token'],result['mode'],result['ordering'],result['pool_positions'])!=(token,mode,kind,order):
                    raise RuntimeError('Cached overlay candidate mapping changed')
            else:
                result=dict(status='ok',k=4,token=token,mode=mode,ordering=kind,pool_positions=order,request_sha256=h)
                try:
                    measured=infer(row); p=[float(x) for x in measured['probabilities']]; validate_probabilities(p,4)
                    result.update(measured,probabilities=p)
                except Exception as exc:
                    result.update(status='technical_failure',error=repr(exc),traceback=traceback.format_exc())
                    save_result(path,result); raise
                save_result(path,result); new+=1
                print(f'OVERLAY {new} new token={token} {mode}/{kind}',flush=True)
            done[path.name]=sha(path)
    status=dict(status='complete' if set(done)==expected else 'partial',completed_requests=len(done),
                expected_requests=len(expected),new_requests=new,outcomes_read=False)
    write(out/'inference_status.json',status)
    if status['status']=='complete': base.freeze(out/'worker_commit.json',dict(results=done))
    return status


def build_decisions(records,samples,results):
    decisions=[]
    expected={name(r['token'],m,o) for r in records for m,o in view.TASKS}
    if set(results)!=expected: raise ValueError('Entire overlay cohort required before labels')
    for r in records:
        token=r['token']; sample=samples[token]; methods={'baseline':sample['baseline_local']}
        for mode in view.MODES:
            vectors=[]
            for kind in view.ORDERS:
                row=results[name(token,mode,kind)]; order=view.order(token,kind)
                if row['pool_positions']!=order: raise ValueError('Overlay order alignment failed')
                validate_probabilities(row['probabilities'],4)
                p=np.empty(4); p[order]=row['probabilities']; vectors.append(p)
            small=dict(pool_local=sample['pool_local'][:4],baseline_local=sample['baseline_local'])
            methods[mode+'_mean3']=old.choice(np.mean(vectors,axis=0),small)
            if mode=='overlay': methods['overlay_single']=old.choice(vectors[0],small)
        j=int(base.digest([base.SEED,token,'random_top4'])[:16],16)%4
        methods['random_top4']=sample['pool_local'][j]
        decisions.append(dict(metadata=r,pool_local=sample['pool_local'],pool_vocab=sample['pool_vocab'],
            baseline_local=sample['baseline_local'],methods={m:dict(candidate_local=i,
            candidate_vocab=sample['pool_vocab'][sample['pool_local'].index(i)]) for m,i in methods.items()}))
    return decisions


def score(package,out,manifest,records,samples):
    commit=read(out/'decision_commit.json')
    if commit['decisions_sha256']!=sha(out/'decisions.json') or commit['plan_sha256']!=sha(out/'plan.json'):
        raise RuntimeError('Overlay decision freeze changed')
    results={}
    for n,h in commit['results'].items():
        p=out/'results'/n
        if sha(p)!=h: raise RuntimeError('Committed overlay result changed')
        results[n]=load_result(p)
        if name(results[n]['token'],results[n]['mode'],results[n]['ordering'])!=n:
            raise RuntimeError('Result scene identity mismatch')
    if {p.name for p in (out/'results').glob('*.json')}!=set(commit['results']):
        raise RuntimeError('Unexpected central result files')
    decisions=build_decisions(records,samples,results)
    if decisions!=read(out/'decisions.json'): raise RuntimeError('Overlay choices disagree with inference')
    # Only here, after ALL conditions/scenes freeze, may outcome arrays be deserialized.
    labels={r['token']:old.load_labels(package,r['token'],manifest,samples[r['token']]) for r in records}
    for d in decisions:
        raw=labels[d['metadata']['token']]; j=int(np.argmax(raw[:4,0]))
        d['methods']['oracle_top4_upper_bound']=dict(candidate_local=d['pool_local'][j],candidate_vocab=d['pool_vocab'][j])
    rows=old.score_decisions(decisions,labels)
    methods={}
    for m in sorted({r['method'] for r in rows}):
        group=[r for r in rows if r['method']==m]
        sids=sorted({r['session_id'] for r in group})
        gains=np.array([np.mean([r['raw_gain_pp'] for r in group if r['session_id']==s]) for s in sids])
        rng=np.random.default_rng(base.SEED); boot=gains[rng.integers(0,len(gains),(2000,len(gains)))].mean(axis=1)
        methods[m]=dict(raw_pdm_mean=float(np.mean([r['raw_pdm'] for r in group])),gain_macro_pp=float(gains.mean()),
            cluster_ci95_pp=np.quantile(boot,[.025,.975]).tolist(),interventions=sum(r['intervention'] for r in group),
            pdm_decreased=sum(r['pdm_decreased'] for r in group),relative_component_harm=sum(r['raw_harm'] for r in group),
            uses_outcomes_for_choice=m.startswith('oracle_'))
    contrasts={}
    for control in ('blank_mean3','separate_mean3'):
        contrast=paired_contrast(rows,'overlay_mean3',control)
        contrasts['overlay_mean3_minus_'+control]={k:contrast[k] for k in ('mean_pp','cluster_ci95_pp')}
    report=dict(status='complete',primary='overlay_mean3',scenes=len(records),requests=len(results),methods=methods,
                paired_contrasts=contrasts,projection=read(out/'projection_report.json'),note=view.NOTE)
    write(out/'per_scene.json',rows); write(out/'report.json',report)
    return report


def gpu_worker(root,package,out,manifest,records,scenes,samples,max_new):
    import torch
    from drex_max.model import DecisionModel
    envroot=out/'runtime_check'; envroot.mkdir(parents=True,exist_ok=True)
    lock=envroot/'requirements.lock.txt'
    if lock.exists() and sha(lock)!=sha(root/'requirements.lock.txt'): raise RuntimeError('Environment lock changed')
    if not lock.exists(): shutil.copyfile(root/'requirements.lock.txt',lock)
    env=environment(envroot,True)
    base.freeze(out/'environment.json',{k:env[k] for k in ('python','torch','cuda_build','transformers','gpu','vram_bytes')})
    target=root/'models/drex-max-v1'
    if remaining_bytes(target,read(root/'model_plan.json')['files']): raise RuntimeError('Missing model files')
    cfg=read(target/'decision_config.json')
    if cfg.get('format_version')!=1 or cfg.get('attention_mode','noncausal_full_attention')!='noncausal_full_attention':
        raise RuntimeError('Unexpected model protocol')
    if torch.cuda.device_count()!=1: raise RuntimeError('Each Slurm worker must see one GPU')
    torch.cuda.reset_peak_memory_stats()
    model=DecisionModel(target,device='cuda:0',max_image_pixels=view.CONFIG['max_image_pixels'])
    if model.readout.weight.dtype!=torch.bfloat16: raise RuntimeError('Expected BF16')
    def infer(row):
        batch=model.prepare([row],max_length=view.CONFIG['max_length'])
        grid=batch.inputs.get('image_grid_thw'); tokens=int(batch.inputs['input_ids'].shape[1])
        if grid is None or grid.shape[0]!=5 or tokens>=view.CONFIG['max_length']: raise RuntimeError('Image count/context mismatch')
        del batch
        torch.cuda.synchronize(); t=time.monotonic()
        p=model.predict([row],batch_size=1,max_length=view.CONFIG['max_length'])[0]
        torch.cuda.synchronize()
        return dict(probabilities=p,input_tokens=tokens,seconds=time.monotonic()-t)
    status=infer_subset(package,out,manifest,records,scenes,samples,infer,max_new)
    status.update(peak_allocated_bytes=int(torch.cuda.max_memory_allocated()))
    write(out/'inference_status.json',status)
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('plan','worker','merge'))
    p.add_argument('--root',type=Path,required=True); p.add_argument('--replicas',type=int,choices=(1,2,4,8),default=2)
    p.add_argument('--index',type=int); p.add_argument('--pilot',action='store_true')
    a=p.parse_args(); root=a.root.expanduser().resolve()
    tag=os.environ.get('DREX_OVERLAY_V2_NAME',f'overlay_v2_top4_gpu{a.replicas}_v1')
    if not re.fullmatch(r'overlay_v2_[A-Za-z0-9_-]{1,64}',tag): raise ValueError('Use a separate overlay_v2_* output')
    out=root/'runs'/tag; package=root/'data/navsim_train_v1/package'
    try:
        if a.stage=='plan':
            plan,manifest,records,scenes,vocab=prepare(root,package,out,a.replicas)
            projection=projection_preflight(package,out,manifest,records,scenes,vocab)
            result=dict(status='planned',scenes=len(records),requests=len(records)*9,replicas=a.replicas,
                        output=str(out),outcomes_read=False,projection=projection,note=view.NOTE)
            print(json.dumps(result,indent=2),flush=True); return
        plan=check_plan(out)
        # Isolated metadata checks in each worker, never concurrent writes to the central freeze.
        shard=out/'shards'/f'{a.index:02d}' if a.stage=='worker' and a.index is not None else out
        manifest,records,scenes,vocab=base.prepare(package,root,shard/'base')
        if sha(shard/'base/identity.json')!=plan['base_identity_sha256']: raise RuntimeError('Worker input identity mismatch')
        if a.stage=='worker':
            if a.index is None or not 0<=a.index<plan['replicas']: raise ValueError('Invalid worker')
            wanted=set(plan['assignments'][a.index]); subset=[r for r in records if r['token'] in wanted]
            samples=samples_for(package,manifest,subset,vocab)
            if (shard/'worker_commit.json').exists():
                def forbidden(_): raise AssertionError('Committed worker must not run inference')
                status=infer_subset(package,shard,manifest,subset,scenes,samples,forbidden,max_new=0)
            else:
                status=gpu_worker(root,package,shard,manifest,subset,scenes,samples,18 if a.pilot else None)
            if status['status']!='complete' and not a.pilot: raise RuntimeError('Partial worker; resume before merge')
            print(json.dumps(status),flush=True); return
        samples=samples_for(package,manifest,records,vocab); results={}; profiles=[]
        for i in range(plan['replicas']):
            worker_out=out/'shards'/f'{i:02d}'; proof=read(worker_out/'worker_commit.json')
            if sha(worker_out/'base/identity.json')!=plan['base_identity_sha256']:
                raise RuntimeError('Worker base identity changed')
            profiles.append(read(worker_out/'environment.json'))
            expected={name(t,m,o) for t in plan['assignments'][i] for m,o in view.TASKS}
            if set(proof['results'])!=expected or {p.name for p in (worker_out/'results').glob('*.json')}!=expected:
                raise RuntimeError('Incomplete/extra worker results')
            for n,h in proof['results'].items():
                source=worker_out/'results'/n
                if n in results or sha(source)!=h: raise RuntimeError('Duplicate/changed worker result')
                result=load_result(source)
                if name(result['token'],result['mode'],result['ordering'])!=n: raise RuntimeError('Worker scene identity mismatch')
                row,order,query=view.payload(package,scenes[result['token']],manifest,samples[result['token']],result['mode'],result['ordering'])
                if query!=result['request_sha256'] or order!=result['pool_positions']: raise RuntimeError('Frozen query changed')
                results[n]=result
        if any(p!=profiles[0] for p in profiles): raise RuntimeError('Different replica profiles')
        decisions=build_decisions(records,samples,results)
        for i in range(plan['replicas']):
            for source in (out/'shards'/f'{i:02d}'/'results').glob('*.json'):
                target=out/'results'/source.name; target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists() and sha(target)!=sha(source): raise RuntimeError('Central result changed')
                if not target.exists():
                    tmp=target.with_suffix('.json.tmp'); shutil.copyfile(source,tmp); tmp.replace(target)
        base.freeze(out/'decisions.json',decisions)
        base.freeze(out/'decision_commit.json',dict(plan_sha256=sha(out/'plan.json'),decisions_sha256=sha(out/'decisions.json'),
            results={n:sha(out/'results'/n) for n in sorted(results)}))
        print(json.dumps(score(package,out,manifest,records,samples),indent=2),flush=True)
    except Exception as exc:
        target=out/'shards'/f'{a.index:02d}' if a.stage=='worker' and a.index is not None else out
        write(target/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc())); raise


if __name__=='__main__': main()
