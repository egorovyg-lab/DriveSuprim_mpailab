"""Astra-only single-candidate raw scoring of a frozen table experiment pool."""
import argparse
import json
import lzma
import os
from pathlib import Path
import pickle
import sys
import traceback

import numpy as np

import real_data as base
import table_data as tables
import rollout_replay as replay
from transfer import read, write, sha, freeze


def frozen_groups(request,package,manifest,vocab):
    if request['protocol']!='table_top64_labels_v1' or request['metrics']!=['pdm_score',*base.METRICS]:
        raise ValueError('Unexpected label request protocol/metrics')
    candidates = tables.requested_candidates(request)
    keys = [(c['token'],c['candidate_local']) for c in candidates]
    if not candidates or len(keys)!=len(set(keys)): raise ValueError('Empty/duplicate candidates')
    groups = {}
    for c in candidates: groups.setdefault(c['token'],[]).append(c)
    for token,group in groups.items():
        sample = tables.load_inputs(package,token,manifest,vocab)
        expected = [dict(token=token,candidate_local=c['local'],candidate_vocab=c['vocab']) for c in sample['candidates']]
        if group!=expected: raise RuntimeError('Request is not the frozen input-only top64 pool')
    return groups


def run(request_path,out,repo,package):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    request = read(request_path)
    if request['protocol']!='table_top64_labels_v1' or request['metrics']!=['pdm_score',*base.METRICS]:
        raise ValueError('Unexpected label request protocol/metrics')
    if sha(package/'package_manifest.json')!=request['package_manifest_sha256']:
        raise RuntimeError('Astra/msu270 portable packages differ')
    candidates = tables.requested_candidates(request)
    keys = [(c['token'],c['candidate_local']) for c in candidates]
    if not candidates or len(keys)!=len(set(keys)): raise ValueError('Empty/duplicate candidates')
    manifest,scenes,vocab,template,caches = replay.bound_context(package,repo,candidates)
    if template.get('map_root'): os.environ['NUPLAN_MAPS_ROOT'] = template['map_root']
    cfg = OmegaConf.create(replay.config_core(template))
    if cfg.traffic_agents!='non_reactive': raise ValueError('Requires original non-reactive protocol')
    simulator,scorer = instantiate(cfg.simulator),instantiate(cfg.scorer)
    scorer._config.human_penalty_filter = False
    policy = instantiate(cfg.traffic_agents_policy.non_reactive,simulator.proposal_sampling)
    sampling = instantiate(cfg.agent_sampling)
    # Validate the complete requested pool before any measurement lookup or scorer call.
    groups = frozen_groups(request,package,manifest,vocab)
    freeze(out/'identity.json',dict(request_sha256=sha(request_path),caches=caches,
        human_penalty_filter=False,score_protocol='single_candidate_raw_v1',
        config=replay.config_core(template),scorer_sources={n:sha(repo/n) for n in replay.REPLAY_SOURCES},
        tools={p.name:sha(p) for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.json')}))
    rows = []; calls = 0
    for i,(token,group) in enumerate(groups.items(),1):
        target = out/'scenes'/(token+'.json')
        if target.exists():
            done = read(target)
            if done['checksum']!=base.digest(done['rows']): raise RuntimeError('Completed labels changed')
            measured = done['rows']
        else:
            relative = 'bundle/examples/'+token+'.npz'
            if sha(package/relative)!=manifest['files'][relative]['sha256']: raise RuntimeError('Changed example')
            with np.load(package/relative,allow_pickle=False) as f:
                old = {int(j):v for j,v in zip(f['label_indices'],f['label_raw'])}
            cache_info = caches[token]
            if sha(Path(cache_info['path']))!=cache_info['sha256']: raise RuntimeError('Changed metric cache')
            with lzma.open(cache_info['path'],'rb') as f: cache_bytes = f.read()
            measured = []
            for c in group:
                cache = pickle.loads(cache_bytes)  # Fresh observation for EACH candidate.
                if cache.log_name!=scenes[token]['log_name']: raise RuntimeError('Cache/camera log mismatch')
                trajectory = Trajectory(vocab[c['candidate_vocab']].astype(float),sampling)
                frame,_ = pdm_score(cache,trajectory,simulator.proposal_sampling,simulator,scorer,policy)
                raw = np.asarray([float(frame.iloc[0][m]) for m in request['metrics']])
                if not np.isfinite(raw).all() or np.any((raw<0)|(raw>1)): raise RuntimeError('Invalid raw scorer output')
                if c['candidate_local'] in old and not np.allclose(raw,old[c['candidate_local']],atol=1e-8,rtol=1e-7):
                    raise RuntimeError('Original top16 raw metric parity failed: '+token)
                measured.append(dict(c,raw=raw.tolist())); calls += 1
            freeze(target,dict(rows=measured,checksum=base.digest(measured)))
        if [(r['token'],r['candidate_local'],r['candidate_vocab']) for r in measured] != [
                (r['token'],r['candidate_local'],r['candidate_vocab']) for r in group]:
            raise RuntimeError('Resumed label alignment changed')
        rows.extend(measured)
        write(out/'progress.json',dict(status='running',completed_scenes=i,total_scenes=len(groups),
            scorer_calls_this_invocation=calls))
        print(f'TABLE LABELS {i}/{len(groups)} {token}',flush=True)
    labels = dict(status='complete',request_digest=base.digest(request),metrics=request['metrics'],rows=rows)
    freeze(out/'labels.json',labels)
    result = dict(status='complete',scenes=len(groups),candidates=len(rows),
        original_top16_parity_checked=len(groups)*16,scorer_calls_this_invocation=calls,
        labels_sha256=sha(out/'labels.json'),note='Raw local PDM, not EPDMS. CPU label collection only; no Drex, training or holdout.')
    write(out/'report.json',result)
    return result


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--package',type=Path,default=Path('/workspace/exp_v2/drex_transfer/navsim_train_v1'))
    a = p.parse_args(); sys.path.insert(0,str(a.repo))
    import fcntl
    a.out.mkdir(parents=True,exist_ok=True)
    score_lock = (a.out/'.scoring.lock').open('a')
    try: fcntl.flock(score_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('Another CPU scorer is still running for this request',flush=True)
        raise SystemExit(2)
    try: print(json.dumps(run(a.input,a.out,a.repo,a.package),indent=2),flush=True)
    except Exception as exc:
        write(a.out/'report.json',dict(status='technical_failure',error=repr(exc),traceback=traceback.format_exc()))
        raise
