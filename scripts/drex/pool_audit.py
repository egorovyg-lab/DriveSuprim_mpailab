"""Freeze the entire original top64 and a verified continuous-audit reuse source."""
import argparse
import json
from pathlib import Path
import re

import continuous_audit as previous
import pool_transport as transport
import rollout_bridge
from transfer import read, write, sha, freeze

TOOLS = tuple(dict.fromkeys(('pool_remote.py', 'pool_reserve.py', *previous.TOOLS)))


def reuse_snapshot(root, name):
    if not re.fullmatch(r'rollout_audit_[A-Za-z0-9_-]{1,64}', name): raise ValueError('Invalid reuse name')
    source = root/'runs'/name; pointer = read(source/'latest_result.json')
    digest = pointer['archive_sha256']
    if not re.fullmatch('[0-9a-f]{64}', digest): raise ValueError('Invalid archive hash')
    archive = source/('audit-'+digest+'.tar')
    if pointer['remote_exit_code'] != 0 or sha(archive) != digest: raise RuntimeError('Verified completed reuse archive required')
    folder = source/('result_'+digest[:12])
    if Path(pointer['result']).resolve() != folder.resolve(): raise RuntimeError('Unexpected reuse artifact directory')
    report = rollout_bridge.extract_result(archive, folder); identity = read(folder/'identity.json')
    request = read(source/'audit_request.json')
    if (report['status'] != 'complete' or report['required_candidates'] != report['metric_parity_passed']
            or len(request['candidates']) != report['required_candidates']
            or sha(source/'audit_request.json') != identity['request_sha256']):
        raise RuntimeError('Complete unchanged original continuous audit required')
    commits = {}; identity_sha = sha(folder/'identity.json')
    for c in request['candidates']:
        key = c['token']+'-'+str(c['candidate_local']); directory = folder/'candidates'/key
        commit = read(directory/'commit.json')
        if commit['candidate'] != c or commit['identity_sha256'] != identity_sha:
            raise RuntimeError('Reuse candidate binding changed')
        for n, h in commit['files'].items():
            if sha(directory/n) != h: raise RuntimeError('Reuse file changed')
        commits[key] = sha(directory/'commit.json')
    return request, dict(name=name, identity_sha256=sha(folder/'identity.json'),
        request_sha256=sha(source/'audit_request.json'), report_sha256=sha(folder/'report.json'), candidate_commits=commits)


def assemble(identity, samples, labels, manifest, old_request, reuse):
    import pool_reserve as spec
    import continuous_metrics as cm
    import real_data as base
    records, candidates = [], []
    old = {r['token']:r for r in old_request['records']}
    if {r['token'] for r in identity['records']} != set(old): raise RuntimeError('Reuse/full-pool cohorts differ')
    for r in identity['records']:
        token = r['token']; sample = samples[token]
        if r['session_id'] != old[token]['session_id'] or sample['baseline_local'] != old[token]['methods']['baseline']:
            raise RuntimeError('Source session or DS changed')
        records.append(dict(token=token, session_id=r['session_id'], baseline_local=sample['baseline_local']))
        for rank, c in enumerate(sample['candidates'][:64], 1):
            candidates.append(dict(token=token, candidate_local=c['local'], candidate_vocab=c['vocab'], teacher_rank=rank,
                metric_keys=['pdm_score', *base.METRICS], expected_raw=labels[token][c['local']].tolist(),
                example_sha256=manifest['files']['bundle/examples/'+token+'.npz']['sha256']))
    request = dict(protocol='continuous_pool_v1', config=spec.CONFIG, metric_config=cm.CONFIG,
        records=records, candidates=candidates, pilot_tokens=old_request['pilot_tokens'], reuse=reuse,
        package_manifest_sha256=identity['package_manifest_sha256'], remote_package=old_request['remote_package'],
        vehicle_profile=old_request['vehicle_profile'], note=spec.NOTE)
    lookup = {c['token']+'-'+str(c['candidate_local']):c for c in candidates}
    for c in old_request['candidates']:
        if lookup.get(c['token']+'-'+str(c['candidate_local'])) != c: raise RuntimeError('Original labels/candidate IDs changed')
    spec.validate(request); return request


def prepare(args):
    import audit_table_cases as audit
    import real_data as base
    out = args.root/'runs'/args.name
    old, reuse = reuse_snapshot(args.root, args.reuse)
    identity, _, _, _, _, labels, samples, _ = audit.validate_source(args.root/'runs'/args.source)
    manifest = read(Path(identity['package'])/'package_manifest.json')
    request = assemble(identity, samples, labels, manifest, old, reuse)
    if request['remote_package'] != args.remote_package: raise RuntimeError('Remote package differs from reuse source')
    request.update(source=str(args.root/'runs'/args.source),
        source_commit_sha256=sha(args.root/'runs'/args.source/'decision_commit.json'),
        tools={n:sha(Path(__file__).with_name(n)) for n in TOOLS})
    freeze(out/'audit_request.json', request)
    plan = dict(status='planned', scenes=len(request['records']), candidates=len(request['candidates']),
        reusable_candidates=len(reuse['candidate_commits']),
        maximum_new_scorer_calls=len(request['candidates'])-len(reuse['candidate_commits']),
        request_sha256=sha(out/'audit_request.json'), model_calls=0, primary=request['config']['primary'])
    freeze(out/'plan.json', plan); return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'pilot', 'run'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--name', default='rollout_audit_pool64_v1'); p.add_argument('--source', default='table_topk_night_gpu8_v1')
    p.add_argument('--reuse', default='rollout_audit_continuous_v1')
    p.add_argument('--host', default='egorov@93.180.62.99'); p.add_argument('--port', type=int, default=27434)
    p.add_argument('--container', default='egorov_drivesuprim'); p.add_argument('--remote-repo', default='/workspace/DriveSuprim')
    p.add_argument('--control', type=Path, default=Path.home()/'.ssh/drex-astra-transfer.sock')
    p.add_argument('--remote-package', default='/workspace/exp_v2/drex_transfer/navsim_train_v1')
    a = p.parse_args(); a.root = a.root.expanduser().resolve()
    if (not re.fullmatch(r'rollout_audit_[A-Za-z0-9_-]{1,64}', a.name) or a.name == a.reuse
            or not re.fullmatch(r'table_[A-Za-z0-9_-]{1,80}', a.source)):
        raise ValueError('Use a separate pool output and original table source')
    if a.stage == 'prepare': print(json.dumps(prepare(a), indent=2)); return 0
    out = a.root/'runs'/a.name; request = read(out/'audit_request.json')
    if request['source'] != str(a.root/'runs'/a.source) or request['reuse']['name'] != a.reuse or request['remote_package'] != a.remote_package:
        raise RuntimeError('Frozen source changed')
    for n, h in request['tools'].items():
        if sha(Path(__file__).with_name(n)) != h: raise RuntimeError('Frozen pool tool changed: '+n)
    if a.stage == 'run':
        pointer = read(out/'latest_result.json'); archive = out/('audit-'+pointer['archive_sha256']+'.tar')
        if pointer['remote_exit_code'] != 0 or sha(archive) != pointer['archive_sha256']: raise RuntimeError('Verified pilot required')
        prior = rollout_bridge.extract_result(archive, Path(pointer['result']))
        if (prior['status'] not in ('pilot_complete', 'complete')
                or read(Path(pointer['result'])/'identity.json')['request_sha256'] != sha(out/'audit_request.json')):
            raise RuntimeError('Matching successful pool pilot required')
    return transport.execute(a, TOOLS, 'pilot' if a.stage == 'pilot' else 'full')


if __name__ == '__main__': raise SystemExit(main())
