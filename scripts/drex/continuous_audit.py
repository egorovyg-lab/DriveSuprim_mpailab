"""Freeze DS/Drex choices on msu270 and bridge a resumable CPU-only Astra audit."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import clearance_bridge as transport
import rollout_bridge
from transfer import read, sha, freeze

TOOLS = tuple(dict.fromkeys(('continuous_remote.py', 'continuous_pilot.py', 'continuous_metrics.py',
    'continuous_report.py', 'clearance_math.py', *rollout_bridge.TOOLS)))


def assemble(decisions, old_decisions, labels, samples, manifest):
    import continuous_metrics as metrics
    import real_data as base
    old = {d['metadata']['token']: d for d in old_decisions if d['k'] == 64}
    records, candidates = [], []
    for d in decisions:
        token = d['metadata']['token']; sample = samples[token]; b = sample['baseline_local']
        methods = dict(d['methods'], baseline=b, old_table=old[token]['methods']['table/mean3'],
                       old_camera=old[token]['methods']['table_camera/mean3'])
        selected = {m: methods[m] for m in metrics.METHODS}
        records.append(dict(token=token, session_id=d['metadata']['session_id'], methods=selected))
        for j in sorted(set(selected.values())):
            rank, c = next((i+1, c) for i, c in enumerate(sample['candidates']) if c['local'] == j)
            candidates.append(dict(token=token, candidate_local=j, candidate_vocab=c['vocab'], teacher_rank=rank,
                metric_keys=['pdm_score', *base.METRICS], expected_raw=labels[token][j].tolist(),
                example_sha256=manifest['files']['bundle/examples/'+token+'.npz']['sha256']))
    # Outcome-independent pilot: hash-ordered sessions, then a hash-ordered scene per session.
    order = lambda s: hashlib.sha256(('continuous-v1:'+s).encode()).hexdigest()
    sessions = sorted({r['session_id'] for r in records}, key=order)
    pilot = [min((r['token'] for r in records if r['session_id'] == s), key=order) for s in sessions[:4]]
    return dict(protocol='continuous_decisions_v1', config=metrics.CONFIG, methods=list(metrics.METHODS),
                records=sorted(records, key=lambda r: r['token']), candidates=candidates, pilot_tokens=pilot)


def prepare(source, out, remote_package):
    import replacement
    import audit_table_cases as old_audit
    import continuous_metrics as metrics
    # Re-derive every replacement from committed results BEFORE loading measured outcomes.
    identity, _, decisions, _ = replacement.collect(source)
    report = read(source/'report.json')
    if (report['status'] != 'complete' or report['decision_commit_sha256'] != sha(source/'decision_commit.json')
            or report['scenes'] != len(decisions)):
        raise RuntimeError('Completed, unchanged replacement report required')
    original = Path(identity['source'])
    old_id, _, _, old_decisions, _, labels, samples, _ = old_audit.validate_source(original)
    package = Path(old_id['package']); manifest = read(package/'package_manifest.json')
    request = assemble(decisions, old_decisions, labels, samples, manifest)
    request.update(package_manifest_sha256=old_id['package_manifest_sha256'], remote_package=remote_package,
        source=str(source), source_decision_commit_sha256=sha(source/'decision_commit.json'),
        source_report_sha256=sha(source/'report.json'), original_decision_commit_sha256=sha(original/'decision_commit.json'),
        vehicle_profile=read(Path(__file__).with_name('vehicle_profile_astra.json')),
        tools={n: sha(Path(__file__).with_name(n)) for n in TOOLS}, note=metrics.NOTE)
    metrics.validate_request(request); freeze(out/'audit_request.json', request)
    plan = dict(status='planned', scenes=len(request['records']), candidates=len(request['candidates']),
        methods=list(metrics.METHODS), pilot_tokens=request['pilot_tokens'], model_calls=0,
        maximum_new_scorer_calls=len(request['candidates']), request_sha256=sha(out/'audit_request.json'),
        note='Only the union of frozen decisions is replayed, not all top64. Existing outcomes do not select the pilot.')
    freeze(out/'plan.json', plan); return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'pilot', 'run'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--name', default='rollout_audit_continuous_v1')
    p.add_argument('--source', default='replacement_gpu2_v1')
    p.add_argument('--host', default='egorov@93.180.62.99'); p.add_argument('--port', type=int, default=27434)
    p.add_argument('--container', default='egorov_drivesuprim')
    p.add_argument('--control', type=Path, default=Path.home()/'.ssh/drex-astra-transfer.sock')
    p.add_argument('--remote-repo', default='/workspace/DriveSuprim')
    p.add_argument('--remote-package', default='/workspace/exp_v2/drex_transfer/navsim_train_v1')
    a = p.parse_args(); a.root = a.root.expanduser().resolve()
    if (not re.fullmatch(r'rollout_audit_[A-Za-z0-9_-]{1,64}', a.name)
            or not re.fullmatch(r'replacement_[A-Za-z0-9_-]{1,64}', a.source)):
        raise ValueError('Invalid dedicated audit/source name')
    out = a.root/'runs'/a.name
    if a.stage == 'prepare':
        print(json.dumps(prepare(a.root/'runs'/a.source, out, a.remote_package), indent=2)); return 0
    request = read(out/'audit_request.json')
    if request['remote_package'] != a.remote_package or request['source'] != str(a.root/'runs'/a.source):
        raise RuntimeError('Frozen source/package changed')
    for name, digest in request['tools'].items():
        if sha(Path(__file__).with_name(name)) != digest: raise RuntimeError('Frozen audit tool changed: '+name)
    if a.stage == 'run':
        pointer = read(out/'latest_result.json'); result = Path(pointer['result'])
        # Recheck the checksum-verified pilot archive instead of trusting a mutable status alone.
        archive = out/('audit-'+pointer['archive_sha256']+'.tar')
        if sha(archive) != pointer['archive_sha256']: raise RuntimeError('Pilot archive changed')
        prior = rollout_bridge.extract_result(archive, result)
        if (pointer['remote_exit_code'] != 0 or prior['status'] not in ('pilot_complete', 'complete')
                or not prior['pilot_ready'] or read(result/'identity.json')['request_sha256'] != sha(out/'audit_request.json')):
            raise RuntimeError('Successful, matching CPU pilot required before full replay')
    return transport.run(a, request=request, tools=TOOLS,
        remote_tool='continuous_pilot.py' if a.stage == 'pilot' else 'continuous_remote.py')


if __name__ == '__main__': raise SystemExit(main())
