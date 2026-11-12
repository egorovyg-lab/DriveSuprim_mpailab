"""Stage the two outcome-selected DS/table pairs for CPU replay on Astra."""
import argparse
from pathlib import Path
import re

import clearance_bridge as transport
import rollout_bridge
from transfer import read, sha, freeze

CASES = (('case01', 'd6e7f58f94d458b4', 'no_at_fault_collisions'),
         ('case02', 'df1898fb8e6953ee', 'lane_keeping'))
TOOLS = tuple(dict.fromkeys(('table_pair_remote.py', *rollout_bridge.TOOLS)))


def candidates_from_decisions(decisions, labels, samples, manifest):
    import real_data as base
    rows = []
    for pair, token, focus in CASES:
        found = [d for d in decisions if d['metadata']['token'] == token and d['k'] == 64]
        if len(found) != 1:
            raise RuntimeError('Missing/duplicate target case '+token)
        d = found[0]; baseline = d['baseline_local']; challenger = d['methods']['table/mean3']
        if challenger == baseline or d['methods']['table_camera/mean3'] != baseline:
            raise RuntimeError('Expected DS-retention/table-switch diagnostic')
        if labels[token][challenger][0] <= labels[token][baseline][0]:
            raise RuntimeError('Expected recorded table gain')
        for member,j in enumerate((baseline, challenger)):
            c = next(c for c in samples[token]['candidates'] if c['local'] == j)
            rows.append(dict(pair_id=pair, token=token, member=member, candidate_local=j,
                candidate_vocab=c['vocab'], focus_metric=focus, better_member=1,
                metric_keys=['pdm_score', *base.METRICS], expected_raw=labels[token][j].tolist(),
                example_sha256=manifest['files']['bundle/examples/'+token+'.npz']['sha256']))
    return rows


def prepare(source, out, remote_package):
    import audit_table_cases as audit
    identity, _, _, decisions, _, labels, samples, _ = audit.validate_source(source)
    package = Path(identity['package']); manifest = read(package/'package_manifest.json')
    request = dict(protocol='table_pair_rollout_v1',
        candidates=candidates_from_decisions(decisions, labels, samples, manifest),
        package_manifest_sha256=identity['package_manifest_sha256'], remote_package=remote_package,
        source_decision_commit_sha256=sha(source/'decision_commit.json'),
        vehicle_profile=read(Path(__file__).with_name('vehicle_profile_astra.json')),
        note='Two outcome-selected TRAIN pairs. Member0=DS/camera, member1=table. '
             'Privileged geometry/traffic are audit-only, never new model inputs.')
    freeze(out/'audit_request.json', request)
    return request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'run'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--name', default='rollout_audit_table_pairs_v1')
    p.add_argument('--source', default='table_topk_night_gpu8_v1')
    p.add_argument('--host', default='egorov@93.180.62.99'); p.add_argument('--port', type=int, default=27434)
    p.add_argument('--container', default='egorov_drivesuprim')
    p.add_argument('--control', type=Path, default=Path.home()/'.ssh/drex-astra-transfer.sock')
    p.add_argument('--remote-repo', default='/workspace/DriveSuprim')
    p.add_argument('--remote-package', default='/workspace/exp_v2/drex_transfer/navsim_train_v1')
    a = p.parse_args(); a.root = a.root.expanduser().resolve()
    if not re.fullmatch(r'rollout_audit_[A-Za-z0-9_-]{1,64}', a.name) or not re.fullmatch(r'table_[A-Za-z0-9_-]{1,80}', a.source):
        raise ValueError('Invalid dedicated output/source name')
    out = a.root/'runs'/a.name
    if a.stage == 'prepare':
        request = prepare(a.root/'runs'/a.source, out, a.remote_package)
        print('Prepared CPU replay: '+str(len(request['candidates']))+' candidates; '+str(out)); return 0
    request = read(out/'audit_request.json')
    if request['remote_package'] != a.remote_package:
        raise RuntimeError('Frozen remote package changed')
    return transport.run(a, request=request, tools=TOOLS, remote_tool='table_pair_remote.py')


if __name__ == '__main__': raise SystemExit(main())
