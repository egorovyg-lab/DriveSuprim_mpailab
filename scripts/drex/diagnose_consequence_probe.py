"""Read-only explanation of a completed consequence probe; standard library only."""
import argparse
from array import array
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

TARGETS = ('jerk_gain_mps3', 'lateral_gain_mps2', 'progress_delta_m', 'pdm_gain_pp',
           'jerk_peak_delta_mps3', 'harm', 'benefit')
GATES = ('benefit', 'risk', 'comfort', 'jerk_peak', 'progress')
NOTE = ('Read-only, post-hoc diagnosis of the original fixed policy. No fitting, new inference, '
        'scoring, threshold selection or policy changes. Standalone/cumulative gate counts and '
        'single-blocker counts explain different intersections, not causal gate importance. '
        'Zero interventions reproduce DS; zero observed new harm does not establish safety. '
        'Cached Drex preference information and all possible Drex adaptations are distinct hypotheses.')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def gates(p, cfg):
    clip = lambda v: min(1., max(0., v))
    return (clip(p[6]) > cfg['primary_threshold'], clip(p[5]) <= cfg['risk_ceiling'],
            ((p[0] >= cfg['jerk_gain_min'] and p[1] >= 0.)
             or (p[1] >= cfg['lateral_gain_min'] and p[0] >= 0.)),
            p[4] <= 0., p[2] >= -cfg['progress_loss_m'])


def quantile(sorted_values, q):
    position = (len(sorted_values)-1)*q
    lower = math.floor(position); upper = math.ceil(position)
    return sorted_values[lower] + (sorted_values[upper]-sorted_values[lower])*(position-lower)


def counter():
    return dict(candidates=0, scenes=set(), actual_benefit_candidates=0, actual_benefit_scenes=set())


def observe(count, token, actual_benefit):
    count['candidates'] += 1; count['scenes'].add(token)
    if actual_benefit:
        count['actual_benefit_candidates'] += 1; count['actual_benefit_scenes'].add(token)


def compact(count):
    return {key: len(value) if isinstance(value, set) else value for key, value in count.items()}


def diagnose(folder):
    paths = {n: folder/n for n in ('report.json', 'per_candidate.csv.gz')}
    digests = {n: sha(p) for n, p in paths.items()}
    manifest_path = folder/'artifact_manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        for name, path in paths.items():
            if manifest.get(name) != dict(size=path.stat().st_size, sha256=digests[name]):
                raise RuntimeError('Source artifact checksum differs: '+name)
    r = read(paths['report.json']); cfg = r['config']
    if (r['status'] != 'complete' or r['targets'] != list(TARGETS) or cfg['version'] != 1
            or r['completed_folds'] != [0, 1, 2, 3] or r['held_out_pairs'] != r['scenes']*63):
        raise ValueError('Complete original four-fold probe required')
    methods = set(r['accuracy']) - {'train_mean'}
    if not methods or any(len(n.split('/')) != 2 for n in methods):
        raise ValueError('Unexpected predictor methods')
    groups = {n: dict(seen=set(), values=[array('d') for _ in TARGETS], standalone={g: counter() for g in GATES},
                     cumulative={g: counter() for g in GATES}, single_blocker={g: counter() for g in GATES},
                     all_gates=counter()) for n in methods}
    truth = {}; sessions = {}; ranks = {}
    with gzip.open(paths['per_candidate.csv.gz'], 'rt', encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            name = row['method']
            if name not in groups:
                raise ValueError('Unknown CSV predictor: '+name)
            token = row['token']; key = (token, int(row['candidate_local'])); rank = int(row['teacher_rank'])
            group = groups[name]
            if key in group['seen'] or not 2 <= rank <= 64:
                raise ValueError('Duplicate/invalid candidate row')
            group['seen'].add(key)
            p = tuple(float(row['predicted_'+k]) for k in TARGETS)
            actual = tuple(float(row['actual_'+k]) for k in TARGETS)
            if not all(map(math.isfinite, (*p, *actual))) or actual[5] not in (0., 1.) or actual[6] not in (0., 1.):
                raise ValueError('Nonfinite prediction/invalid binary target')
            bound = (row['session_id'], rank, actual)
            if key in truth and truth[key] != bound:
                raise ValueError('Targets/session/rank differ across predictor arms')
            truth[key] = bound
            if token in sessions and sessions[token] != row['session_id']:
                raise ValueError('Scene session changed')
            sessions[token] = row['session_id']; ranks.setdefault(token, set()).add(rank)
            for values, value in zip(group['values'], p): values.append(value)
            allowed = gates(p, cfg); prefix = True
            for j, gate in enumerate(GATES):
                if allowed[j]: observe(group['standalone'][gate], token, actual[6])
                prefix = prefix and allowed[j]
                if prefix: observe(group['cumulative'][gate], token, actual[6])
                if not allowed[j] and all(v for i, v in enumerate(allowed) if i != j):
                    observe(group['single_blocker'][gate], token, actual[6])
            if all(allowed): observe(group['all_gates'], token, actual[6])
    if (len(truth) != r['held_out_pairs'] or len(sessions) != r['scenes']
            or len(set(sessions.values())) != r['sessions'] or any(v != set(range(2, 65)) for v in ranks.values())):
        raise ValueError('Incomplete top64 query cohort')
    output = {}
    for name, group in sorted(groups.items()):
        if group['seen'] != set(truth):
            raise ValueError('Predictor arm candidate membership differs')
        policy = name+'/t'+str(cfg['primary_threshold'])
        if len(group['all_gates']['scenes']) != r['policies'][policy]['intervention']:
            raise RuntimeError('Gate reproduction differs from frozen policy: '+name)
        distribution = {}
        for target, values in zip(TARGETS, group['values']):
            sorted_values = sorted(values)
            distribution[target] = {q: quantile(sorted_values, v) for q, v in (
                ('min', 0.), ('p05', .05), ('p50', .5), ('p95', .95), ('max', 1.))}
        output[name] = dict(predicted_quantiles=distribution, all_gates=compact(group['all_gates']),
                            **{kind: {gate: compact(value) for gate, value in group[kind].items()}
                               for kind in ('standalone', 'cumulative', 'single_blocker')})
    return dict(status='diagnosis_complete', source_hashes=digests, artifact_manifest_verified=manifest_path.exists(),
                scenes=r['scenes'], sessions=r['sessions'], pairs=len(truth), config=cfg,
                actual_benefit_pairs=sum(v[2][6] for v in truth.values()),
                actual_benefit_scenes=len({k[0] for k, v in truth.items() if v[2][6]}),
                gate_diagnostics=output, accuracy=r['accuracy'], policies=r['policies'],
                continuous_mae_road_minus_control=r['continuous_mae_road_minus_control'], note=NOTE)


def render(d):
    number = lambda v: 'NA' if v is None else f'{v:.5g}'
    print(f"COMPLETE: scenes={d['scenes']} sessions={d['sessions']} pairs={d['pairs']} manifest_verified={d['artifact_manifest_verified']}")
    print(f"ACTUAL BENEFIT: pairs={int(d['actual_benefit_pairs'])} scenes={d['actual_benefit_scenes']} (original composite target)")
    print('OOF QUALITY: method | jerk MAE | lateral MAE | benefit AP | harm AP (session macro)')
    for name, value in d['accuracy'].items():
        print(name, *(number(value[k][metric]) for k, metric in (
            ('jerk_gain_mps3', 'mae_session_macro'), ('lateral_gain_mps2', 'mae_session_macro'),
            ('benefit', 'average_precision_session_macro'), ('harm', 'average_precision_session_macro'))), sep=' | ')
    print('PRIMARY GATES: passing candidates / all pairs; ALL=scenes with any admissible alternative')
    for name, group in d['gate_diagnostics'].items():
        print(name+': '+', '.join(g+'='+str(group['standalone'][g]['candidates']) for g in GATES)
              +f"; ALL={group['all_gates']['scenes']}/{d['scenes']}")
        print('  sole blocker candidates: '+', '.join(g+'='+str(group['single_blocker'][g]['candidates']) for g in GATES))
        print('  raw score p50/p95/max: '+', '.join(k+'='+ '/'.join(number(group['predicted_quantiles'][k][q])
              for q in ('p50', 'p95', 'max')) for k in ('benefit', 'harm', 'jerk_peak_delta_mps3')))
    print('PREDECLARED THRESHOLDS WITH CHANGES: method | interventions | benefits | harms')
    changes = [(name, value) for name, value in d['policies'].items() if value['intervention']]
    if not changes: print('none at any predeclared threshold/family')
    for name, value in changes:
        print(name, value['intervention'], value['actual_benefit'], value['actual_harm'], sep=' | ')
    print('ROAD MINUS CONTROL MAE: negative=lower error; paired session CI95')
    for control, values in d['continuous_mae_road_minus_control'].items():
        for target in ('jerk_gain_mps3', 'lateral_gain_mps2'):
            value = values[target]
            print(control, target, number(value['mean']), '['+', '.join(map(number, value['ci95']))+']', sep=' | ')
    print(NOTE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path, nargs='?'); parser.add_argument('--run', type=Path)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    if bool(args.result) == bool(args.run):
        parser.error('Specify a result directory OR --run with latest_result.json')
    if args.run:
        run = args.run.expanduser().resolve(); pointer = read(run/'latest_result.json'); digest = pointer['archive_sha256']
        if len(digest) != 64 or any(v not in '0123456789abcdef' for v in digest):
            raise ValueError('Invalid source archive hash')
        folder = run/('result_'+digest[:12])
        if (pointer['remote_exit_code'] != 0 or Path(pointer['result']).resolve() != folder
                or sha(run/('audit-'+digest+'.tar')) != digest):
            raise RuntimeError('Expected the SHA-verified fetched result')
    else:
        folder = args.result.expanduser().resolve()
    if args.json:
        target = args.json.expanduser().resolve()
        if target == folder or folder in target.parents or target.exists():
            raise ValueError('Use a new JSON path outside the frozen result directory')
    value = diagnose(folder)
    if args.json:
        with target.open('x', encoding='utf-8') as f: json.dump(value, f, indent=2, allow_nan=False)
    render(value)


if __name__ == '__main__': main()
