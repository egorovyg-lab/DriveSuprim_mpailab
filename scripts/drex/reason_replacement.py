"""Controlled offline test of Drex naming a reason to replace DriveSuprim."""

import argparse
from collections import Counter
import csv
import gzip
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import traceback

import numpy as np

from continuous_metrics import paired_stats
from direct256 import COLUMNS, inputs as absolute_inputs
from night import load_result, save_result, validate_probabilities
from real_data import METRICS, SAFETY, digest, package_metadata
from runtime import environment, remaining_bytes
from table_data import cohort
from table_group import gpu_binding
from transfer import checked, freeze, read, sha, write


CONFIG = dict(protocol='reason_replacement_v1', top_k=16, predicted_progress_slack=.01,
              progress_loss_m=1., progress_loss_fraction=.05, minimum_pdm_gain=.005,
              max_tokens=4096, primary='drex_reason')
SOURCES = ('reason_replacement.py', 'run_reason_replacement.sh', 'slurm_options.sh',
           'direct256.py', 'real_data.py', 'table_data.py', 'table_group.py',
           'night.py', 'runtime.py', 'transfer.py', 'continuous_metrics.py')
ORDERS = ('primary', 'reversed')
KINDS = ('reason', 'binary')
KEEP = 'KEEP'
REPLACE = 'REPLACE'
REASONS = {
    'COLLISION': ('no_at_fault_collisions', 'Better avoidance of at-fault collisions'),
    'DRIVABLE_AREA': ('drivable_area_compliance', 'Better compliance with the drivable area'),
    'DIRECTION': ('driving_direction_compliance', 'Better compliance with travel direction'),
    'TRAFFIC_LIGHT': ('traffic_light_compliance', 'Better compliance with traffic lights'),
    'TTC': ('time_to_collision_within_bound', 'Better time-to-collision margin'),
    'PROGRESS': ('ego_progress', 'More useful progress along the route'),
    'LANE_KEEPING': ('lane_keeping', 'Better lane keeping'),
    'HISTORY_COMFORT': ('history_comfort', 'Better comfort considering recent ego motion'),
}
METRIC_INDEX = {name: i for i, name in enumerate(METRICS)}
CONTINUOUS = ('route_progress_m', 'all_jerk_norm_mps3_rms', 'all_accel_lat_mps2_rms')
EPS = 1e-8
NOTE = ('Exploratory offline replay on the previously examined TRAIN cohort. Both prompts get the '
        'same fixed pair and available estimates. All measured outcomes are read only after choices '
        'are committed. Neither a reason nor model confidence is a safety guarantee. Local PDM.')


def head_proxy(candidate):
    """Same head aggregation as the Astra absolute-head control."""
    nc, dac, ddc, light, progress, ttc, lane, comfort = candidate['values'][:8]
    return nc * dac * ddc * light * (5 * progress + 5 * ttc + 2 * lane + 2 * comfort) / 14


def predicted_guard(reference, challenger):
    left, right = reference['values'], challenger['values']
    safe = all(right[METRIC_INDEX[name]] >= left[METRIC_INDEX[name]] - EPS for name in SAFETY)
    progress = METRIC_INDEX['ego_progress']
    return safe and right[progress] >= left[progress] - CONFIG['predicted_progress_slack'] - EPS


def make_pair(arrays, vocab):
    inputs = absolute_inputs(arrays, vocab)
    scores = np.asarray(arrays['selection_scores'], dtype=float)
    baseline = np.asarray(arrays['selected_local'])
    if (scores.shape != (256,) or not np.isfinite(scores).all() or baseline.shape != ()
            or baseline.dtype.kind not in 'iu'):
        raise ValueError('Invalid original DS scores or baseline index')
    ranking = np.argsort(-scores, kind='stable')[:CONFIG['top_k']]
    if int(baseline) != ranking[0]:
        raise ValueError('DS baseline and score ranking disagree')
    top = [dict(inputs['candidates'][j], rank=rank) for rank, j in enumerate(ranking, 1)]
    reference = top[0]
    # Prefer eligible alternatives, then the strongest head proxy. Rank breaks exact ties.
    challenger = max(top[1:], key=lambda c: (predicted_guard(reference, c), head_proxy(c), -c['rank']))
    guard = predicted_guard(reference, challenger)
    return dict(reference=reference, challenger=challenger, numeric_guard=guard,
                numeric_replace=guard and head_proxy(challenger) > head_proxy(reference) + EPS,
                ego_velocity=inputs['ego_velocity'], ego_acceleration=inputs['ego_acceleration'])


def option_keys(kind, order):
    if kind not in KINDS or order not in ORDERS:
        raise ValueError('Unknown question or order')
    keys = [KEEP, *REASONS] if kind == 'reason' else [KEEP, REPLACE]
    return keys if order == 'primary' else keys[::-1]


def payload(pair, kind, order):
    keys = option_keys(kind, order)
    descriptions = {
        KEEP: 'Keep the current DriveSuprim trajectory: no sufficiently supported reason to replace it',
        REPLACE: 'Replace the current DriveSuprim trajectory with the proposed alternative',
        **{key: 'Replace with the alternative because of: ' + desc for key, (_, desc) in REASONS.items()},
    }
    state = dict(
        columns=COLUMNS,
        trajectories=[['CURRENT_DS', *pair['reference']['values']],
                      ['ALTERNATIVE', *pair['challenger']['values']]],
        velocity_xy_mps=pair['ego_velocity'], acceleration_xy_mps2=pair['ego_acceleration'],
        coordinates='Current rear axle origin; x forward, y left; heading CCW; SI units.',
        evidence='pred_* fields are DriveSuprim estimates on a 0-1 scale, with larger values preferred. '
                 'Positions at 1, 2, and 4 seconds describe nominal motion. Outcomes and future actor motion '
                 'are unknown. Total path length is not route progress.',
    )
    question = ('Should the proposed alternative replace the current DriveSuprim trajectory? '
                'Require evidence of a concrete improvement. Keep the current trajectory when evidence is '
                'insufficient, when any safety indicator is worse, or when estimated useful progress '
                'is more than 0.01 lower on the 0-1 scale.')
    if kind == 'reason':
        question += ' If replacing, choose the single strongest supported reason.'
    row = dict(state=state, question=dict(type='choice', instructions=question,
                                         criteria={key: descriptions[key] for key in keys}))
    # Canonical JSON hashing sorts mapping keys. Explicitly bind the order that Drex sees.
    return row, keys, digest(dict(row=row, option_order=keys))


def top_option(probabilities, keys):
    validate_probabilities(probabilities, len(keys))
    best = max(probabilities)
    winners = [key for key, value in zip(keys, probabilities) if value == best]
    return winners[0] if len(winners) == 1 else KEEP


def policies(pair, votes):
    return {
        'ds': False,
        'numeric': pair['numeric_replace'],
        'drex_binary': pair['numeric_guard'] and all(votes['binary'][o] == REPLACE for o in ORDERS),
        'drex_reason': pair['numeric_guard'] and votes['reason']['primary'] != KEEP
                       and votes['reason']['primary'] == votes['reason']['reversed'],
        'drex_reason_primary_only': votes['reason']['primary'] != KEEP,
    }


def plan(root, out, replicas):
    package = root / 'data/navsim_train_v1/package'
    manifest, index, _, vocab = package_metadata(package)
    records = cohort(index, 'all')
    if len(records) != 651 or len({r['session_id'] for r in records}) != 32:
        raise ValueError('Expected frozen TRAIN cohort of 651 scenes and 32 sessions')
    pairs = {}
    pilot = []
    seen_sessions = set()
    for record in records:
        token = record['token']
        if len(pilot) < 4 and record['session_id'] not in seen_sessions:
            pilot.append(token)
            seen_sessions.add(record['session_id'])
        relative = f'bundle/examples/{token}.npz'
        path = checked(package, relative)
        if sha(path) != manifest['files'][relative]['sha256']:
            raise RuntimeError(f'Changed input for {token}')
        with np.load(path, allow_pickle=False) as arrays:
            pairs[token] = make_pair(arrays, vocab)
    freeze(out / 'plan.json', dict(config=CONFIG, records=records, pairs=pairs, pilot=pilot, replicas=replicas,
        manifest_sha256=sha(package / 'package_manifest.json'), model_plan=read(root / 'model_plan.json'),
        requirements_sha256=sha(root / 'requirements.lock.txt'),
        sources={name: sha(Path(__file__).with_name(name)) for name in SOURCES}))
    print(json.dumps(dict(status='planned', scenes=len(pairs), pilot_scenes=len(pilot),
                          planned_requests=4 * len(pairs), numeric_guard_passes=sum(p['numeric_guard'] for p in pairs.values()))))


def load_plan(root, out, replicas):
    snapshot = read(out / 'plan.json')
    if (snapshot['config'] != CONFIG or snapshot['replicas'] != replicas
            or snapshot['model_plan'] != read(root / 'model_plan.json')
            or snapshot['requirements_sha256'] != sha(root / 'requirements.lock.txt')):
        raise RuntimeError('Frozen protocol, runtime, or replica count changed')
    for name in SOURCES:
        if sha(Path(__file__).with_name(name)) != snapshot['sources'][name]:
            raise RuntimeError(f'Frozen source changed: {name}')
    return snapshot


def tasks(snapshot, phase):
    tokens = snapshot['pilot'] if phase == 'pilot' else [r['token'] for r in snapshot['records']]
    return [(token, kind, order) for token in tokens for kind in KINDS for order in ORDERS]


def result_path(out, task):
    return out / 'results' / ('-'.join(task) + '.json')


def checked_result(path, request_hash, plan_hash, task, keys):
    result = load_result(path, request_hash)
    if (result['plan_sha256'] != plan_hash or result['task'] != list(task)
            or result['keys'] != keys or result['k'] != len(keys)):
        raise RuntimeError(f'Request identity mismatch: {path}')
    return result


def worker(root, out, replicas, phase):
    import torch
    from drex_max.model import DecisionModel

    snapshot = load_plan(root, out, replicas)
    plan_hash = sha(out / 'plan.json')
    if phase == 'full':
        pilot = read(out / 'pilot_report.json')
        if (pilot['status'] != 'pilot_complete' or pilot['plan_sha256'] != plan_hash
                or pilot['decisions_sha256'] != sha(out / 'pilot_decisions.json')):
            raise RuntimeError('A matching committed pilot is required')
    rank = gpu_binding(out, replicas, 'reason-' + phase)
    directory = out / 'workers' / str(rank)
    assigned = tasks(snapshot, phase)[rank::replicas]
    pending = []
    for task in assigned:
        token, kind, order = task
        row, keys, request_hash = payload(snapshot['pairs'][token], kind, order)
        path = result_path(out, task)
        if path.exists():
            checked_result(path, request_hash, plan_hash, task, keys)
        else:
            pending.append((task, row, keys, request_hash, path))
    cached = len(assigned) - len(pending)
    status = directory / 'status.json'
    write(status, dict(state='loading' if pending else 'complete', phase=phase, done=cached, total=len(assigned)))
    if not pending:
        return

    envroot = directory / 'runtime'
    envroot.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / 'requirements.lock.txt', envroot / 'requirements.lock.txt')
    env = environment(envroot, True)
    freeze(directory / 'environment.json', {k: env[k] for k in ('python', 'torch', 'cuda_build', 'transformers', 'gpu', 'vram_bytes')})
    model_path = root / 'models/drex-max-v1'
    if remaining_bytes(model_path, snapshot['model_plan']['files']):
        raise RuntimeError('Pinned model files missing or changed size')
    decision = read(model_path / 'decision_config.json')
    if decision.get('format_version') != 1 or decision.get('attention_mode', 'noncausal_full_attention') != 'noncausal_full_attention':
        raise RuntimeError('Unexpected Drex decision format')
    model = DecisionModel(model_path, device='cuda:0', max_image_pixels=256 * 256)
    if model.readout.weight.dtype != torch.bfloat16:
        raise RuntimeError('Expected BF16')
    maximum = CONFIG['max_tokens']
    for number, (task, row, keys, request_hash, path) in enumerate(pending, cached + 1):
        start = time.monotonic()
        batch = model.prepare([row], max_length=maximum)
        tokens = int(batch.inputs['input_ids'].shape[1])
        if tokens >= maximum or 'pixel_values' in batch.inputs:
            raise RuntimeError(f'Invalid text request {task}: {tokens} tokens')
        del batch
        probabilities = [float(p) for p in model.predict([row], batch_size=1, max_length=maximum)[0]]
        validate_probabilities(probabilities, len(keys))
        save_result(path, dict(status='ok', task=list(task), keys=keys, k=len(keys), probabilities=probabilities,
                              request_sha256=request_hash, plan_sha256=plan_hash, input_tokens=tokens,
                              seconds=time.monotonic() - start))
        if number % 25 == 0 or number == len(assigned):
            write(status, dict(state='complete' if number == len(assigned) else 'running', phase=phase,
                               done=number, total=len(assigned), token=task[0]))


def collect(root, out, replicas, phase):
    snapshot = load_plan(root, out, replicas)
    plan_hash = sha(out / 'plan.json')
    request_list = tasks(snapshot, phase)
    votes = {}
    for task in request_list:
        token, kind, order = task
        _, keys, request_hash = payload(snapshot['pairs'][token], kind, order)
        result = checked_result(result_path(out, task), request_hash, plan_hash, task, keys)
        entry = votes.setdefault(token, {**{kind: {} for kind in KINDS}, 'margin': {kind: {} for kind in KINDS}})
        probabilities = dict(zip(keys, result['probabilities']))
        winner = top_option(result['probabilities'], keys)
        entry[kind][order] = winner
        entry['margin'][kind][order] = probabilities[winner] - probabilities[KEEP]
    choice_file = out / ('pilot_decisions.json' if phase == 'pilot' else 'decisions.json')
    freeze(choice_file, votes)
    interventions = Counter()
    for token, vote in votes.items():
        interventions.update({method: int(take) for method, take in policies(snapshot['pairs'][token], vote).items()})
    report = dict(status='pilot_complete' if phase == 'pilot' else 'choices_complete',
                  plan_sha256=plan_hash, decisions_sha256=sha(choice_file),
                  scenes=len(votes), requests=len(request_list), outcomes_read=False,
                  reason_disagreements=sum(v['reason']['primary'] != v['reason']['reversed'] for v in votes.values()),
                  interventions=dict(interventions), note=NOTE)
    freeze(out / ('pilot_report.json' if phase == 'pilot' else 'report.json'), report)
    print(json.dumps(report, indent=2))


def measured_pairs(snapshot, astra):
    source = read(astra / 'request.json')
    report = read(astra / 'report.json')
    if (source['manifest_sha256'] != snapshot['manifest_sha256'] or report['status'] != 'complete'
            or report['holdout_processed'] != 0 or report['scenes'] != len(snapshot['pairs'])
            or report['completed_candidates'] != 256 * len(snapshot['pairs'])
            or report['anchors_parity_passed'] != len(snapshot['pairs'])):
        raise ValueError('Expected completed Astra measurements for the identical frozen package')
    expected = {}
    for record in snapshot['records']:
        token = record['token']
        pair = snapshot['pairs'][token]
        original = source['candidates'][token]
        for role in ('reference', 'challenger'):
            candidate = pair[role]
            local = candidate['local']
            rank = candidate['rank']
            if (not 1 <= rank <= CONFIG['top_k'] or (role == 'reference') != (rank == 1)
                    or not 0 <= local < 256 or original['ranked'][rank - 1] != local
                    or original['vocab'][local] != candidate['vocab']):
                raise ValueError(f'Measurement candidate mismatch for {token}/{role}')
            expected[token, local] = (role, record['session_id'], rank, candidate['vocab'])
    data = {}
    metrics = ('pdm_score', *METRICS, *CONTINUOUS)
    row_count = 0
    seen = set()
    with gzip.open(astra / 'per_candidate.csv.gz', 'rt', newline='', encoding='utf-8') as file:
        for row in csv.DictReader(file):
            row_count += 1
            key = row['token'], int(row['candidate_local'])
            if key in seen:
                raise ValueError(f'Duplicate measured row for {key}')
            if key not in expected:
                continue
            role, session, rank, vocab = expected.pop(key)
            if (row['session_id'], int(row['teacher_rank']), int(row['candidate_vocab'])) != (session, rank, vocab):
                raise ValueError(f'Astra row identity mismatch for {key}')
            values = {m: float(row[m]) for m in metrics}
            if (not all(math.isfinite(v) for v in values.values())
                    or not all(0 <= values[m] <= 1 for m in ('pdm_score', *METRICS))):
                raise ValueError(f'Invalid Astra metrics for {key}')
            data.setdefault(key[0], {})[role] = values
            seen.add(key)
    if expected or row_count != report['completed_candidates']:
        raise ValueError('Incomplete Astra measurements')
    return data


def outcome(reference, challenger):
    delta = {name: value - reference[name] for name, value in challenger.items()}
    harm = [m for m in METRICS if m != 'ego_progress' and delta[m] < -EPS]
    loss_budget = min(CONFIG['progress_loss_m'], CONFIG['progress_loss_fraction'] * max(0., reference['route_progress_m']))
    progress_violation = delta['route_progress_m'] < -loss_budget - EPS
    supported = {name for name, (metric, _) in REASONS.items() if delta[metric] > EPS}
    if delta['route_progress_m'] < .5 - EPS:
        supported.discard('PROGRESS')
    helpful = delta['pdm_score'] >= CONFIG['minimum_pdm_gain'] - EPS and not harm and not progress_violation
    return delta, harm, progress_violation, supported, helpful


def evaluate(out, astra):
    report = read(out / 'report.json')
    if (report['status'] != 'choices_complete' or report['plan_sha256'] != sha(out / 'plan.json')
            or report['decisions_sha256'] != sha(out / 'decisions.json')):
        raise ValueError('Completed committed model choices required before reading outcomes')
    snapshot, votes = read(out / 'plan.json'), read(out / 'decisions.json')
    tokens = [r['token'] for r in snapshot['records']]
    if (snapshot['config'] != CONFIG or not tokens or len(tokens) != len(set(tokens)) or len(tokens) != report['scenes']
            or report['requests'] != len(KINDS) * len(ORDERS) * len(tokens)
            or {r['role'] for r in snapshot['records']} != {'train'}
            or set(tokens) != set(votes) or set(tokens) != set(snapshot['pairs'])):
        raise ValueError('Incomplete or incompatible model decisions')
    for token in tokens:
        for kind in KINDS:
            for order in ORDERS:
                margin = votes[token]['margin'][kind][order]
                if (votes[token][kind][order] not in option_keys(kind, order)
                        or not math.isfinite(margin) or not 0 <= margin <= 1):
                    raise ValueError(f'Invalid vote for {token}')
    measured = measured_pairs(snapshot, astra)
    sessions = [r['session_id'] for r in snapshot['records']]
    comparable = ('numeric', 'drex_binary', 'drex_reason')
    strength = {name: [] for name in comparable}
    rows = {}
    reasons = {name: Counter(issued=0, supported=0, supported_without_guard_violations=0) for name in REASONS}
    verified_pairs = 0
    for token in tokens:
        pair, vote = snapshot['pairs'][token], votes[token]
        delta, harm, violation, supported, helpful = outcome(**measured[token])
        verified_pairs += helpful
        choices = policies(pair, vote)
        choices['pair_oracle'] = helpful
        for name, take in choices.items():
            rows.setdefault(name, []).append((take, delta, harm, violation, helpful))
        for name in comparable:
            if choices[name]:
                score = (head_proxy(pair['challenger']) - head_proxy(pair['reference']) if name == 'numeric'
                         else min(vote['margin'][name.removeprefix('drex_')].values()))
                strength[name].append((score, token))
        if choices['drex_reason']:
            label = vote['reason']['primary']
            reasons[label]['issued'] += 1
            reasons[label]['supported'] += label in supported
            reasons[label]['supported_without_guard_violations'] += label in supported and not harm and not violation

    methods = {}
    for name, values in rows.items():
        methods[name] = dict(interventions=sum(take for take, *_ in values),
            pdm_decreases=sum(take and delta['pdm_score'] < -EPS for take, delta, *_ in values),
            measured_component_harm=sum(take and bool(harm) for take, _, harm, _, _ in values),
            component_harms={m: sum(take and m in harm for take, _, harm, _, _ in values)
                             for m in METRICS if m != 'ego_progress'},
            measured_progress_violation=sum(take and violation for take, _, _, violation, _ in values),
            helpful_interventions=sum(take and helpful for take, _, _, _, helpful in values),
            pdm_gain_pp=paired_stats([100 * delta['pdm_score'] if take else 0. for take, delta, *_ in values], sessions),
            **{m: paired_stats([delta[m] if take else 0. for take, delta, *_ in values], sessions) for m in CONTINUOUS})
    comparisons = {}
    for control in ('ds', 'numeric', 'drex_binary'):
        difference = [100 * delta['pdm_score'] * (int(take) - int(other[0]))
                      for (take, delta, *_), other in zip(rows['drex_reason'], rows[control])]
        comparisons['reason_minus_' + control] = paired_stats(difference, sessions)
    coverage = min(map(len, strength.values()))
    matched_methods = {}
    for name, ranked in strength.items():
        selected = {token for _, token in sorted(ranked, key=lambda x: (-x[0], x[1]))[:coverage]}
        values = rows[name]
        matched_methods[name] = dict(pdm_gain_pp=paired_stats(
            [100 * delta['pdm_score'] if token in selected else 0.
             for token, (_, delta, *_) in zip(tokens, values)], sessions),
            measured_component_harm=sum(bool(harm) and token in selected
                                        for token, (_, _, harm, _, _) in zip(tokens, values)))
    output = dict(status='evaluated', scenes=len(tokens), sessions=len(set(sessions)),
                  helpful_pairs=verified_pairs, methods=methods, paired_pdm_differences_pp=comparisons,
                  matched_coverage=dict(interventions=coverage, methods=matched_methods),
                  measured_reasons={k: dict(v) for k, v in reasons.items()},
                  decisions_sha256=report['decisions_sha256'], note=NOTE)
    write(out / 'evaluation.json', output)
    print(json.dumps(output, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'worker', 'collect', 'evaluate'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--astra', type=Path)
    parser.add_argument('--replicas', type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument('--phase', choices=('pilot', 'full'), default='pilot')
    args = parser.parse_args()
    out = args.out.resolve()
    if args.stage == 'evaluate':
        if args.astra is None:
            parser.error('evaluate requires --astra')
        evaluate(out, args.astra.resolve())
        return
    if args.root is None or not re.fullmatch(r'reason_[A-Za-z0-9_-]+', out.name):
        parser.error('GPU stages require --root and a reason_* output directory')
    root = args.root.resolve()
    if args.stage == 'plan':
        plan(root, out, args.replicas)
    elif args.stage == 'collect':
        collect(root, out, args.replicas, args.phase)
    else:
        try:
            worker(root, out, args.replicas, args.phase)
        except Exception as exc:
            rank = int(os.environ.get('SLURM_PROCID', '0'))
            write(out / 'workers' / str(rank) / 'status.json', dict(state='failed', phase=args.phase,
                  error=repr(exc), traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
