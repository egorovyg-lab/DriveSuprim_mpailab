"""Rate each DS trajectory independently on matched road and white backgrounds."""

import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import overlay_v2_data as projection
import real_data as base
from consequence_run import csv_gz
from continuous_metrics import paired_stats
from direct255 import membership
from night import validate_probabilities
from reason_replacement import CONTINUOUS, EPS, outcome
from runtime import environment, remaining_bytes
from table_data import cohort
from table_group import gpu_binding
from transfer import checked, freeze, read, sha, write


MODES = ('blank', 'road')
POOLS = (16, 64)
OPTIONS = ('SUITABLE', 'UNSUITABLE')
PILOT_RANKS = (0, 1, 3, 7, 15, 31, 47, 63)
CONFIG = dict(protocol='pointwise_v1', pools=list(POOLS), modes=list(MODES),
              image_size=[768, 640], max_tokens=4096, options=list(OPTIONS), pilot_ranks=list(PILOT_RANKS))
SOURCES = ('pointwise.py', 'run_pointwise.sh', 'real_data.py', 'overlay_v2_data.py', 'overlay_data.py',
           'direct255.py', 'runtime.py', 'transfer.py', 'slurm_options.sh', 'table_data.py',
           'table_group.py', 'night.py', 'reason_replacement.py', 'continuous_metrics.py')
METRICS = ('pdm_score', *base.METRICS, *CONTINUOUS)
NOTE = ('Previously inspected TRAIN cohort. Each request sees one candidate, without DS scores, '
        'heads, rank, route command or outcome information. Both image conditions use identical '
        'v2 projected geometry and a full-path schematic. Scores are preference signals. Local PDM.')


def scene_inputs(arrays, vocab_size):
    ranked, _, _ = membership(arrays)
    ids = np.asarray(arrays['candidate_ids'])
    ego = np.asarray(arrays['ego_status'], dtype=float)
    if (ids.shape != (256,) or ids.dtype.kind not in 'iu' or len(set(ids.tolist())) != 256
            or ids.min() < 0 or ids.max() >= vocab_size
            or ego.ndim != 1 or len(ego) < 8 or len(ego) % 8 or not np.isfinite(ego).all()):
        raise ValueError('Invalid DS candidate IDs or ego status')
    ranks = ranked[:max(POOLS)]
    return dict(local=ranks, vocab=ids[ranks].tolist(), velocity=ego[4:6].tolist(), acceleration=ego[6:8].tolist())


def scale_for(trajectories):
    xy = np.vstack((np.zeros((1, 2)), trajectories[:, :, :2].reshape(-1, 2)))
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    scale = min(224 / (hi[0] - lo[0] + 4), 680 / (hi[1] - lo[1] + 4))
    return (hi + lo) / 2, scale


def candidate_images(photo, calib, poses, scale):
    """Render the same line mask and BEV panel on both backgrounds."""
    size = (768, 352)
    road = ImageOps.contain(photo, size)
    position = ((size[0] - road.width) // 2, (size[1] - road.height) // 2)
    mask = Image.new('1', road.size)
    draw = ImageDraw.Draw(mask)
    segments = projection.path_segments(poses, calib, photo.size)
    sx, sy = road.width / photo.width, road.height / photo.height
    for (x0, y0), (x1, y1) in segments:
        draw.line((x0 * sx, y0 * sy, x1 * sx, y1 * sy), fill=1, width=2)

    image = Image.new('RGB', tuple(CONFIG['image_size']), 'white')
    draw = ImageDraw.Draw(image)
    draw.line((0, 352, 767, 352), fill='#aaa')
    draw.text((12, 360), 'Full 4 s path | up = forward | left = left', fill='black', font=base.font(16))
    mid, pixels_per_m = scale

    def pixel(x, y):
        return 384 - (y - mid[1]) * pixels_per_m, 500 - (x - mid[0]) * pixels_per_m

    origin = pixel(0., 0.)
    points = [origin, *(pixel(x, y) for x, y, _ in poses)]
    draw.line(points, fill='#0057a8', width=3)
    x, y = origin
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill='black')
    for second in range(1, 5):
        x, y = points[second * 10]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill='#cf5a00')
        draw.text((x + 5, y - 6), str(second), fill='#8b3900', font=base.font(13))
    x, y, _ = poses[-1]
    draw.text((12, 614), f'4 s: forward {x:.1f} m, left {y:.1f} m | {pixels_per_m:.2f} px/m',
              fill='black', font=base.font(15))

    both = {mode: image.copy() for mode in MODES}
    both['road'].paste(road, position)
    for rendered in both.values():
        rendered.paste('#007acc', (*position, position[0] + road.width, position[1] + road.height), mask)
    return both, bool(segments)


def payload(inputs, image):
    return dict(state=dict(
        image='Top: the visible path projected onto the current front view using a flat-ground approximation, '
              'with the background possibly masked. Bottom: full nominal path in ego coordinates, '
              'with markers at 1 to 4 seconds.',
        velocity_xy_mps=inputs['velocity'], acceleration_xy_mps2=inputs['acceleration']),
        question=dict(type='choice', criteria={key: text for key, text in zip(OPTIONS, (
            'The path is a suitable continuation of the observed motion and scene.',
            'The path is an unsuitable continuation of the observed motion and scene.'))},
            instructions='Rate this single four-second trajectory. Use its motion continuity, plausible '
                         'forward progress and comfort, and any visible road context. Use the full path '
                         'when the projection is incomplete. Base the decision on available evidence.'),
        images=[image])


def package(root):
    return (root / 'data/navsim_train_v1/package').resolve()


def read_photo(folder, manifest, scene):
    path = base.camera_paths(folder, scene, manifest)[4]
    relative = path.relative_to(folder).as_posix()
    if sha(path) != manifest['files'][relative]['sha256']:
        raise ValueError(f'Changed current photograph: {path}')
    with Image.open(path) as image:
        photo = image.convert('RGB')
    return photo, projection.calibration(scene, photo.size)


def plan(root, out, replicas):
    folder = package(root)
    manifest, index, scenes, vocab = base.package_metadata(folder)
    records = cohort(index, 'all')
    if len(records) != 651 or len({r['session_id'] for r in records}) != 32:
        raise ValueError('Expected frozen TRAIN cohort: 651 scenes across 32 sessions')
    entries = {}
    pilot = {}
    for record in records:
        token = record['token']
        if len(pilot) < 8:
            pilot.setdefault(record['session_id'], token)
        relative = f'bundle/examples/{token}.npz'
        path = checked(folder, relative)
        if sha(path) != manifest['files'][relative]['sha256']:
            raise ValueError(f'Changed DS inputs for {token}')
        with np.load(path, allow_pickle=False) as arrays:
            entries[token] = scene_inputs(arrays, len(vocab))
    freeze(out / 'plan.json', dict(config=CONFIG, records=records, entries=entries, replicas=replicas,
        sources={name: sha(Path(__file__).with_name(name)) for name in SOURCES}, pilot=list(pilot.values()),
        manifest_sha256=sha(folder / 'package_manifest.json'), model_plan=read(root / 'model_plan.json'),
        requirements_sha256=sha(root / 'requirements.lock.txt')))
    token = next(iter(pilot.values()))
    photo, calib = read_photo(folder, manifest, scenes[token])
    paths = vocab[entries[token]['vocab']]
    images, _ = candidate_images(photo, calib, paths[0], scale_for(paths))
    for mode, image in images.items():
        image.save(out / f'preview-{mode}.png')
    print(json.dumps(dict(status='planned', scenes=len(records), requests=len(records) * 128,
                          pilot_requests=len(pilot) * len(PILOT_RANKS) * len(MODES))))


def load_plan(root, out, replicas):
    snapshot = read(out / 'plan.json')
    if (snapshot['config'] != CONFIG or snapshot['replicas'] != replicas
            or snapshot['manifest_sha256'] != sha(package(root) / 'package_manifest.json')
            or snapshot['requirements_sha256'] != sha(root / 'requirements.lock.txt')
            or snapshot['model_plan'] != read(root / 'model_plan.json')
            or snapshot['sources'] != {name: sha(Path(__file__).with_name(name)) for name in SOURCES}):
        raise ValueError('Frozen inputs, runtime, code or replica count changed')
    return snapshot


def checkpoint(out, token, binding, entry, values=None):
    path = out / 'scores' / f'{token}.json'
    identity = dict(token=token, plan_sha256=binding, local=entry['local'], vocab=entry['vocab'])
    if values is not None:
        record = dict(identity=identity, values=values)
        write(path, dict(record, sha256=base.digest(record)))
        return
    if not path.exists():
        return {}
    record = read(path)
    if record.pop('sha256') != base.digest(record) or record['identity'] != identity:
        raise ValueError(f'Changed score checkpoint for {token}')
    positions = {str(i) for i in range(max(POOLS))}
    for position, row in record['values'].items():
        if (position not in positions or set(row) != {'visible', *MODES}
                or not isinstance(row['visible'], bool)):
            raise ValueError(f'Invalid checkpoint row for {token}/{position}')
        for mode in MODES:
            item = row[mode]
            if (not math.isfinite(item['score']) or not 0 <= item['score'] <= 1
                    or not 0 < item['tokens'] < CONFIG['max_tokens']
                    or not math.isfinite(item['seconds']) or item['seconds'] < 0):
                raise ValueError(f'Invalid suitability score for {token}/{position}/{mode}')
    return record['values']


def infer_scene(entry, photo, calib, vocab, positions, values, infer, save):
    paths = vocab[entry['vocab']]
    scale = scale_for(paths)
    for position in positions:
        key = str(position)
        if key in values:
            continue
        images, visible = candidate_images(photo, calib, paths[position], scale)
        # Alternate execution order to avoid tying cold starts to one background.
        modes = MODES if position % 2 == 0 else MODES[::-1]
        values[key] = {'visible': visible, **{m: infer(payload(entry, images[m])) for m in modes}}
        save(values)


def model_inference(root, directory, model_plan):
    import torch
    from drex_max.model import DecisionModel

    envroot = directory / 'runtime'
    envroot.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / 'requirements.lock.txt', envroot / 'requirements.lock.txt')
    env = environment(envroot, True)
    freeze(directory / 'environment.json', {k: env[k] for k in ('python', 'torch', 'cuda_build', 'transformers', 'gpu', 'vram_bytes')})
    target = root / 'models/drex-max-v1'
    if remaining_bytes(target, model_plan['files']):
        raise ValueError('Model files missing or changed size')
    cfg = read(target / 'decision_config.json')
    if cfg.get('format_version') != 1 or cfg.get('attention_mode', 'noncausal_full_attention') != 'noncausal_full_attention':
        raise ValueError('Unexpected Drex decision format')
    model = DecisionModel(target, device='cuda:0', max_image_pixels=int(np.prod(CONFIG['image_size'])))
    if model.readout.weight.dtype != torch.bfloat16:
        raise ValueError('Expected BF16')

    def infer(row):
        start = time.monotonic()
        limit = CONFIG['max_tokens']
        batch = model.prepare([row], max_length=limit)
        tokens = int(batch.inputs['input_ids'].shape[1])
        grid = batch.inputs.get('image_grid_thw')
        if tokens >= limit or 'pixel_values' not in batch.inputs or grid is None or int(grid.shape[0]) != 1:
            raise ValueError('Expected one untruncated candidate image')
        del batch
        probabilities = [float(v) for v in model.predict([row], batch_size=1, max_length=limit)[0]]
        validate_probabilities(probabilities, len(OPTIONS))
        return dict(score=probabilities[0], tokens=tokens, seconds=time.monotonic() - start)

    return infer


def worker(root, out, replicas, phase):
    snapshot = load_plan(root, out, replicas)
    binding = sha(out / 'plan.json')
    if phase == 'full':
        pilot = read(out / 'pilot_report.json')
        if pilot['status'] != 'pilot_complete' or pilot['plan_sha256'] != binding:
            raise ValueError('Run the matching pilot first')
    rank = gpu_binding(out, replicas, 'pointwise-' + phase)
    directory = out / 'workers' / str(rank)
    full = [r['token'] for r in snapshot['records']]
    # Pilot and full own the same scene checkpoints on the same ranks.
    tokens = full[rank::replicas]
    positions = range(max(POOLS))
    if phase == 'pilot':
        tokens = [t for t in tokens if t in snapshot['pilot']]
        positions = PILOT_RANKS
    entries = snapshot['entries']
    cached = {t: checkpoint(out, t, binding, entries[t]) for t in tokens}
    done = sum(str(i) in cached[t] for t in tokens for i in positions) * len(MODES)
    total = len(tokens) * len(positions) * len(MODES)
    start = time.monotonic()
    initial = done

    def status(state):
        write(directory / 'status.json', dict(state=state, phase=phase, done=done, total=total,
              new_requests=done - initial, elapsed_seconds=time.monotonic() - start))

    status('loading' if done < total else 'complete')
    if done == total:
        return
    folder = package(root)
    manifest, _, scenes, vocab = base.package_metadata(folder)
    infer = model_inference(root, directory, snapshot['model_plan'])
    for token in tokens:
        if all(str(i) in cached[token] for i in positions):
            continue
        photo, calib = read_photo(folder, manifest, scenes[token])

        def save(values):
            nonlocal done
            checkpoint(out, token, binding, entries[token], values)
            done += len(MODES)
            if done % 32 == 0 or done == total:
                status('running')

        infer_scene(entries[token], photo, calib, vocab, positions, cached[token], infer, save)
    status('complete')


def select(values, mode, k):
    scores = [values[str(i)][mode]['score'] for i in range(k)]
    best = max(scores)
    winners = [i for i, score in enumerate(scores) if score == best]
    # DS is the default on ties involving its trajectory, then the earlier DS rank wins.
    return dict(position=winners[0], tie=len(winners) > 1, score=best)


def collect(root, out, replicas, phase):
    snapshot = load_plan(root, out, replicas)
    binding = sha(out / 'plan.json')
    tokens = snapshot['pilot'] if phase == 'pilot' else [r['token'] for r in snapshot['records']]
    positions = PILOT_RANKS if phase == 'pilot' else range(max(POOLS))
    scores = {}
    for token in tokens:
        values = checkpoint(out, token, binding, snapshot['entries'][token])
        if any(str(i) not in values for i in positions):
            raise ValueError(f'Unfinished suitability scores for {token}')
        scores[token] = {str(i): values[str(i)] for i in positions}
    report = dict(status='pilot_complete' if phase == 'pilot' else 'choices_complete',
        plan_sha256=binding, scenes=len(tokens), requests=len(tokens) * len(positions) * len(MODES),
        candidates_with_visible_projection=sum(v['visible'] for rows in scores.values() for v in rows.values()),
        max_tokens=max(v[m]['tokens'] for rows in scores.values() for v in rows.values() for m in MODES),
        outcomes_read=False, note=NOTE)
    if phase == 'pilot':
        report['median_model_seconds_by_mode'] = {m: float(np.median(
            [v[m]['seconds'] for rows in scores.values() for v in rows.values()])) for m in MODES}
    if phase == 'full':
        decisions = {t: {f'{m}/top{k}': select(rows, m, k) for m in MODES for k in POOLS}
                     for t, rows in scores.items()}
        freeze(out / 'decisions.json', decisions)
        report['decisions_sha256'] = sha(out / 'decisions.json')
        report['methods'] = {name: dict(replacements=sum(rows[name]['position'] != 0 for rows in decisions.values()),
                                       ties=sum(rows[name]['tie'] for rows in decisions.values()))
                             for name in next(iter(decisions.values()))}
    freeze(out / ('pilot_report.json' if phase == 'pilot' else 'report.json'), report)
    print(json.dumps(report, indent=2))


def measurements(snapshot, astra, choices):
    source, report = read(astra / 'request.json'), read(astra / 'report.json')
    count = len(snapshot['records'])
    if (source['manifest_sha256'] != snapshot['manifest_sha256'] or report['status'] != 'complete'
            or report['holdout_processed'] != 0 or report['oracle_scenes'] != count
            or report['completed_candidates'] != 256 * count or report['anchors_parity_passed'] != count):
        raise ValueError('Expected the matching completed Astra top256 measurements')

    expected, metrics = {}, {}
    for record in snapshot['records']:
        token = record['token']
        entry, reference = snapshot['entries'][token], source['candidates'][token]
        if (entry['local'] != reference['ranked'][:max(POOLS)]
                or entry['vocab'] != [reference['vocab'][j] for j in entry['local']]):
            raise ValueError(f'Physical candidate alignment changed for {token}')
        positions = {0, *(choice['position'] for choice in choices[token].values())}
        for i in positions:
            expected[(token, entry['local'][i])] = (record['session_id'], i + 1, entry['vocab'][i])

    with gzip.open(astra / 'per_candidate.csv.gz', 'rt', newline='', encoding='utf-8') as file:
        for row in csv.DictReader(file):
            key = row['token'], int(row['candidate_local'])
            if key not in expected:
                continue
            if key in metrics or expected[key] != (row['session_id'], int(row['teacher_rank']), int(row['candidate_vocab'])):
                raise ValueError(f'Mismatched or duplicate Astra row for {key}')
            values = {name: float(row[name]) for name in METRICS}
            if (not all(math.isfinite(v) for v in values.values())
                    or not all(0 <= values[m] <= 1 for m in ('pdm_score', *base.METRICS))):
                raise ValueError(f'Invalid Astra metrics for {key}')
            metrics[key] = values
    if set(metrics) != set(expected):
        raise ValueError('Incomplete Astra outcomes for selected candidates')
    return metrics


def evaluate(out, astra):
    report = read(out / 'report.json')
    if (report['status'] != 'choices_complete' or report['outcomes_read'] is not False
            or report['plan_sha256'] != sha(out / 'plan.json')
            or report['decisions_sha256'] != sha(out / 'decisions.json')):
        raise ValueError('Expected complete committed decisions')
    snapshot, choices = read(out / 'plan.json'), read(out / 'decisions.json')
    records = snapshot['records']
    tokens = {r['token'] for r in records}
    methods = {f'{m}/top{k}': k for m in MODES for k in POOLS}
    if (snapshot['config'] != CONFIG or {r['role'] for r in records} != {'train'}
            or len(tokens) != len(records) or set(choices) != tokens or set(snapshot['entries']) != tokens
            or report['scenes'] != len(tokens) or report['requests'] != len(tokens) * max(POOLS) * len(MODES)):
        raise ValueError('Incomplete or incompatible choices')
    for token, rows in choices.items():
        if set(rows) != set(methods) or any(
                type(choice['position']) is not int or not 0 <= choice['position'] < methods[name]
                for name, choice in rows.items()):
            raise ValueError(f'Invalid choice for {token}')
    metrics = measurements(snapshot, astra, choices)
    rows = {m: [] for m in methods}
    for record in records:
        token = record['token']
        entry = snapshot['entries'][token]
        reference = metrics[token, entry['local'][0]]
        for name, choice in choices[token].items():
            i = choice['position']
            local = entry['local'][i]
            delta, harm, progress_loss, _, helpful = outcome(reference, metrics[token, local])
            rows[name].append(dict(token=token, session_id=record['session_id'], method=name,
                candidate_local=local, candidate_vocab=entry['vocab'][i], teacher_rank=i + 1,
                score=choice['score'], tie=choice['tie'], replacement=i != 0,
                pdm_gain_pp=100 * delta['pdm_score'], helpful=helpful,
                component_harm='|'.join(harm), progress_budget_exceeded=progress_loss,
                **{m: delta[m] for m in CONTINUOUS}))
    sessions = [r['session_id'] for r in records]
    fields = ('pdm_gain_pp', *CONTINUOUS)
    statistics = {name: dict(replacements=sum(r['replacement'] for r in data),
        ties=sum(r['tie'] for r in data), helpful=sum(r['helpful'] for r in data),
        pdm_decreased=sum(r['pdm_gain_pp'] < -100 * EPS for r in data),
        component_harms={m: sum(m in r['component_harm'].split('|') for r in data)
                         for m in base.METRICS if m != 'ego_progress'},
        progress_budget_exceeded=sum(r['progress_budget_exceeded'] for r in data),
        **{m: paired_stats([r[m] for r in data], sessions) for m in fields})
        for name, data in rows.items()}
    comparisons = [(f'road/top{k}', f'blank/top{k}') for k in POOLS] + [
        (f'{m}/top64', f'{m}/top16') for m in MODES]
    contrasts = {f'{a} minus {b}': dict(
        choices_changed=sum(x['candidate_local'] != y['candidate_local'] for x, y in zip(rows[a], rows[b])),
        **{m: paired_stats([x[m] - y[m] for x, y in zip(rows[a], rows[b])], sessions) for m in fields})
        for a, b in comparisons}
    result = dict(status='evaluated', scenes=len(records), sessions=len(set(sessions)), holdout_processed=0,
                  decisions_sha256=report['decisions_sha256'], methods=statistics, contrasts=contrasts, note=NOTE)
    csv_gz(out / 'per_scene.csv.gz', [r for data in rows.values() for r in data])
    write(out / 'evaluation.json', result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'worker', 'collect', 'evaluate'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--root', type=Path)
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
    if args.root is None:
        parser.error('Slurm stages require --root')
    root = args.root.resolve()
    if args.stage == 'plan':
        plan(root, out, args.replicas)
    elif args.stage == 'collect':
        collect(root, out, args.replicas, args.phase)
    else:
        worker(root, out, args.replicas, args.phase)


if __name__ == '__main__':
    main()
