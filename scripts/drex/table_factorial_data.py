"""Matched image x final-DS-score ablation; input-only, fixed top64 pool."""
import hashlib
import io

from PIL import Image

import real_data as base
import table_data as original

IMAGES = ('none', 'blank', 'road')
SCORES = ('full', 'hidden')
MODES = tuple(image+'_'+score for image in IMAGES for score in SCORES)
ORDERS = original.ORDERS
CONFIG = dict(protocol='table_factorial_v1', k=64, images=list(IMAGES), scores=list(SCORES),
              orders=list(ORDERS), seed=original.CONFIG['seed'], max_length=16384,
              max_image_pixels=1536*640, bootstrap_reps=2000, primary='road_minus_blank/full/mean3')
NOTE = ('Exploratory train replay after inspecting previous outcomes; not independent validation. '
        'Fixed original top64, aliases, three orders, rounded features and weights. Only ds_score_gap '
        'is removed in hidden: component heads and imitation prior remain, and top64 is still selected '
        'by DS. Blank preserves board dimensions and view/time headings, masking photograph tiles. '
        'All six conditions rerun; old predictions are not substituted. No training, new labels, map '
        'inputs or holdout. Probabilities are not calibrated risk. Raw local PDM, not full EPDMS.')


def blank_board(photo):
    if photo.size != (1536, 640):
        raise ValueError('Expected original 1536x640 six-view board')
    result = photo.convert('RGB').copy()
    for y in (0, 320):
        # Keep exactly the existing title strip; erase the full photograph area.
        result.paste(Image.new('RGB', (1536, 292), (255, 255, 255)), (0, y+28))
    return result


def payload(token, sample, mode, ordering, photo):
    if mode not in MODES:
        raise ValueError('Unknown factorial condition')
    image, score = mode.split('_')
    row, permutation, _ = original.payload(token, sample, 64, 'table', ordering)
    if score == 'hidden':
        table = row['state']['candidates']
        j = table['columns'].index('ds_score_gap')
        table['columns'] = table['columns'][:j]+table['columns'][j+1:]
        table['rows'] = [r[:j]+r[j+1:] for r in table['rows']]
        row['state']['predictions'] = (
            'pred_* are sigmoid DS head estimates, not measured outcomes or calibrated safety.')
    pixels = []
    if image != 'none':
        value = blank_board(photo) if image == 'blank' else photo
        row['images'] = [value]
        buffer = io.BytesIO(); value.save(buffer, format='PNG')
        pixels = [hashlib.sha256(buffer.getvalue()).hexdigest()]
    digest = base.digest(dict(row={k:v for k,v in row.items() if k != 'images'},
                              pixels=pixels, order=permutation))
    return row, permutation, digest


def tasks(records):
    return [(r['token'], mode, order) for r in records for mode in MODES for order in ORDERS]


def task_name(token, mode, order):
    return token+'-'+mode+'-'+order+'.json'
