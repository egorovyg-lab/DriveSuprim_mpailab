"""Fixed CPU information probe. Model APIs consume TRAIN labels only."""
import hashlib

import numpy as np

ARMS = ('ds', 'ds_drex_text', 'ds_drex_blank', 'ds_drex_road', 'ds_drex_shuffled')
FAMILIES = ('linear', 'random_features')
TARGETS = ('jerk_gain_mps3', 'lateral_gain_mps2', 'progress_delta_m', 'pdm_gain_pp',
           'jerk_peak_delta_mps3', 'harm', 'benefit')
CONFIG = dict(version=1, seed=20261116, folds=4, ridge_lambda=.1, random_features=64,
              benefit_thresholds=[.01, .025, .05, .1, .25, .5, .75, 1.01],
              primary_family='random_features', primary_threshold=.1,
              risk_ceiling=.05, jerk_gain_min=.2, jerk_relative_min=.2,
              lateral_gain_min=.05, lateral_relative_min=.2,
              road_margin_loss_m=.1, actor_margin_loss_m=.25,
              progress_loss_m=1., progress_loss_fraction=.05,
              bootstrap_reps=2000)
NOTE = ('Development-only session OOF on repeatedly inspected TRAIN data, not independent validation. '
        'No Drex inference/fine-tuning, new scorer calls or holdout. Tests information in existing '
        'cached Drex pair preferences, not every possible Drex representation or prompt. '
        'Ridge benefit/risk scores are uncalibrated. Thresholds are predeclared, not selected on OOF. '
        'Maps, future actors and simulated outcomes are targets/evaluation only. '
        'Margins can be unavailable; missing margins are never zero-imputed. '
        'Local PDM, not full EPDMS; sampled non-reactive rollouts are not driving safety certification.')


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def folds(sessions):
    names = sorted(set(sessions), key=lambda s: digest((CONFIG['seed'], s)))
    if len(names) < CONFIG['folds']:
        raise ValueError('At least four complete sessions required')
    mapping = {s: i % CONFIG['folds'] for i, s in enumerate(names)}
    return np.array([mapping[s] for s in sessions], dtype=np.int64)


def weights(sessions, candidates):
    """Equal mass per session, then scene, then candidate (no iid row weighting)."""
    _, inverse, counts = np.unique(sessions, return_inverse=True, return_counts=True)
    return np.repeat(1. / counts[inverse] / len(counts) / candidates, candidates)


def inputs(x, preferences, tokens, arm):
    if arm not in ARMS or preferences.shape != (*x.shape[:2], 3, 4):
        raise ValueError('Unknown arm/alignment')
    extra = np.zeros((*x.shape[:2], 4))
    if arm != 'ds':
        mode = {'ds_drex_text': 0, 'ds_drex_blank': 1, 'ds_drex_road': 2,
                'ds_drex_shuffled': 2}[arm]
        extra = preferences[:, :, mode].copy()
        if arm == 'ds_drex_shuffled':
            for i, token in enumerate(tokens):
                rng = np.random.default_rng(int(digest((CONFIG['seed'], token, 'shuffle'))[:16], 16))
                extra[i] = extra[i, rng.permutation(x.shape[1])]
    result = np.concatenate((x, extra), axis=-1)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite predictor input')
    return result


def design(x, model):
    z = np.clip((x - model['x_mean']) / model['x_scale'], -10., 10.)
    parts = [np.ones((len(x), 1)), z]
    if model['projection'].shape[1]:
        a = z @ model['projection']
        parts += [np.sin(a) / np.sqrt(a.shape[1]), np.cos(a) / np.sqrt(a.shape[1])]
    return np.concatenate(parts, axis=1)


def fit(x_train, y_train, weight, family):
    """No held-out feature/label parameters and no outcome-based hyperparameter search."""
    if family not in FAMILIES or not (np.isfinite(x_train).all() and np.isfinite(y_train).all()):
        raise ValueError('Invalid training data/family')
    w = np.asarray(weight, float)
    if len(w) != len(x_train) or np.any(w <= 0):
        raise ValueError('Positive row weights required')
    w = w / w.sum()
    xm, ym = w @ x_train, w @ y_train
    xs = np.sqrt(w @ (x_train - xm)**2); xs[xs < 1e-8] = 1.
    ys = np.sqrt(w @ (y_train - ym)**2); ys[ys < 1e-8] = 1.
    rng = np.random.default_rng(CONFIG['seed'])
    p = (rng.normal(size=(x_train.shape[1], CONFIG['random_features'])) / np.sqrt(x_train.shape[1])
         if family == 'random_features' else np.empty((x_train.shape[1], 0)))
    model = dict(x_mean=xm, x_scale=xs, y_mean=ym, y_scale=ys, projection=p)
    a = design(x_train, model)
    gram = a.T @ (w[:, None] * a)
    penalty = np.eye(a.shape[1]) * CONFIG['ridge_lambda']; penalty[0, 0] = 0.
    model['coef'] = np.linalg.solve(gram + penalty, a.T @ (w[:, None] * ((y_train - ym) / ys)))
    return model


def predict(model, x):
    return (design(x, model) @ model['coef']) * model['y_scale'] + model['y_mean']


def choose(prediction, threshold):
    """Return -1 for DS. Predicted values only; physical truth cannot enter the gate."""
    p = np.asarray(prediction, float)
    if p.ndim != 3 or p.shape[-1] != len(TARGETS) or not np.isfinite(p).all():
        raise ValueError('Invalid aligned prediction block')
    benefit = np.clip(p[:, :, 6], 0., 1.)
    comfort_gain = (((p[:, :, 0] >= CONFIG['jerk_gain_min']) & (p[:, :, 1] >= 0.))
                    | ((p[:, :, 1] >= CONFIG['lateral_gain_min']) & (p[:, :, 0] >= 0.)))
    allowed = ((benefit > threshold) & (np.clip(p[:, :, 5], 0., 1.) <= CONFIG['risk_ceiling'])
               & comfort_gain & (p[:, :, 4] <= 0.)
               & (p[:, :, 2] >= -CONFIG['progress_loss_m']))
    score = np.where(allowed, benefit, -np.inf)
    best = np.argmax(score, axis=1)  # Stable original teacher order on alternative ties.
    return np.where(allowed.any(axis=1), best, -1)


def average_precision(y, score):
    """Group equal scores; unavailable if a held-out session contains no positives."""
    y, score = np.asarray(y), np.asarray(score)
    if not y.sum():
        return None
    order = np.argsort(-score, kind='stable'); y, score = y[order], score[order]
    ends = np.r_[np.flatnonzero(np.diff(score)), len(score) - 1]
    positive = np.cumsum(y)[ends]
    return float(np.sum(np.diff(np.r_[0., positive]) * positive / (ends + 1)) / y.sum())


def paired_interval(values, sessions):
    values = np.asarray(values, float)
    means = np.array([values[np.asarray(sessions) == s].mean() for s in sorted(set(sessions))])
    rng = np.random.default_rng(CONFIG['seed'])
    draws = means[rng.integers(len(means), size=(CONFIG['bootstrap_reps'], len(means)))].mean(axis=1)
    return dict(mean=float(means.mean()), ci95=[float(v) for v in np.quantile(draws, [.025, .975])],
                sessions=len(means))
