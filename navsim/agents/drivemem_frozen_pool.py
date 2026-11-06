"""Replay only saved candidate IDs; never inspect their measured outcomes."""
import numpy as np


def apply_pool(output, coarse_states, saved_ids):
    saved = np.asarray(saved_ids)
    vocab = output["trajectory_vocab"]
    if (saved.ndim != 1 or saved.dtype.kind not in "iu" or len(set(saved.tolist())) != len(saved)
            or not len(saved) or saved.min() < 0 or saved.max() >= len(vocab)):
        raise ValueError("Invalid frozen candidate IDs")
    if len(output["refinement"]) != 1 or coarse_states.ndim != 3 or coarse_states.shape[0] != 1 or coarse_states.shape[1] != len(vocab):
        raise ValueError("Frozen replay requires one stage and batch=1 full-vocab states")
    stage = output["refinement"][0]
    native = stage["indices_absolute"]
    if native.shape != (1, len(saved)):
        raise ValueError("Frozen/current pool size differs")
    ids = native.new_tensor(saved).reshape(1, -1) if hasattr(native, "new_tensor") else saved.astype(native.dtype)[None]
    stage["indices_absolute"] = ids
    stage["trajs"] = vocab[ids]
    stage["trajs_status"] = coarse_states[:, ids[0], :]
    stage["coarse_score"] = {k: output[k][:, ids[0]] for k in stage["coarse_score"]}
    return output


def describe_pool(saved, natural):
    a, b = set(map(int, saved)), set(map(int, natural))
    return dict(same_set=a == b, same_order=bool(np.array_equal(saved, natural)),
                missing_from_natural=sorted(a - b), added_by_natural=sorted(b - a))
