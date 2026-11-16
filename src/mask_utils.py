"""Run-length helpers for 1D boolean masks shared across the pipeline."""

import numpy as np


def find_true_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Locate maximal runs of True values in a 1D boolean mask.

    Args:
        mask: Boolean array of shape (n,). May be empty.

    Returns:
        Tuple (starts, ends) of int64 arrays describing half-open spans [start, end).
    """
    flags = np.asarray(mask, dtype=bool).ravel()
    if flags.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    edges = np.diff(np.concatenate(([0], flags.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1).astype(np.int64)
    ends = np.flatnonzero(edges == -1).astype(np.int64)
    return starts, ends


def remove_short_spans(mask: np.ndarray, min_length: int) -> np.ndarray:
    """Return a copy of mask with True runs shorter than min_length cleared."""
    out = np.array(mask, dtype=bool)
    starts, ends = find_true_runs(out)
    short = (ends - starts) < min_length
    for start, end in zip(starts[short].tolist(), ends[short].tolist()):
        out[start:end] = False
    return out


def fill_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Return a copy of mask with interior False gaps of length <= max_gap set to True.

    Gaps touching either end of the array are never filled.
    """
    out = np.array(mask, dtype=bool)
    starts, ends = find_true_runs(~out)
    fillable = (starts > 0) & (ends < len(out)) & ((ends - starts) <= max_gap)
    for start, end in zip(starts[fillable].tolist(), ends[fillable].tolist()):
        out[start:end] = True
    return out
