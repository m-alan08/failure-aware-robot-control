"""Bootstrap intervals.

`bootstrap_confidence_interval` is migrated from quantrl-fetch src/quantrl/eval/stats.py
(commit f2591d0); `hierarchical_bootstrap_interval` from src/quantrl/disturbances/statistics.py
(commit fad4381). See PROVENANCE.md.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bootstrap_confidence_interval(
    values: Sequence[float] | np.ndarray,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    resample_indices = rng.integers(0, data.size, size=(resamples, data.size))
    resample_means = data[resample_indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(resample_means, [tail, 1.0 - tail])
    return float(low), float(high)


def hierarchical_bootstrap_interval(
    groups: Sequence[np.ndarray],
    resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Two-level percentile bootstrap of the pooled mean.

    Each resample draws training seeds (groups) with replacement and then, independently for
    every drawn seed, its episodes with replacement. With 3 training seeds the outer level is
    coarse; the interval is a rough description of seed-plus-episode variability, not a test.
    For a paired difference, pass per-episode differences (condition minus baseline on the
    same episode seed): resampling an episode then resamples the pair.
    """
    groups = [np.asarray(group, dtype=float) for group in groups]
    if not groups or all(group.size == 0 for group in groups):
        return float("nan"), float("nan")
    count = len(groups)
    sums = np.zeros((count, resamples, count))
    sizes = np.array([group.size for group in groups], dtype=float)
    for index, group in enumerate(groups):
        if group.size:
            picks = rng.integers(0, group.size, size=(resamples, count, group.size))
            sums[index] = group[picks].sum(axis=-1)
    chosen = rng.integers(0, count, size=(resamples, count))
    rows = np.arange(resamples)[:, None]
    slots = np.arange(count)[None, :]
    total = sums[chosen, rows, slots].sum(axis=1)
    weight = sizes[chosen].sum(axis=1)
    valid = weight > 0
    means = total[valid] / weight[valid]
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)
