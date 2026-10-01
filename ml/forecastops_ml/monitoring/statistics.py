"""Distribution comparisons for drift monitoring.

Population stability index (PSI) bins a numeric baseline and compares those
bin shares with a recent window. The Kolmogorov-Smirnov statistic is the
largest gap between the two empirical distribution functions. Categorical
coverage uses the absolute difference of shares.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]

# Empty bins would make the PSI logarithm undefined. A share below this floor
# is raised to the floor, then both share vectors are scaled to sum to 1.
PSI_EPSILON = 1e-4


def population_stability_index(
    baseline: Sequence[float],
    recent: Sequence[float],
    *,
    edges: Sequence[float] | None = None,
    bin_count: int = 10,
) -> float:
    """Return the PSI of ``recent`` against ``baseline``.

    Bin edges come from ``edges`` when those are supplied. Otherwise they are
    the unique quantiles of ``baseline`` at ``bin_count`` intervals. Recent
    values outside the outer edges stay in the first or last bin. A constant
    baseline uses one bin for that value and a second bin for every other
    value.
    """

    baseline_values = _numbers(baseline, "PSI baseline")
    recent_values = _numbers(recent, "PSI recent")
    resolved = _edges(baseline_values, edges, bin_count)
    baseline_shares = _shares(baseline_values, resolved)
    recent_shares = _shares(recent_values, resolved)
    ratio = recent_shares / baseline_shares
    return float(np.sum((recent_shares - baseline_shares) * np.log(ratio)))


def kolmogorov_smirnov(baseline: Sequence[float], recent: Sequence[float]) -> float:
    """Return the two-sample Kolmogorov-Smirnov statistic.

    The statistic is the maximum of ``|F_baseline(x) - F_recent(x)|`` over the
    combined sample. ``F(x)`` is the share of observations at or below ``x``.
    """

    baseline_values = np.sort(_numbers(baseline, "KS baseline"))
    recent_values = np.sort(_numbers(recent, "KS recent"))
    points = np.unique(np.concatenate([baseline_values, recent_values]))
    baseline_cdf = np.searchsorted(baseline_values, points, side="right") / baseline_values.size
    recent_cdf = np.searchsorted(recent_values, points, side="right") / recent_values.size
    return float(np.max(np.abs(baseline_cdf - recent_cdf)))


def absolute_frequency_deltas(
    baseline: Sequence[str],
    recent: Sequence[str],
) -> dict[str, float]:
    """Return ``|recent share - baseline share|`` for every observed label.

    Labels that appear on only one side use a share of zero on the other side.
    The result is keyed in label order.
    """

    baseline_shares = _label_shares(baseline, "baseline labels")
    recent_shares = _label_shares(recent, "recent labels")
    labels = sorted(set(baseline_shares) | set(recent_shares))
    return {
        label: abs(recent_shares.get(label, 0.0) - baseline_shares.get(label, 0.0))
        for label in labels
    }


def frequency_delta(baseline: Sequence[bool], recent: Sequence[bool]) -> float:
    """Return the absolute difference between two boolean rates."""

    return abs(_rate(baseline, "baseline flags") - _rate(recent, "recent flags"))


def _numbers(values: Sequence[float], label: str) -> F64:
    if isinstance(values, str | bytes):
        raise ValueError(f"{label} must be a sequence of finite numbers.")
    if any(isinstance(item, bool) or not isinstance(item, int | float) for item in values):
        raise ValueError(f"{label} must be a sequence of finite numbers.")
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{label} must contain at least one value.")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be a sequence of finite numbers.")
    return array


def _edges(baseline: F64, edges: Sequence[float] | None, bin_count: int) -> F64:
    if edges is not None:
        return _explicit_edges(edges)
    if bin_count < 2:
        raise ValueError("PSI needs at least two bins.")
    probabilities = np.linspace(0.0, 1.0, bin_count + 1)
    quantiles = np.unique(np.asarray(np.quantile(baseline, probabilities), dtype=np.float64))
    if quantiles.size < 2:
        return np.asarray(quantiles, dtype=np.float64)
    return quantiles


def _explicit_edges(edges: Sequence[float]) -> F64:
    if isinstance(edges, str | bytes):
        raise ValueError("PSI edges must be strictly increasing finite numbers.")
    if any(isinstance(item, bool) or not isinstance(item, int | float) for item in edges):
        raise ValueError("PSI edges must be strictly increasing finite numbers.")
    resolved = np.asarray(list(edges), dtype=np.float64)
    if resolved.size < 2 or not np.isfinite(resolved).all() or np.any(np.diff(resolved) <= 0.0):
        raise ValueError("PSI edges must be strictly increasing finite numbers.")
    return resolved


def _shares(values: F64, edges: F64) -> F64:
    if edges.size == 1:
        at_center = int(np.count_nonzero(values == float(edges[0])))
        counts = np.array([at_center, values.size - at_center], dtype=np.float64)
    else:
        cuts = edges[1:-1]
        indexes = np.digitize(values, cuts, right=False)
        counts = np.bincount(indexes, minlength=int(cuts.size) + 1).astype(np.float64)
    shares = counts / counts.sum()
    shares = np.maximum(shares, PSI_EPSILON)
    total = float(shares.sum())
    return np.asarray(shares / total, dtype=np.float64)


def _label_shares(values: Sequence[str], label: str) -> dict[str, float]:
    if isinstance(values, str | bytes):
        raise ValueError(f"{label} must be a sequence of strings.")
    if any(not isinstance(item, str) for item in values):
        raise ValueError(f"{label} must be a sequence of strings.")
    if len(values) == 0:
        raise ValueError(f"{label} must contain at least one value.")
    counts: dict[str, int] = {}
    for item in values:
        counts[item] = counts.get(item, 0) + 1
    total = float(len(values))
    return {key: count / total for key, count in counts.items()}


def _rate(values: Sequence[bool], label: str) -> float:
    if isinstance(values, str | bytes):
        raise ValueError(f"{label} must be a sequence of booleans.")
    if any(type(item) is not bool for item in values):
        raise ValueError(f"{label} must be a sequence of booleans.")
    if len(values) == 0:
        raise ValueError(f"{label} must contain at least one value.")
    return float(sum(1 for item in values if item) / len(values))
