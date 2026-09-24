"""Forecast accuracy metrics.

WAPE is the primary business metric. MAE, RMSE, sMAPE, and bias are always
reported beside it. Pinball loss is reported only when a quantile forecast is
present. A zero sum of actual demand is an error for WAPE and bias.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]


def mae(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """Return the mean absolute error."""

    actual_values, forecast_values = _pair(actual, forecast, "MAE")
    return float(np.mean(np.abs(actual_values - forecast_values)))


def rmse(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """Return the root mean squared error."""

    actual_values, forecast_values = _pair(actual, forecast, "RMSE")
    return float(np.sqrt(np.mean(np.square(actual_values - forecast_values))))


def wape(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """Return the weighted absolute percentage error.

    ``sum(|actual - forecast|) / sum(actual)``. The denominator is the sum of
    actual demand. A sum of zero raises ``ValueError`` instead of returning 0.
    """

    actual_values, forecast_values = _pair(actual, forecast, "WAPE")
    denominator = _actual_sum(actual_values, "WAPE")
    return float(np.sum(np.abs(actual_values - forecast_values)) / denominator)


def smape(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """Return the symmetric mean absolute percentage error.

    Each term is ``2 * |actual - forecast| / (|actual| + |forecast|)``. A term
    is 0 when both values are 0. The result is a fraction, not a percent.
    """

    actual_values, forecast_values = _pair(actual, forecast, "sMAPE")
    denominator = np.abs(actual_values) + np.abs(forecast_values)
    loss = np.zeros(actual_values.shape, dtype=np.float64)
    present = denominator > 0.0
    loss[present] = (
        2.0 * np.abs(actual_values[present] - forecast_values[present]) / denominator[present]
    )
    return float(np.mean(loss))


def bias(actual: Sequence[float], forecast: Sequence[float]) -> float:
    """Return ``sum(forecast - actual) / sum(actual)``.

    Positive bias means the forecast is high. A sum of actual demand of zero
    raises ``ValueError``.
    """

    actual_values, forecast_values = _pair(actual, forecast, "bias")
    denominator = _actual_sum(actual_values, "Bias")
    return float(np.sum(forecast_values - actual_values) / denominator)


def pinball_loss(actual: Sequence[float], forecast: Sequence[float], quantile: float) -> float:
    """Return the mean pinball loss at ``quantile``.

    For gap ``actual - forecast``, the loss is ``max(q * gap, (q - 1) * gap)``.
    Pass 0.1 with ``p10`` and 0.9 with ``p90``.
    """

    if not 0.0 < quantile < 1.0:
        raise ValueError("Pinball loss quantile must be between 0 and 1.")
    actual_values, forecast_values = _pair(actual, forecast, "pinball loss")
    gap = actual_values - forecast_values
    loss = np.maximum(quantile * gap, (quantile - 1.0) * gap)
    return float(np.mean(loss))


def _pair(actual: Sequence[float], forecast: Sequence[float], metric: str) -> tuple[F64, F64]:
    actual_values = _as_float(actual, "Actual")
    forecast_values = _as_float(forecast, "Forecast")
    if actual_values.shape != forecast_values.shape:
        raise ValueError(f"{metric} requires actual and forecast sequences of the same length.")
    if actual_values.size == 0:
        raise ValueError(f"Cannot compute {metric} for an empty forecast.")
    return actual_values, forecast_values


def _actual_sum(actual: F64, metric: str) -> float:
    total = float(np.sum(actual))
    if total == 0.0:
        raise ValueError(f"{metric} is undefined because the sum of actual demand is zero.")
    return total


def _as_float(values: Sequence[float], name: str) -> F64:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence.")
    if array.size and not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} contains a non-finite value.")
    return array
