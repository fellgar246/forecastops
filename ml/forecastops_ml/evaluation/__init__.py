"""Rolling-origin backtests and forecast metrics.

:class:`Backtester` fits a :class:`Forecaster` on each origin and scores the
pooled forecasts. A published score uses at least three folds.
"""

from forecastops_ml.evaluation.backtest import (
    Backtester,
    EvaluationReport,
    Forecaster,
    ForecastFrame,
    MetricSummary,
    SkippedSeries,
    SliceMetrics,
    UnscoredModelError,
    demand_quartiles,
    make_series_id,
    split_fold,
)
from forecastops_ml.evaluation.folds import (
    MIN_PUBLISHED_FOLDS,
    PREFERRED_BENCHMARK_FOLDS,
    Fold,
    build_folds,
)
from forecastops_ml.evaluation.metrics import bias, mae, pinball_loss, rmse, smape, wape

__all__ = [
    "MIN_PUBLISHED_FOLDS",
    "PREFERRED_BENCHMARK_FOLDS",
    "Backtester",
    "EvaluationReport",
    "Fold",
    "ForecastFrame",
    "Forecaster",
    "MetricSummary",
    "SkippedSeries",
    "SliceMetrics",
    "UnscoredModelError",
    "bias",
    "build_folds",
    "demand_quartiles",
    "mae",
    "make_series_id",
    "pinball_loss",
    "rmse",
    "smape",
    "split_fold",
    "wape",
]
