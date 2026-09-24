"""Reference report used when no production model is available.

The gate compares a candidate with the current production evaluation. When
that model does not exist, the reference is seasonal naive scored on the
same observation frame and dataset version.
"""

from datetime import date
from statistics import median

import pyarrow as pa

from forecastops_ml.baselines.naive import (
    SeasonalNaiveForecaster,
    weekly_seasonal_naive,
    yearly_seasonal_naive,
)
from forecastops_ml.evaluation.backtest import Backtester, EvaluationReport


def reference_report(
    frame: pa.Table,
    *,
    folds: int,
    horizon: int,
    dataset_version: str | None,
    production: EvaluationReport | None = None,
) -> EvaluationReport:
    """Return the production report, or score seasonal naive on ``frame``.

    ``production`` is the current production evaluation. When it is omitted,
    seasonal naive is fit on ``frame`` with the same folds, horizon, and
    dataset version the candidate used. Daily frames use a 7-day lag. Weekly
    frames use a 52-week lag.
    """

    if production is not None:
        _require_matching_version(production, dataset_version)
        return production
    forecaster = seasonal_naive_for(frame)
    return Backtester().evaluate(
        forecaster,
        frame,
        folds,
        horizon,
        dataset_version=dataset_version,
    )


def seasonal_naive_for(frame: pa.Table) -> SeasonalNaiveForecaster:
    """Return the seasonal naive forecaster that matches ``frame`` grain."""

    if _is_weekly(frame):
        return yearly_seasonal_naive()
    return weekly_seasonal_naive()


def _require_matching_version(production: EvaluationReport, dataset_version: str | None) -> None:
    if dataset_version is None or production.dataset_version is None:
        return
    if production.dataset_version != dataset_version:
        raise ValueError("Production report dataset version does not match the candidate dataset.")


def _is_weekly(frame: pa.Table) -> bool:
    if "date" not in frame.column_names:
        raise ValueError("Observation table is missing date.")
    values = frame.column("date").to_pylist()
    dates = sorted({value for value in values if type(value) is date})
    if len(dates) < 2:
        return False
    gaps = [(dates[index + 1] - dates[index]).days for index in range(len(dates) - 1)]
    return median(gaps) >= 6
