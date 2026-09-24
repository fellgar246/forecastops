"""Holt-Winters on category-week series, including skips and comparison."""

import math
from datetime import date, timedelta

import numpy as np
import pyarrow as pa
import pytest
from numpy.typing import NDArray

from forecastops_ml.baselines import (
    MIN_HISTORY_WEEKS,
    HoltWintersForecaster,
    ModelCatalog,
    NaiveForecaster,
)
from forecastops_ml.baselines import holt_winters as holt_winters_module
from forecastops_ml.evaluation import MIN_PUBLISHED_FOLDS, Backtester

F64 = NDArray[np.float64]

START = date(2024, 1, 1)
HORIZON = 4


def test_seasonal_category_week_forecast_is_finite() -> None:
    assert START.weekday() == 0
    model = HoltWintersForecaster()
    model.fit(_category_week("beverages", MIN_HISTORY_WEEKS), None)
    forecast = model.predict(HORIZON, _future("beverages", MIN_HISTORY_WEEKS, HORIZON))

    assert model.model_family == "holt_winters"
    assert len(forecast.p50) == HORIZON
    assert all(math.isfinite(value) for value in forecast.p50)
    assert forecast.p10 is None
    assert forecast.p90 is None
    assert forecast.series_id == ("beverages",) * HORIZON
    assert model.fit_runtime_ms > 0
    assert model.skipped_series == ()


def test_short_series_is_skipped() -> None:
    model = HoltWintersForecaster()
    model.fit(_category_week("beverages", 10), None)

    assert len(model.skipped_series) == 1
    assert model.skipped_series[0].series_id == "beverages"
    assert "fewer than 104" in model.skipped_series[0].reason
    with pytest.raises(ValueError, match="fewer than 104"):
        model.predict(1, _future("beverages", 10, 1))


def test_short_category_is_skipped_and_long_category_scores() -> None:
    long_weeks = MIN_HISTORY_WEEKS + MIN_PUBLISHED_FOLDS
    frame = _panel(
        {
            "beverages": long_weeks,
            "snacks": 12,
        }
    )
    report = Backtester().evaluate(
        HoltWintersForecaster(),
        frame,
        folds=MIN_PUBLISHED_FOLDS,
        horizon=1,
    )

    assert report.model_family == "holt_winters"
    assert report.global_metrics.row_count == MIN_PUBLISHED_FOLDS
    assert report.runtime_ms > 0
    assert [step for step, _wape in report.wape_by_horizon] == [1]
    assert report.global_metrics.pinball_loss_p10 is None
    assert report.global_metrics.pinball_loss_p90 is None
    assert [item.series_id for item in report.skipped_series] == ["all|snacks"]
    assert "fewer than 104" in report.skipped_series[0].reason
    assert {item.key for item in report.by_category} == {"beverages"}


def test_convergence_failure_skips_one_category_and_comparison_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_fit = holt_winters_module._fit_category

    def fail_snacks(category_id: str, values: F64) -> object:
        if category_id == "snacks":
            raise ValueError("optimization failed")
        return real_fit(category_id, values)

    monkeypatch.setattr(holt_winters_module, "_fit_category", fail_snacks)
    long_weeks = MIN_HISTORY_WEEKS + MIN_PUBLISHED_FOLDS
    frame = _panel({"beverages": long_weeks, "snacks": long_weeks})
    catalog = ModelCatalog(
        {
            "naive": NaiveForecaster(),
            "holt_winters": HoltWintersForecaster(),
        }
    )
    table = catalog.compare(frame, folds=MIN_PUBLISHED_FOLDS, horizon=1)
    holt = table.models[1]

    assert [row.family for row in table.models] == ["naive", "holt_winters"]
    assert holt.wape >= 0.0
    assert isinstance(holt.bias, float)
    assert holt.runtime_ms > 0


def _category_week(category_id: str, weeks: int) -> pa.Table:
    return pa.table(
        {
            "category_id": pa.array([category_id] * weeks, type=pa.string()),
            "week_start": pa.array(_mondays(weeks), type=pa.date32()),
            "units_sold": pa.array(
                [_seasonal_units(offset) for offset in range(weeks)],
                type=pa.int64(),
            ),
        }
    )


def _future(category_id: str, origin_week: int, horizon: int) -> pa.Table:
    days = [_mondays(origin_week + horizon)[origin_week + offset] for offset in range(horizon)]
    return pa.table(
        {
            "category_id": pa.array([category_id] * horizon, type=pa.string()),
            "week_start": pa.array(days, type=pa.date32()),
        }
    )


def _panel(lengths: dict[str, int]) -> pa.Table:
    dates: list[date] = []
    stores: list[str] = []
    sku_ids: list[str] = []
    category_ids: list[str] = []
    units: list[int] = []
    for category_id, weeks in lengths.items():
        for offset, day in enumerate(_mondays(weeks)):
            dates.append(day)
            stores.append("all")
            sku_ids.append(category_id)
            category_ids.append(category_id)
            units.append(_seasonal_units(offset))
    return pa.table(
        {
            "date": pa.array(dates, type=pa.date32()),
            "store_id": pa.array(stores, type=pa.string()),
            "sku_id": pa.array(sku_ids, type=pa.string()),
            "category_id": pa.array(category_ids, type=pa.string()),
            "units_sold": pa.array(units, type=pa.int64()),
        }
    )


def _mondays(weeks: int) -> list[date]:
    return [START + timedelta(weeks=offset) for offset in range(weeks)]


def _seasonal_units(offset: int) -> int:
    return int(round(80 + 25 * math.sin(2 * math.pi * offset / 52)))
