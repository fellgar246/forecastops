"""Forecast equations for naive baselines and the local comparison table."""

import json
from datetime import date, timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from forecastops_contracts import SCHEMA_VERSION
from forecastops_ml.baselines import (
    COMPARISON_FILENAME,
    WEEKLY_SEASON_LENGTH,
    YEARLY_SEASON_LENGTH,
    ComparisonTable,
    ModelCatalog,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    local_catalog,
    weekly_seasonal_naive,
    write_comparison,
    yearly_season_available,
    yearly_seasonal_naive,
)
from forecastops_ml.data.synthetic import DEFAULT_MAX_ROWS, DEFAULT_SEED, generate_dataset
from forecastops_ml.evaluation import MIN_PUBLISHED_FOLDS

START = date(2026, 1, 1)
SERIES = "store-01|sku-1"


def test_naive_repeats_the_last_point_with_null_quantiles() -> None:
    model = NaiveForecaster()
    model.fit(
        _frame([(START, 4), (START + timedelta(days=1), 10)]),
        None,
    )
    forecast = model.predict(3, _future(START + timedelta(days=2), 3))

    assert forecast.p50 == (10.0, 10.0, 10.0)
    assert forecast.p10 is None
    assert forecast.p90 is None
    assert model.fallback_count == 0


def test_weekly_seasonal_naive_uses_lag_7_on_daily_data() -> None:
    history = [(START + timedelta(days=offset), offset + 1) for offset in range(14)]
    target = START + timedelta(days=14)
    model = weekly_seasonal_naive()
    model.fit(_frame(history), None)
    forecast = model.predict(1, _future(target, 1))

    assert model.season_length == WEEKLY_SEASON_LENGTH
    assert model.period == "day"
    assert forecast.p50 == (float(history[7][1]),)
    assert forecast.p10 is None
    assert forecast.p90 is None
    assert model.fallback_count == 0


def test_yearly_seasonal_naive_uses_lag_52_weeks() -> None:
    origin = date(2025, 1, 6)
    assert origin.weekday() == 0
    history = [(origin + timedelta(weeks=offset), 100 + offset) for offset in range(53)]
    target = origin + timedelta(weeks=52)
    model = yearly_seasonal_naive()
    model.fit(_frame(history), {"season_length": 52})
    forecast = model.predict(1, _future(target, 1))

    assert model.season_length == YEARLY_SEASON_LENGTH
    assert model.period == "week"
    assert forecast.p50 == (100.0,)
    assert forecast.p10 is None
    assert forecast.p90 is None
    assert model.fallback_count == 0


def test_short_history_falls_back_to_the_last_observation() -> None:
    model = SeasonalNaiveForecaster(WEEKLY_SEASON_LENGTH)
    model.fit(
        _frame([(START, 1), (START + timedelta(days=1), 2), (START + timedelta(days=2), 9)]),
        None,
    )
    forecast = model.predict(3, _future(START + timedelta(days=3), 3))

    assert forecast.p50 == (9.0, 9.0, 9.0)
    assert model.fallback_count == 3


def test_partial_lag_falls_back_only_for_missing_steps() -> None:
    history = [(START + timedelta(days=offset), offset + 1) for offset in range(8)]
    model = weekly_seasonal_naive()
    model.fit(_frame(history), {"fold": 1})
    forecast = model.predict(2, _future(START + timedelta(days=8), 2))

    # Day 8 lags day 1 (value 2). Day 9 lags day 2 (value 3).
    assert forecast.p50 == (2.0, 3.0)
    assert model.fallback_count == 0

    model.fit(_frame(history[:3]), {"fold": 2})
    short = model.predict(2, _future(START + timedelta(days=3), 2))
    assert short.p50 == (3.0, 3.0)
    assert model.fallback_count == 2


def test_fallback_repeats_a_zero_last_observation() -> None:
    model = weekly_seasonal_naive()
    model.fit(_frame([(START, 0)]), None)
    forecast = model.predict(1, _future(START + timedelta(days=1), 1))

    assert forecast.p50 == (0.0,)
    assert model.fallback_count == 1


def test_missing_series_is_an_error() -> None:
    model = NaiveForecaster()
    model.fit(_frame([(START, 10)]), None)
    future = pa.table(
        {
            "series_id": pa.array(["store-01|other"], type=pa.string()),
            "date": pa.array([START + timedelta(days=1)], type=pa.date32()),
        }
    )
    with pytest.raises(ValueError, match="No training observation"):
        model.predict(1, future)


def test_fold_one_restarts_fallback_count() -> None:
    model = weekly_seasonal_naive()
    model.fit(_frame([(START, 4)]), {"fold": 1})
    model.predict(1, _future(START + timedelta(days=1), 1))
    assert model.fallback_count == 1

    model.fit(_frame([(START, 4)]), {"fold": 1})
    assert model.fallback_count == 0


def test_catalog_comparison_has_one_row_per_family(tmp_path) -> None:
    generate_dataset(tmp_path, profile="test", seed=DEFAULT_SEED, max_rows=DEFAULT_MAX_ROWS)
    frame = pq.read_table(tmp_path / "observations.parquet")
    dates = frame.column("date").to_pylist()
    assert yearly_season_available(dates) is False

    catalog = local_catalog()
    assert catalog.families() == ("naive", "seasonal_naive", "holt_winters", "gradient_boosting")
    seasonal = catalog.get("seasonal_naive")
    assert isinstance(seasonal, SeasonalNaiveForecaster)
    assert seasonal.season_length == WEEKLY_SEASON_LENGTH

    table = catalog.compare(
        frame,
        folds=MIN_PUBLISHED_FOLDS,
        horizon=7,
        dataset_version=SCHEMA_VERSION,
    )
    destination = write_comparison(table, tmp_path)
    document = json.loads(destination.read_text(encoding="utf-8"))

    assert destination.name == COMPARISON_FILENAME
    assert document["dataset_version"] == SCHEMA_VERSION
    assert document["fold_count"] == 3
    assert document["horizon"] == 7
    assert [row["family"] for row in document["models"]] == ["naive", "seasonal_naive"]
    for row in document["models"]:
        assert set(row) == {"bias", "family", "runtime_ms", "smape", "wape"}
        assert row["wape"] >= 0.0
        assert row["runtime_ms"] >= 0
    assert "p10" not in destination.read_text(encoding="utf-8")
    assert "p90" not in destination.read_text(encoding="utf-8")
    assert table == ComparisonTable(
        models=table.models,
        fold_count=3,
        horizon=7,
        dataset_version=SCHEMA_VERSION,
    )


def test_catalog_rejects_a_mismatched_family() -> None:
    with pytest.raises(ValueError, match="does not match model_family"):
        ModelCatalog({"seasonal_naive": NaiveForecaster()})


def _frame(points: list[tuple[date, int]]) -> pa.Table:
    return pa.table(
        {
            "series_id": pa.array([SERIES] * len(points), type=pa.string()),
            "date": pa.array([day for day, _sold in points], type=pa.date32()),
            "units_sold": pa.array([sold for _day, sold in points], type=pa.int64()),
        }
    )


def _future(start: date, steps: int) -> pa.Table:
    days = [start + timedelta(days=offset) for offset in range(steps)]
    return pa.table(
        {
            "series_id": pa.array([SERIES] * steps, type=pa.string()),
            "date": pa.array(days, type=pa.date32()),
        }
    )
