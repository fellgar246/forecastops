"""Rolling-origin folds, hand-computed metrics, and stub forecasters."""

import json
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from forecastops_contracts import SCHEMA_VERSION
from forecastops_ml import evaluation
from forecastops_ml.data.synthetic import DEFAULT_MAX_ROWS, DEFAULT_SEED, generate_dataset
from forecastops_ml.evaluation import (
    MIN_PUBLISHED_FOLDS,
    PREFERRED_BENCHMARK_FOLDS,
    Backtester,
    ForecastFrame,
    SliceMetrics,
    bias,
    build_folds,
    demand_quartiles,
    mae,
    pinball_loss,
    rmse,
    smape,
    split_fold,
    wape,
)

START = date(2026, 1, 1)
ACTUAL = [10.0, 20.0, 30.0, 40.0]
FORECAST = [13.0, 15.0, 36.0, 30.0]
P10 = [8.0, 22.0, 28.0, 50.0]
P90 = [14.0, 15.0, 36.0, 30.0]


class LastValueForecaster:
    """Repeats each series' latest training observation across the horizon."""

    model_family = "naive_stub"

    def __init__(self) -> None:
        self._last: dict[str, float] = {}

    def fit(self, history: pa.Table, config: Mapping[str, object] | None) -> None:
        series_ids = history.column("series_id").to_pylist()
        days = history.column("date").to_pylist()
        units = history.column("units_sold").to_pylist()
        order = sorted(range(len(series_ids)), key=lambda index: (series_ids[index], days[index]))
        last: dict[str, float] = {}
        for index in order:
            last[str(series_ids[index])] = float(units[index])
        self._last = last

    def predict(self, horizon: int, known_future: pa.Table) -> ForecastFrame:
        series_ids = tuple(str(value) for value in known_future.column("series_id").to_pylist())
        days = tuple(known_future.column("date").to_pylist())
        missing = sorted({series_id for series_id in series_ids if series_id not in self._last})
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"No training observation for {joined}.")
        return ForecastFrame(
            series_id=series_ids,
            date=days,
            p50=tuple(self._last[series_id] for series_id in series_ids),
        )


class ConstantForecaster:
    """Emits one p50 for every known-future row, with optional quantiles."""

    def __init__(
        self,
        value: float,
        *,
        family: str = "constant_stub",
        p10: float | None = None,
        p90: float | None = None,
    ) -> None:
        self._value = value
        self.model_family = family
        self._p10 = p10
        self._p90 = p90

    def fit(self, history: pa.Table, config: Mapping[str, object] | None) -> None:
        return None

    def predict(self, horizon: int, known_future: pa.Table) -> ForecastFrame:
        series_ids = tuple(str(value) for value in known_future.column("series_id").to_pylist())
        days = tuple(known_future.column("date").to_pylist())
        count = len(series_ids)
        return ForecastFrame(
            series_id=series_ids,
            date=days,
            p50=tuple([self._value] * count),
            p10=None if self._p10 is None else tuple([self._p10] * count),
            p90=None if self._p90 is None else tuple([self._p90] * count),
        )


class RecordingForecaster:
    """Records the dates passed to fit and predict."""

    model_family = "recorder"

    def __init__(self) -> None:
        self.train_dates: list[set[date]] = []
        self.future_dates: list[set[date]] = []
        self.future_has_target = False

    def fit(self, history: pa.Table, config: Mapping[str, object] | None) -> None:
        self.train_dates.append(set(history.column("date").to_pylist()))

    def predict(self, horizon: int, known_future: pa.Table) -> ForecastFrame:
        if "units_sold" in known_future.column_names:
            self.future_has_target = True
        series_ids = tuple(str(value) for value in known_future.column("series_id").to_pylist())
        days = tuple(known_future.column("date").to_pylist())
        self.future_dates.append(set(days))
        return ForecastFrame(series_id=series_ids, date=days, p50=tuple(1.0 for _ in series_ids))


def test_metrics_match_hand_computed_values() -> None:
    # |10-13| + |20-15| + |30-36| + |40-30| = 3 + 5 + 6 + 10 = 24
    # sum(actual) = 100
    # squares = 9 + 25 + 36 + 100 = 170
    # signed forecast - actual = 3 - 5 + 6 - 10 = -6
    assert mae(ACTUAL, FORECAST) == pytest.approx(24 / 4)
    assert rmse(ACTUAL, FORECAST) == pytest.approx(math.sqrt(170 / 4))
    assert wape(ACTUAL, FORECAST) == pytest.approx(24 / 100)
    assert bias(ACTUAL, FORECAST) == pytest.approx(-6 / 100)
    assert smape(ACTUAL, FORECAST) == pytest.approx((6 / 23 + 2 / 7 + 2 / 11 + 2 / 7) / 4)

    # p10 losses: 0.1*2, 0.9*2, 0.1*2, 0.9*10 = 0.2 + 1.8 + 0.2 + 9.0
    assert pinball_loss(ACTUAL, P10, 0.1) == pytest.approx(11.2 / 4)
    # p90 losses: 0.1*4, 0.9*5, 0.1*6, 0.9*10 = 0.4 + 4.5 + 0.6 + 9.0
    assert pinball_loss(ACTUAL, P90, 0.9) == pytest.approx(14.5 / 4)


def test_smape_treats_a_zero_zero_term_as_zero() -> None:
    assert smape([0.0, 10.0], [0.0, 10.0]) == pytest.approx(0.0)


def test_zero_actual_sum_is_an_error_for_wape_and_bias() -> None:
    with pytest.raises(
        ValueError, match="WAPE is undefined because the sum of actual demand is zero"
    ):
        wape([0.0, 0.0], [1.0, 2.0])
    with pytest.raises(
        ValueError, match="Bias is undefined because the sum of actual demand is zero"
    ):
        bias([0.0, 0.0], [1.0, 2.0])


def test_folds_move_forward_and_keep_training_rows_out_of_validation() -> None:
    frame = _panel(n_days=10, skus=("sku-1", "sku-2"), units_for=lambda _sku, _offset: 1)
    folds = build_folds(_dates(frame), folds=3, horizon=2)
    previous_origin: date | None = None
    covered: set[date] = set()
    for fold in folds:
        train, valid = split_fold(frame, fold)
        assert _keys(train).isdisjoint(_keys(valid))
        assert train.num_rows > 0
        assert all(day < fold.origin for day in train.column("date").to_pylist())
        assert set(valid.column("date").to_pylist()) == set(fold.validation_dates)
        assert min(valid.column("date").to_pylist()) == fold.origin
        if previous_origin is not None:
            assert fold.validation_dates[0] > previous_origin
        assert covered.isdisjoint(fold.validation_dates)
        covered.update(fold.validation_dates)
        previous_origin = fold.origin

    recorder = RecordingForecaster()
    Backtester().evaluate(recorder, frame, folds=3, horizon=2)
    assert recorder.future_has_target is False
    assert len(recorder.train_dates) == 3
    for train_dates, future_dates in zip(recorder.train_dates, recorder.future_dates, strict=True):
        assert train_dates.isdisjoint(future_dates)
        assert max(train_dates) < min(future_dates)
    origins = [min(days) for days in recorder.future_dates]
    assert origins[1] > origins[0]
    assert origins[2] > origins[1]


def test_demand_quartiles_follow_training_totals() -> None:
    # Eight training days: W=8, X=80, Y=160, Z=320.
    # Linear edges are 62, 120, and 200.
    assert demand_quartiles(["W", "X", "Y", "Z"], [8, 80, 160, 320]) == {
        "W": "q1",
        "X": "q2",
        "Y": "q3",
        "Z": "q4",
    }
    # Adding W's validation spike moves W out of q1 and X into it.
    leaked = demand_quartiles(["W", "X", "Y", "Z"], [208, 100, 200, 400])
    assert leaked["W"] == "q3"
    assert leaked["X"] == "q1"


def test_backtester_quartiles_ignore_validation_demand() -> None:
    frame = _panel(n_days=14, skus=("W", "X", "Y", "Z"), units_for=_quartile_units)
    report = Backtester().evaluate(LastValueForecaster(), frame, folds=3, horizon=2)
    quartile = {item.key: item.metrics for item in report.by_demand_quartile}

    # Fold 1 trains on days 0-7, so W is q1 and its last value is 1.
    # Validation actuals are 100 and 100. Later folds label X as q1 with a
    # perfect last-value forecast. Pooled q1 absolute error is 99 + 99.
    # Pooled q1 actuals are 200 + 20 + 20.
    assert quartile["q1"].row_count == 6
    assert quartile["q1"].wape == pytest.approx(198 / 240)


def test_naive_stub_matches_a_hand_computed_wape() -> None:
    frame = _panel(n_days=10, skus=("sku-1",), units_for=lambda _sku, offset: offset + 1)
    report = Backtester().evaluate(LastValueForecaster(), frame, folds=3, horizon=2)

    # Last training values are 4, 6, and 8. Validation actuals are 5, 6, 7, 8, 9, 10.
    # Absolute errors are 1, 2, 1, 2, 1, 2. Sum of actuals is 45.
    assert report.model_family == "naive_stub"
    assert report.fold_count == 3
    assert report.horizon == 2
    assert report.global_metrics.row_count == 6
    assert report.global_metrics.wape == pytest.approx(9 / 45)
    assert report.global_metrics.mae == pytest.approx(9 / 6)
    assert report.global_metrics.bias == pytest.approx(-9 / 45)
    assert report.global_metrics.rmse == pytest.approx(math.sqrt(15 / 6))
    assert report.global_metrics.smape == pytest.approx(
        (2 / 9 + 2 / 5 + 2 / 13 + 2 / 7 + 2 / 17 + 2 / 9) / 6
    )
    assert report.global_metrics.pinball_loss_p10 is None
    assert report.global_metrics.pinball_loss_p90 is None


def test_quantile_stub_reports_hand_computed_pinball_loss() -> None:
    frame = _panel(n_days=10, skus=("sku-1",), units_for=lambda _sku, _offset: 10)
    report = Backtester().evaluate(
        ConstantForecaster(10.0, family="quantile_stub", p10=8.0, p90=12.0),
        frame,
        folds=3,
        horizon=2,
    )

    # Every actual is 10. p10=8 contributes 0.1 * 2. p90=12 contributes 0.1 * 2.
    assert report.global_metrics.pinball_loss_p10 == pytest.approx(0.2)
    assert report.global_metrics.pinball_loss_p90 == pytest.approx(0.2)
    assert report.global_metrics.wape == pytest.approx(0.0)


def test_constant_and_naive_stubs_score_the_test_fixture(tmp_path: Path) -> None:
    generate_dataset(tmp_path, profile="test", seed=DEFAULT_SEED, max_rows=DEFAULT_MAX_ROWS)
    frame = pq.read_table(tmp_path / "observations.parquet")
    backtester = Backtester()

    naive = backtester.evaluate(
        LastValueForecaster(),
        frame,
        folds=MIN_PUBLISHED_FOLDS,
        horizon=7,
        dataset_version=SCHEMA_VERSION,
    )
    constant = backtester.evaluate(
        ConstantForecaster(0.0),
        frame,
        folds=MIN_PUBLISHED_FOLDS,
        horizon=7,
        dataset_version=SCHEMA_VERSION,
    )

    for report, family in ((naive, "naive_stub"), (constant, "constant_stub")):
        assert report.model_family == family
        assert report.fold_count == 3
        assert report.horizon == 7
        assert report.dataset_version == SCHEMA_VERSION
        assert {item.key for item in report.by_category} == {"beverages", "snacks"}
        assert {item.key for item in report.by_store} == {"store-01", "store-02"}
        assert [item.key for item in report.by_horizon_step] == ["1", "2", "3", "4", "5", "6", "7"]
        assert report.by_demand_quartile
        assert _row_count(report.by_category) == report.global_metrics.row_count
        assert _row_count(report.by_store) == report.global_metrics.row_count
        assert _row_count(report.by_horizon_step) == report.global_metrics.row_count
        assert _row_count(report.by_demand_quartile) == report.global_metrics.row_count
        document = json.loads(json.dumps(report.to_dict()))
        assert set(document) == {
            "dataset_version",
            "fold_count",
            "horizon",
            "metrics",
            "model_family",
            "runtime_ms",
            "skipped_series",
            "wape_by_horizon",
        }
        assert document["runtime_ms"] >= 0
        assert document["skipped_series"] == []
        assert [item["step"] for item in document["wape_by_horizon"]] == [1, 2, 3, 4, 5, 6, 7]
        assert set(document["metrics"]["global"]) == {
            "bias",
            "mae",
            "pinball_loss_p10",
            "pinball_loss_p90",
            "rmse",
            "row_count",
            "smape",
            "wape",
        }

    assert naive.global_metrics.pinball_loss_p10 is None
    assert naive.global_metrics.pinball_loss_p90 is None
    assert constant.global_metrics.wape == pytest.approx(1.0)
    assert constant.global_metrics.bias == pytest.approx(-1.0)
    assert naive.global_metrics.wape != constant.global_metrics.wape


def test_zero_actual_category_rejects_the_score() -> None:
    frame = _panel(
        n_days=10,
        skus=("beverages-sku", "zeroed-sku"),
        units_for=lambda sku, _offset: 0 if sku == "zeroed-sku" else 5,
        categories={"beverages-sku": "beverages", "zeroed-sku": "zeroed"},
    )
    with pytest.raises(
        ValueError,
        match="WAPE is undefined because the sum of actual demand is zero for category zeroed",
    ):
        Backtester().evaluate(ConstantForecaster(1.0), frame, folds=3, horizon=2)


def test_published_score_requires_three_folds_and_a_positive_horizon() -> None:
    frame = _panel(n_days=12, skus=("sku-1",), units_for=lambda _sku, _offset: 4)
    with pytest.raises(ValueError, match="at least 3 folds"):
        Backtester().evaluate(ConstantForecaster(1.0), frame, folds=2, horizon=2)
    with pytest.raises(ValueError, match="positive number of periods"):
        Backtester().evaluate(ConstantForecaster(1.0), frame, folds=3, horizon=0)
    with pytest.raises(ValueError, match="too short"):
        Backtester().evaluate(ConstantForecaster(1.0), frame, folds=3, horizon=7)

    long_enough = _panel(n_days=11, skus=("sku-1",), units_for=lambda _sku, _offset: 4)
    report = Backtester().evaluate(
        ConstantForecaster(1.0),
        long_enough,
        folds=PREFERRED_BENCHMARK_FOLDS,
        horizon=2,
    )
    assert PREFERRED_BENCHMARK_FOLDS == 5
    assert report.fold_count == 5


def test_public_api_has_no_random_holdout() -> None:
    names = {name for name in dir(evaluation) if not name.startswith("_")}
    blocked = {"random_split", "train_test_split", "holdout", "random_holdout", "kfold"}
    assert blocked.isdisjoint(names)
    assert blocked.isdisjoint(evaluation.__all__)
    assert not any("random" in name.lower() for name in names)


def _panel(
    *,
    n_days: int,
    skus: Sequence[str],
    units_for: Callable[[str, int], int],
    categories: Mapping[str, str] | None = None,
    store_id: str = "store-01",
) -> pa.Table:
    category_for = categories or {sku: "beverages" for sku in skus}
    dates: list[date] = []
    stores: list[str] = []
    sku_ids: list[str] = []
    category_ids: list[str] = []
    units: list[int] = []
    for offset in range(n_days):
        day = START + timedelta(days=offset)
        for sku in skus:
            dates.append(day)
            stores.append(store_id)
            sku_ids.append(sku)
            category_ids.append(category_for[sku])
            units.append(units_for(sku, offset))
    return pa.table(
        {
            "date": pa.array(dates, type=pa.date32()),
            "store_id": pa.array(stores, type=pa.string()),
            "sku_id": pa.array(sku_ids, type=pa.string()),
            "category_id": pa.array(category_ids, type=pa.string()),
            "units_sold": pa.array(units, type=pa.int64()),
        }
    )


def _quartile_units(sku: str, offset: int) -> int:
    if sku == "W" and offset in {8, 9}:
        return 100
    return {"W": 1, "X": 10, "Y": 20, "Z": 40}[sku]


def _dates(frame: pa.Table) -> list[date]:
    return frame.column("date").to_pylist()


def _keys(frame: pa.Table) -> set[tuple[date, str, str]]:
    days = frame.column("date").to_pylist()
    stores = frame.column("store_id").to_pylist()
    skus = frame.column("sku_id").to_pylist()
    return set(zip(days, stores, skus, strict=True))


def _row_count(slices: Sequence[SliceMetrics]) -> int:
    return sum(item.metrics.row_count for item in slices)
