"""PSI, KS, frequency deltas, and the three monitoring statuses."""

import math
from datetime import date, timedelta

import pyarrow as pa
import pytest

from forecastops_ml.monitoring import (
    DEFAULT_PSI_RETRAIN,
    PSI_EPSILON,
    DriftThresholds,
    absolute_frequency_deltas,
    build_monitoring_report,
    dataset_age_days,
    frequency_delta,
    kolmogorov_smirnov,
    monitoring_status,
    population_stability_index,
    split_recent_window,
)

START = date(2026, 1, 1)
QUIET_PRICES = [1.0] * 14 + [5.0] * 14
WARNING_PRICES = [1.0] * 8 + [5.0] * 20
RETRAIN_PRICES = [1.0] * 7 + [5.0] * 21


def test_psi_matches_a_hand_computed_two_bin_shift() -> None:
    baseline = [1.0, 2.0, 3.0, 11.0, 12.0, 13.0]
    recent = [1.0, 2.0, 11.0, 12.0, 13.0, 14.0]
    low = 1.0 / 3.0
    high = 2.0 / 3.0
    expected = (low - 0.5) * math.log(low / 0.5) + (high - 0.5) * math.log(high / 0.5)

    assert population_stability_index(baseline, recent, edges=[0.0, 10.0, 20.0]) == pytest.approx(
        expected
    )


def test_identical_samples_have_zero_psi_and_zero_ks() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert population_stability_index(values, values) == pytest.approx(0.0)
    assert kolmogorov_smirnov(values, values) == pytest.approx(0.0)


def test_ks_matches_a_hand_computed_shift() -> None:
    assert kolmogorov_smirnov([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]) == pytest.approx(0.25)
    assert kolmogorov_smirnov([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]) == pytest.approx(1.0)


def test_empty_bins_use_the_share_floor() -> None:
    baseline = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    recent = [1.0] * 10
    edges = [0.0, 5.0, 10.0, 15.0]
    baseline_shares = [0.4, 0.5, 0.1]
    recent_shares = _floored([1.0, 0.0, 0.0])

    assert population_stability_index(baseline, recent, edges=edges) == pytest.approx(
        _psi(baseline_shares, recent_shares)
    )


def test_frequency_delta_matches_a_hand_computed_shift() -> None:
    deltas = absolute_frequency_deltas(["a", "a", "b", "c"], ["a", "b", "b", "b"])

    assert deltas == {"a": pytest.approx(0.25), "b": pytest.approx(0.5), "c": pytest.approx(0.25)}
    assert frequency_delta([False, False, True, True], [True, True, True, False]) == pytest.approx(
        0.25
    )


def test_statistic_helpers_reject_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        population_stability_index([], [1.0])
    with pytest.raises(ValueError, match="at least one value"):
        kolmogorov_smirnov([1.0], [])
    with pytest.raises(ValueError, match="at least one value"):
        absolute_frequency_deltas([], ["a"])


def test_quiet_window_is_healthy() -> None:
    report = _report(QUIET_PRICES, approved_wape=0.10, recent_wape=0.12)

    assert report.status == "HEALTHY"
    assert report.reasons == ()
    assert report.dataset_age_days == dataset_age_days(report.date_max, report.as_of)
    assert report.dataset_age_days == 0
    assert report.wape_degradation == pytest.approx(0.2)
    assert _signal(report, "price").psi == pytest.approx(0.0)
    assert _signal(report, "units_sold").ks == pytest.approx(0.0)


def test_moderate_price_shift_is_a_warning() -> None:
    report = _report(WARNING_PRICES, approved_wape=0.10, recent_wape=0.11)

    assert report.status == "WARNING"
    assert report.reasons == ("PSI for price is above 0.1 and at or below 0.2.",)
    assert _signal(report, "price").psi == pytest.approx(_two_bin_psi(8))
    assert _signal(report, "price").ks == pytest.approx(6 / 28)
    assert report.wape_degradation == pytest.approx(0.1)


def test_wape_jump_recommends_retraining() -> None:
    report = _report(QUIET_PRICES, approved_wape=0.10, recent_wape=0.13)

    assert report.status == "RETRAIN_RECOMMENDED"
    assert report.reasons == ("Recent WAPE is more than 20% worse than the approved model's WAPE.",)
    assert report.wape_degradation == pytest.approx(0.3)


def test_price_shift_above_the_psi_threshold_is_at_least_a_warning() -> None:
    report = _report(RETRAIN_PRICES, approved_wape=0.10, recent_wape=0.12)

    assert report.status == "RETRAIN_RECOMMENDED"
    assert report.wape_degradation == pytest.approx(0.2)
    assert _signal(report, "price").psi == pytest.approx(_two_bin_psi(7))
    assert _signal(report, "price").psi > DEFAULT_PSI_RETRAIN
    assert report.reasons == ("PSI for price is above 0.2.",)


def test_dataset_age_uses_date_max_against_as_of() -> None:
    date_max = START + timedelta(days=55)
    fresh = _report(QUIET_PRICES, approved_wape=0.10, recent_wape=0.10, as_of=date_max)
    boundary = _report(
        QUIET_PRICES,
        approved_wape=0.10,
        recent_wape=0.10,
        as_of=date_max + timedelta(days=30),
    )
    stale = _report(
        QUIET_PRICES,
        approved_wape=0.10,
        recent_wape=0.10,
        as_of=date_max + timedelta(days=31),
    )

    assert fresh.dataset_age_days == 0
    assert fresh.status == "HEALTHY"
    assert boundary.dataset_age_days == (boundary.as_of - boundary.date_max).days == 30
    assert boundary.status == "HEALTHY"
    assert stale.dataset_age_days == 31
    assert stale.as_of == stale.date_max + timedelta(days=31)
    assert stale.status == "RETRAIN_RECOMMENDED"
    assert stale.reasons == ("Dataset age is greater than 30 days.",)


def test_status_boundaries_are_strict() -> None:
    assert _status(psi=0.1, degradation=0.2, age=30) == "HEALTHY"
    assert _status(psi=0.2, degradation=0.0, age=0) == "WARNING"
    assert _status(psi=0.2 + 1e-9, degradation=0.0, age=0) == "RETRAIN_RECOMMENDED"
    assert _status(psi=0.0, degradation=0.2 + 1e-9, age=0) == "RETRAIN_RECOMMENDED"
    assert _status(psi=0.0, degradation=0.0, age=31) == "RETRAIN_RECOMMENDED"


def test_store_coverage_movement_is_a_warning() -> None:
    baseline, recent = _windows(QUIET_PRICES)
    moved = _table(
        [START + timedelta(days=28 + offset) for offset in range(28)],
        QUIET_PRICES,
        stores=["s1"] * 14 + ["s2"] * 14,
    )
    report = build_monitoring_report(
        baseline,
        moved,
        as_of=START + timedelta(days=55),
        date_max=START + timedelta(days=55),
        approved_wape=0.10,
        recent_wape=0.10,
    )

    assert report.status == "WARNING"
    assert report.reasons == ("Store coverage moved by more than 0.05.",)
    store = next(item for item in report.coverage if item.feature == "store")
    assert store.absolute_deltas["s1"] == pytest.approx(0.5)
    assert store.absolute_deltas["s2"] == pytest.approx(0.5)


def test_recent_window_is_the_last_inclusive_span() -> None:
    frame = _table(
        [START + timedelta(days=offset) for offset in range(56)],
        QUIET_PRICES + QUIET_PRICES,
    )

    baseline, recent = split_recent_window(frame, recent_window_days=28)

    assert baseline.num_rows == 28
    assert recent.num_rows == 28
    assert max(baseline.column("date").to_pylist()) == START + timedelta(days=27)
    assert min(recent.column("date").to_pylist()) == START + timedelta(days=28)


def test_zero_approved_wape_with_recent_error_recommends_retraining() -> None:
    status, reasons = monitoring_status(
        psi={"price": 0.0},
        frequency_deltas={},
        approved_wape=0.0,
        recent_wape=0.01,
        dataset_age_days=0,
    )

    assert status == "RETRAIN_RECOMMENDED"
    assert reasons == ("Recent WAPE is worse than an approved WAPE of zero.",)


def _status(*, psi: float, degradation: float, age: int) -> str:
    approved = 0.10
    status, _reasons = monitoring_status(
        psi={"price": psi, "units_sold": 0.0},
        frequency_deltas={},
        approved_wape=approved,
        recent_wape=approved * (1.0 + degradation),
        dataset_age_days=age,
        thresholds=DriftThresholds(),
    )
    return status


def _report(
    recent_prices: list[float],
    *,
    approved_wape: float,
    recent_wape: float,
    as_of: date | None = None,
):
    baseline, recent = _windows(recent_prices)
    date_max = START + timedelta(days=55)
    return build_monitoring_report(
        baseline,
        recent,
        as_of=date_max if as_of is None else as_of,
        date_max=date_max,
        approved_wape=approved_wape,
        recent_wape=recent_wape,
    )


def _windows(recent_prices: list[float]) -> tuple[pa.Table, pa.Table]:
    baseline_days = [START + timedelta(days=offset) for offset in range(28)]
    recent_days = [START + timedelta(days=28 + offset) for offset in range(28)]
    return _table(baseline_days, QUIET_PRICES), _table(recent_days, recent_prices)


def _table(days: list[date], prices: list[float], *, stores: list[str] | None = None) -> pa.Table:
    count = len(days)
    return pa.table(
        {
            "date": pa.array(days, type=pa.date32()),
            "store_id": stores or ["s1"] * count,
            "sku_id": ["k1"] * count,
            "category_id": ["c1"] * count,
            "units_sold": pa.array([4] * count, type=pa.int64()),
            "price": pa.array(prices, type=pa.float64()),
            "promotion": pa.array([False] * count, type=pa.bool_()),
            "stockout": pa.array([False] * count, type=pa.bool_()),
        }
    )


def _signal(report, feature: str):
    return next(item for item in report.numeric if item.feature == feature)


def _two_bin_psi(recent_low: int, total: int = 28) -> float:
    low = recent_low / total
    high = 1.0 - low
    return (low - 0.5) * math.log(low / 0.5) + (high - 0.5) * math.log(high / 0.5)


def _floored(shares: list[float]) -> list[float]:
    raised = [max(share, PSI_EPSILON) for share in shares]
    total = sum(raised)
    return [share / total for share in raised]


def _psi(baseline: list[float], recent: list[float]) -> float:
    return sum(
        (recent_share - baseline_share) * math.log(recent_share / baseline_share)
        for baseline_share, recent_share in zip(baseline, recent, strict=True)
    )
