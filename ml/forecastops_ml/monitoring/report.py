"""Monitoring report for a recent window against a training baseline.

The report records PSI and a KS statistic for price and units sold, absolute
frequency deltas for promotion, zero demand, stock-outs, stores, and
categories, and one status. Dataset age is ``as_of - date_max`` in days.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from forecastops_ml.monitoring.statistics import (
    absolute_frequency_deltas,
    frequency_delta,
    kolmogorov_smirnov,
    population_stability_index,
)
from forecastops_ml.monitoring.thresholds import DriftThresholds

StatusName = Literal["HEALTHY", "WARNING", "RETRAIN_RECOMMENDED"]

HEALTHY: StatusName = "HEALTHY"
WARNING: StatusName = "WARNING"
RETRAIN_RECOMMENDED: StatusName = "RETRAIN_RECOMMENDED"
STATUSES = (HEALTHY, WARNING, RETRAIN_RECOMMENDED)

BoolArray = NDArray[np.bool_]

_NUMERIC = ("price", "units_sold")
_FREQUENCIES = ("promotion", "zero_demand", "stockout", "store", "category")
_FREQUENCY_LABELS = {
    "promotion": "Promotion frequency",
    "zero_demand": "Zero-demand frequency",
    "stockout": "Stock-out frequency",
    "store": "Store coverage",
    "category": "Category coverage",
}


@dataclass(frozen=True)
class NumericDrift:
    """PSI and KS for one numeric column."""

    feature: str
    psi: float
    ks: float

    def to_dict(self) -> dict[str, object]:
        """Return the JSON object for this column."""

        return {"feature": self.feature, "ks": self.ks, "psi": self.psi}


@dataclass(frozen=True)
class FrequencyDrift:
    """Absolute change in one rate. Null fields mean the column was absent."""

    feature: str
    baseline: float | None
    recent: float | None
    absolute_delta: float | None

    def to_dict(self) -> dict[str, object]:
        """Return the JSON object for this rate."""

        return {
            "absolute_delta": self.absolute_delta,
            "baseline": self.baseline,
            "feature": self.feature,
            "recent": self.recent,
        }


@dataclass(frozen=True)
class CoverageDrift:
    """Absolute share changes for one categorical column."""

    feature: str
    absolute_deltas: dict[str, float]
    max_absolute_delta: float

    def to_dict(self) -> dict[str, object]:
        """Return the JSON object for this coverage comparison."""

        return {
            "absolute_deltas": self.absolute_deltas,
            "feature": self.feature,
            "max_absolute_delta": self.max_absolute_delta,
        }


@dataclass(frozen=True)
class MonitoringReport:
    """One comparison of a recent window with the training baseline."""

    status: StatusName
    as_of: date
    date_max: date
    baseline_start: date
    baseline_end: date
    recent_start: date
    recent_end: date
    dataset_age_days: int
    approved_wape: float | None
    recent_wape: float | None
    wape_degradation: float | None
    numeric: tuple[NumericDrift, ...]
    frequencies: tuple[FrequencyDrift, ...]
    coverage: tuple[CoverageDrift, ...]
    reasons: tuple[str, ...]
    thresholds: DriftThresholds

    def to_dict(self) -> dict[str, object]:
        """Return the JSON object for this report."""

        return {
            "approved_wape": self.approved_wape,
            "as_of": self.as_of.isoformat(),
            "baseline_end": self.baseline_end.isoformat(),
            "baseline_start": self.baseline_start.isoformat(),
            "coverage": [item.to_dict() for item in self.coverage],
            "dataset_age_days": self.dataset_age_days,
            "date_max": self.date_max.isoformat(),
            "frequencies": [item.to_dict() for item in self.frequencies],
            "numeric": [item.to_dict() for item in self.numeric],
            "reasons": list(self.reasons),
            "recent_end": self.recent_end.isoformat(),
            "recent_start": self.recent_start.isoformat(),
            "recent_wape": self.recent_wape,
            "status": self.status,
            "thresholds": self.thresholds.to_dict(),
            "wape_degradation": self.wape_degradation,
        }


def dataset_age_days(date_max: date, as_of: date) -> int:
    """Return how many days ``date_max`` sits before ``as_of``."""

    return (as_of - date_max).days


def split_recent_window(
    frame: pa.Table,
    *,
    recent_window_days: int,
) -> tuple[pa.Table, pa.Table]:
    """Split ``frame`` into the baseline and the inclusive recent window.

    The recent window is the last ``recent_window_days`` calendar days ending
    on the latest observation date. Earlier rows are the training baseline.
    """

    if recent_window_days < 1:
        raise ValueError("recent_window_days must be at least 1.")
    days = _dates(frame, "observation table")
    latest = max(days)
    recent_start = latest - timedelta(days=recent_window_days - 1)
    baseline_index = [index for index, day in enumerate(days) if day < recent_start]
    recent_index = [index for index, day in enumerate(days) if day >= recent_start]
    if not baseline_index or not recent_index:
        raise ValueError("The observation table needs rows before and inside the recent window.")
    return frame.take(baseline_index), frame.take(recent_index)


def build_monitoring_report(
    baseline: pa.Table,
    recent: pa.Table,
    *,
    as_of: date,
    date_max: date,
    approved_wape: float | None,
    recent_wape: float | None,
    thresholds: DriftThresholds | None = None,
) -> MonitoringReport:
    """Compare ``recent`` with ``baseline`` and assign a monitoring status."""

    limits = thresholds or DriftThresholds()
    numeric = tuple(_numeric(baseline, recent, feature, limits) for feature in _NUMERIC)
    frequencies = (
        _flag_frequency(baseline, recent, "promotion"),
        _zero_demand(baseline, recent),
        _flag_frequency(baseline, recent, "stockout"),
    )
    coverage = (
        _coverage(baseline, recent, "store", "store_id"),
        _coverage(baseline, recent, "category", "category_id"),
    )
    degradation = _wape_signal(approved_wape, recent_wape, limits.wape_degradation_limit)[0]
    age = dataset_age_days(date_max, as_of)
    frequency_deltas = {
        item.feature: item.absolute_delta for item in frequencies if item.absolute_delta is not None
    }
    for item in coverage:
        frequency_deltas[item.feature] = item.max_absolute_delta
    status, reasons = monitoring_status(
        psi={item.feature: item.psi for item in numeric},
        frequency_deltas=frequency_deltas,
        approved_wape=approved_wape,
        recent_wape=recent_wape,
        dataset_age_days=age,
        thresholds=limits,
    )
    return MonitoringReport(
        status=status,
        as_of=as_of,
        date_max=date_max,
        baseline_start=min(_dates(baseline, "baseline window")),
        baseline_end=max(_dates(baseline, "baseline window")),
        recent_start=min(_dates(recent, "recent window")),
        recent_end=max(_dates(recent, "recent window")),
        dataset_age_days=age,
        approved_wape=approved_wape,
        recent_wape=recent_wape,
        wape_degradation=degradation,
        numeric=numeric,
        frequencies=frequencies,
        coverage=coverage,
        reasons=reasons,
        thresholds=limits,
    )


def monitoring_status(
    *,
    psi: Mapping[str, float],
    frequency_deltas: Mapping[str, float],
    approved_wape: float | None,
    recent_wape: float | None,
    dataset_age_days: int,
    thresholds: DriftThresholds | None = None,
) -> tuple[StatusName, tuple[str, ...]]:
    """Return the status and the English reasons that produced it.

    Retraining is recommended when recent WAPE is more than the configured
    limit worse than the approved WAPE, when any PSI is above ``psi_retrain``,
    or when dataset age is greater than ``max_dataset_age_days``. A PSI above
    ``psi_warning`` and at or below ``psi_retrain``, or a frequency move above
    ``frequency_warning_delta``, is a warning when no retrain rule holds.
    """

    limits = thresholds or DriftThresholds()
    _finite_metrics(psi, "PSI")
    _finite_metrics(frequency_deltas, "frequency delta")
    degradation, wape_exceeds = _wape_signal(
        approved_wape,
        recent_wape,
        limits.wape_degradation_limit,
    )
    retrain: list[str] = []
    warning: list[str] = []
    if wape_exceeds and degradation is None:
        retrain.append("Recent WAPE is worse than an approved WAPE of zero.")
    elif wape_exceeds:
        retrain.append(
            "Recent WAPE is more than "
            f"{limits.wape_degradation_limit:.0%} worse than the approved model's WAPE."
        )
    for feature in _ordered(psi, _NUMERIC):
        value = psi[feature]
        if value > limits.psi_retrain:
            retrain.append(f"PSI for {feature} is above {_format_number(limits.psi_retrain)}.")
        elif value > limits.psi_warning:
            warning.append(
                f"PSI for {feature} is above {_format_number(limits.psi_warning)} "
                f"and at or below {_format_number(limits.psi_retrain)}."
            )
    if dataset_age_days > limits.max_dataset_age_days:
        retrain.append(f"Dataset age is greater than {limits.max_dataset_age_days} days.")
    boundary = _format_number(limits.frequency_warning_delta)
    for feature in _ordered(frequency_deltas, _FREQUENCIES):
        if frequency_deltas[feature] > limits.frequency_warning_delta:
            label = _FREQUENCY_LABELS.get(feature, feature)
            warning.append(f"{label} moved by more than {boundary}.")
    if retrain:
        return RETRAIN_RECOMMENDED, tuple(retrain + warning)
    if warning:
        return WARNING, tuple(warning)
    return HEALTHY, ()


def _numeric(
    baseline: pa.Table,
    recent: pa.Table,
    feature: str,
    thresholds: DriftThresholds,
) -> NumericDrift:
    baseline_values = _numbers(baseline, feature)
    recent_values = _numbers(recent, feature)
    return NumericDrift(
        feature=feature,
        psi=population_stability_index(
            baseline_values,
            recent_values,
            bin_count=thresholds.psi_bin_count,
        ),
        ks=kolmogorov_smirnov(baseline_values, recent_values),
    )


def _flag_frequency(baseline: pa.Table, recent: pa.Table, feature: str) -> FrequencyDrift:
    baseline_flags = _optional_flags(baseline, feature)
    recent_flags = _optional_flags(recent, feature)
    if baseline_flags is None or recent_flags is None:
        return FrequencyDrift(feature, None, None, None)
    baseline_rate = float(np.mean(baseline_flags))
    recent_rate = float(np.mean(recent_flags))
    return FrequencyDrift(
        feature,
        baseline_rate,
        recent_rate,
        frequency_delta(_python_flags(baseline_flags), _python_flags(recent_flags)),
    )


def _zero_demand(baseline: pa.Table, recent: pa.Table) -> FrequencyDrift:
    baseline_units = _numbers(baseline, "units_sold")
    recent_units = _numbers(recent, "units_sold")
    baseline_flags = [bool(value == 0.0) for value in baseline_units]
    recent_flags = [bool(value == 0.0) for value in recent_units]
    baseline_rate = float(sum(baseline_flags) / len(baseline_flags))
    recent_rate = float(sum(recent_flags) / len(recent_flags))
    return FrequencyDrift(
        "zero_demand",
        baseline_rate,
        recent_rate,
        frequency_delta(baseline_flags, recent_flags),
    )


def _coverage(baseline: pa.Table, recent: pa.Table, feature: str, column: str) -> CoverageDrift:
    deltas = absolute_frequency_deltas(_labels(baseline, column), _labels(recent, column))
    maximum = max(deltas.values(), default=0.0)
    return CoverageDrift(feature, deltas, maximum)


def _wape_signal(
    approved: float | None,
    recent: float | None,
    limit: float,
) -> tuple[float | None, bool]:
    if approved is None or recent is None:
        return None, False
    if type(approved) is not float and type(approved) is not int:
        raise ValueError("WAPE must be zero or greater.")
    if type(recent) is not float and type(recent) is not int:
        raise ValueError("WAPE must be zero or greater.")
    if float(approved) < 0.0 or float(recent) < 0.0:
        raise ValueError("WAPE must be zero or greater.")
    if float(approved) == 0.0:
        return None, float(recent) > 0.0
    degradation = (float(recent) - float(approved)) / float(approved)
    return degradation, degradation > limit


def _dates(frame: pa.Table, label: str) -> list[date]:
    if frame.num_rows == 0:
        raise ValueError(f"The {label} has no rows.")
    if "date" not in frame.column_names:
        raise ValueError(f"The {label} is missing date.")
    values = frame.column("date").to_pylist()
    if any(not isinstance(item, date) for item in values):
        raise ValueError(f"The {label} has a date that is not a calendar day.")
    return [item for item in values if isinstance(item, date)]


def _numbers(frame: pa.Table, name: str) -> list[float]:
    if name not in frame.column_names:
        raise ValueError(f"The observation table is missing {name}.")
    values = frame.column(name).to_pylist()
    if any(isinstance(item, bool) or not isinstance(item, int | float) for item in values):
        raise ValueError(f"Column {name} must be numeric.")
    return [float(item) for item in values if isinstance(item, int | float)]


def _labels(frame: pa.Table, name: str) -> list[str]:
    if name not in frame.column_names:
        raise ValueError(f"The observation table is missing {name}.")
    values = frame.column(name).to_pylist()
    if any(not isinstance(item, str) or item == "" for item in values):
        raise ValueError(f"Column {name} must be non-empty strings.")
    return [item for item in values if isinstance(item, str)]


def _python_flags(values: BoolArray) -> list[bool]:
    return [bool(item) for item in values.tolist()]


def _optional_flags(frame: pa.Table, name: str) -> BoolArray | None:
    if name not in frame.column_names:
        return None
    values = frame.column(name).to_pylist()
    if any(type(item) is not bool for item in values):
        raise ValueError(f"Column {name} must be boolean.")
    return np.asarray(values, dtype=np.bool_)


def _ordered(values: Mapping[str, float], preferred: Sequence[str]) -> list[str]:
    first = [name for name in preferred if name in values]
    rest = sorted(set(values) - set(first))
    return first + rest


def _finite_metrics(values: Mapping[str, float], label: str) -> None:
    for feature, value in values.items():
        if type(value) is not float and type(value) is not int:
            raise ValueError(f"{label} for {feature} must be a finite number.")
        if not np.isfinite(float(value)):
            raise ValueError(f"{label} for {feature} must be a finite number.")


def _format_number(value: float) -> str:
    return f"{value:g}"
