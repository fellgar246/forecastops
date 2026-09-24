"""Local catalog of baseline forecasters and their comparison table.

The default catalog registers ``naive``, weekly ``seasonal_naive``,
``holt_winters``, and ``gradient_boosting``. Yearly seasonal naive uses the
same family on weekly history. It stays out of the default comparison when
the frame covers fewer than 52 distinct weeks. Holt-Winters scores
category-week aggregates. Gradient boosting scores store-SKU weeks and is
omitted on a daily frame. A model whose every series is skipped is left out
of the table.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa

from forecastops_ml.baselines.holt_winters import HoltWintersForecaster
from forecastops_ml.baselines.naive import (
    YEARLY_SEASON_LENGTH,
    NaiveForecaster,
    weekly_seasonal_naive,
)
from forecastops_ml.evaluation.backtest import Backtester, Forecaster, UnscoredModelError
from forecastops_ml.training.gradient_boosting import GradientBoostingForecaster

COMPARISON_FILENAME = "baseline_comparison.json"


@dataclass(frozen=True)
class ComparisonRow:
    """One family's score on the shared folds."""

    family: str
    wape: float
    smape: float
    bias: float
    runtime_ms: int

    def to_dict(self) -> dict[str, object]:
        """Return the JSON object for this row."""

        return {
            "bias": self.bias,
            "family": self.family,
            "runtime_ms": self.runtime_ms,
            "smape": self.smape,
            "wape": self.wape,
        }


@dataclass(frozen=True)
class ComparisonTable:
    """Families scored by one backtest each, on the same folds and horizon."""

    models: tuple[ComparisonRow, ...]
    fold_count: int
    horizon: int
    dataset_version: str | None

    def to_dict(self) -> dict[str, object]:
        """Return the JSON document for this table."""

        return {
            "dataset_version": self.dataset_version,
            "fold_count": self.fold_count,
            "horizon": self.horizon,
            "models": [row.to_dict() for row in self.models],
        }


class ModelCatalog:
    """Forecasters registered by ``model_family``."""

    def __init__(self, models: Mapping[str, Forecaster]) -> None:
        if not models:
            raise ValueError("A model catalog needs at least one model.")
        stored: dict[str, Forecaster] = {}
        for family, model in models.items():
            if not isinstance(family, str) or family.strip() == "":
                raise ValueError("Catalog keys must be non-empty model family names.")
            actual = getattr(model, "model_family", None)
            if actual != family:
                raise ValueError(f"Catalog key {family} does not match model_family {actual}.")
            if family in stored:
                raise ValueError(f"Catalog already registers {family}.")
            stored[family] = model
        self._models = stored
        self._order = tuple(stored)

    def families(self) -> tuple[str, ...]:
        """Return registered families in insertion order."""

        return self._order

    def get(self, family: str) -> Forecaster:
        """Return the forecaster registered under ``family``."""

        try:
            return self._models[family]
        except KeyError as exc:
            raise ValueError(f"Catalog has no model family {family}.") from exc

    def compare(
        self,
        frame: pa.Table,
        *,
        folds: int,
        horizon: int,
        dataset_version: str | None = None,
    ) -> ComparisonTable:
        """Score every registered model once, on the same folds.

        Each call uses :meth:`Backtester.evaluate`. The table keeps family,
        WAPE, sMAPE, bias, and fit runtime. A model that skips every series
        is omitted so one unfit family does not drop the other rows.
        """

        backtester = Backtester()
        rows: list[ComparisonRow] = []
        skipped_run: UnscoredModelError | None = None
        for family in self._order:
            try:
                report = backtester.evaluate(
                    self._models[family],
                    frame,
                    folds,
                    horizon,
                    dataset_version=dataset_version,
                )
            except UnscoredModelError as exc:
                skipped_run = exc
                continue
            metrics = report.global_metrics
            rows.append(
                ComparisonRow(
                    family=report.model_family,
                    wape=metrics.wape,
                    smape=metrics.smape,
                    bias=metrics.bias,
                    runtime_ms=report.runtime_ms,
                )
            )
        if not rows:
            if skipped_run is not None:
                raise skipped_run
            raise ValueError("Comparison scored no models.")
        return ComparisonTable(
            models=tuple(rows),
            fold_count=folds,
            horizon=horizon,
            dataset_version=dataset_version,
        )


def local_catalog() -> ModelCatalog:
    """Return naive, weekly seasonal naive, Holt-Winters, and gradient boosting.

    Yearly seasonal naive is a separate ``seasonal_naive`` configuration for
    weekly history. Register it yourself when the frame has at least 52
    distinct weeks. The default catalog leaves it out. Holt-Winters stays in
    the catalog and scores a category-week frame. On a daily frame it skips
    every series until the caller aggregates to one row per category and week.
    Gradient boosting scores a store-SKU week frame. On a daily frame it skips
    every series unless the caller builds it with daily grain and profile full.
    """

    return ModelCatalog(
        {
            "naive": NaiveForecaster(),
            "seasonal_naive": weekly_seasonal_naive(),
            "holt_winters": HoltWintersForecaster(),
            "gradient_boosting": GradientBoostingForecaster(),
        }
    )


def yearly_season_available(dates: Sequence[date]) -> bool:
    """Return whether ``dates`` cover at least 52 distinct weeks.

    The default comparison omits yearly seasonal naive when this is false.
    """

    weeks = {day - timedelta(days=day.weekday()) for day in dates}
    return len(weeks) >= YEARLY_SEASON_LENGTH


def write_comparison(table: ComparisonTable, directory: Path) -> Path:
    """Write ``baseline_comparison.json`` in ``directory`` and return that path."""

    if directory.exists() and not directory.is_dir():
        raise ValueError(f"Comparison path {directory} is a file. Pass a directory.")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / COMPARISON_FILENAME
    payload = json.dumps(table.to_dict(), indent=2, sort_keys=True) + "\n"
    destination.write_text(payload, encoding="utf-8")
    return destination
