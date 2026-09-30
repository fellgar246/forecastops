"""Build the explanation package from forecast totals, metrics, and signals."""

import math
from collections.abc import Sequence

from forecastops_ml.explanations.models import (
    ExplanationPackage,
    ForecastWindow,
    HistoryWindow,
    Signal,
)

# Fields a global probabilistic model may describe. They are context, not causes.
GLOBAL_CONTEXT = (
    "historical_context",
    "recent_trend",
    "known_future_covariates",
    "seasonal_pattern",
    "uncertainty",
    "baseline_comparison",
)

BASELINE_CONTEXT = (
    "historical_context",
    "recent_trend",
    "seasonal_pattern",
    "uncertainty",
    "baseline_comparison",
)


def build_explanation_package(
    *,
    scope: dict[str, str],
    model_family: str,
    horizon_weeks: float,
    p50: float,
    p10: float | None = None,
    p90: float | None = None,
    previous_period_units: float | None = None,
    year_over_year_pct: float | None = None,
    wape: float | None = None,
    bias: float | None = None,
    positive_drivers: Sequence[str] = (),
    negative_drivers: Sequence[str] = (),
    driver_values: dict[str, float] | None = None,
) -> ExplanationPackage:
    """Assemble the package an explanation adapter is allowed to see.

    Tree models may list ``positive_drivers`` and ``negative_drivers``. Those
    names are associations from the trees. The global probabilistic model
    receives historical context, recent trend, known future covariates,
    seasonal pattern, uncertainty, and baseline comparison. Those names are
    context, not causes.
    """

    if not scope:
        raise ValueError("Explanation scope must name the series or slice.")
    if any(not key or not value for key, value in scope.items()):
        raise ValueError("Explanation scope keys and values must be non-empty.")
    _finite("horizon_weeks", horizon_weeks)
    if horizon_weeks <= 0:
        raise ValueError("horizon_weeks must be positive.")
    _finite("p50", p50)
    _optional("p10", p10)
    _optional("p90", p90)
    _optional("previous_period_units", previous_period_units)
    _optional("year_over_year_pct", year_over_year_pct)
    _optional("wape", wape)
    _optional("bias", bias)
    metrics: dict[str, float] = {}
    if wape is not None:
        metrics["wape"] = wape
    if bias is not None:
        metrics["bias"] = bias
    positive = _names(positive_drivers, "positive_drivers")
    negative = _names(negative_drivers, "negative_drivers")
    values = driver_values or {}
    signals = _signals(model_family, positive, negative, values)
    return ExplanationPackage(
        scope=dict(scope),
        forecast=ForecastWindow(
            horizon_weeks=horizon_weeks,
            p10=p10,
            p50=p50,
            p90=p90,
        ),
        history=HistoryWindow(
            previous_period_units=previous_period_units,
            year_over_year_pct=year_over_year_pct,
        ),
        signals=signals,
        model_metrics=metrics,
        positive_drivers=positive,
        negative_drivers=negative,
    )


def _signals(
    model_family: str,
    positive: list[str],
    negative: list[str],
    values: dict[str, float],
) -> list[Signal]:
    if model_family == "gradient_boosting":
        return [
            Signal(
                name=name,
                direction="positive",
                value=values.get(name),
                kind="driver",
            )
            for name in positive
        ] + [
            Signal(
                name=name,
                direction="negative",
                value=values.get(name),
                kind="driver",
            )
            for name in negative
        ]
    names = GLOBAL_CONTEXT if model_family == "deepar" else BASELINE_CONTEXT
    return [Signal(name=name, kind="context") for name in names]


def _names(values: Sequence[str], label: str) -> list[str]:
    names: list[str] = []
    for value in values:
        if not isinstance(value, str) or value == "":
            raise ValueError(f"{label} must be non-empty strings.")
        names.append(value)
    return names


def _optional(label: str, value: float | None) -> None:
    if value is not None:
        _finite(label, value)


def _finite(label: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a finite number.")
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number.")
