"""Frozen forecast scenarios and their deterministic explanation checks.

The gate is number preservation, known signals, and an uncertainty note.
An optional judge may score readability later. It is not part of this module.
"""

from dataclasses import dataclass

from forecastops_ml.explanations import (
    ExplanationClient,
    ExplanationPackage,
    MockExplanationClient,
    Signal,
    build_explanation_package,
    grade_explanation,
)

MIN_EXPLANATION_SCENARIOS = 50
_KINDS = ("normal", "promotion", "stockout", "wide_interval", "flat")


@dataclass(frozen=True)
class ExplanationScenario:
    """One forecast package the explanation checks grade."""

    scenario_id: str
    kind: str
    package: ExplanationPackage

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"Unknown explanation scenario kind {self.kind}.")
        if self.scenario_id.strip() == "":
            raise ValueError("Explanation scenario id must be non-empty.")


@dataclass(frozen=True)
class ExplanationSummary:
    """Counts from the deterministic explanation checks."""

    scenarios: int
    number_failures: int
    signal_failures: int
    uncertainty_failures: int

    def to_dict(self) -> dict[str, int]:
        """Return the JSON object for this summary."""

        return {
            "number_failures": self.number_failures,
            "scenarios": self.scenarios,
            "signal_failures": self.signal_failures,
            "uncertainty_failures": self.uncertainty_failures,
        }


def explanation_scenarios() -> tuple[ExplanationScenario, ...]:
    """Return at least 50 frozen scenarios covering the required situations.

    The situations are ordinary demand, a promotion, a stock-out context
    signal, a wide prediction interval, and a flat forecast.
    """

    scenarios = (
        *(_normal(index) for index in range(20)),
        *(_promotion(index) for index in range(10)),
        *(_stockout(index) for index in range(8)),
        *(_wide(index) for index in range(6)),
        *(_flat(index) for index in range(6)),
    )
    _require_scenario_set(scenarios)
    return scenarios


def score_explanations(
    scenarios: tuple[ExplanationScenario, ...] | None = None,
    client: ExplanationClient | None = None,
) -> ExplanationSummary:
    """Grade each scenario with the deterministic checks.

    The default client is the local mock. This function does not open a
    remote explanation client.
    """

    selected = explanation_scenarios() if scenarios is None else scenarios
    if len(selected) < MIN_EXPLANATION_SCENARIOS:
        raise ValueError(
            f"The explanation suite needs at least {MIN_EXPLANATION_SCENARIOS} scenarios."
        )
    explainer = MockExplanationClient() if client is None else client
    number_failures = 0
    signal_failures = 0
    uncertainty_failures = 0
    for scenario in selected:
        checks = grade_explanation(scenario.package, explainer.explain(scenario.package))
        if not checks.number_preserved:
            number_failures += 1
        if not checks.signals_known:
            signal_failures += 1
        if not checks.uncertainty_acknowledged:
            uncertainty_failures += 1
    return ExplanationSummary(
        scenarios=len(selected),
        number_failures=number_failures,
        signal_failures=signal_failures,
        uncertainty_failures=uncertainty_failures,
    )


def _normal(index: int) -> ExplanationScenario:
    p50 = 200 + index * 15
    return ExplanationScenario(
        scenario_id=f"normal-{index + 1:02d}",
        kind="normal",
        package=_baseline(
            store="store-01",
            sku=f"sku-{index + 1:03d}",
            p50=p50,
            p10=p50 - 12,
            p90=p50 + 12,
            previous=p50 - 8,
            year_over_year=3 + index,
            horizon_weeks=4,
        ),
    )


def _promotion(index: int) -> ExplanationScenario:
    p50 = 700 + index * 20
    package = build_explanation_package(
        scope={"sku": f"sku-{index + 1:03d}", "store": "store-01"},
        model_family="gradient_boosting",
        horizon_weeks=6,
        p10=float(p50 - 18),
        p50=float(p50),
        p90=float(p50 + 18),
        previous_period_units=float(p50 - 30),
        year_over_year_pct=float(4 + index),
        positive_drivers=("promotion",),
        negative_drivers=("price",),
        driver_values={"promotion": 2.0, "price": -1.0},
    )
    return ExplanationScenario(
        scenario_id=f"promotion-{index + 1:02d}",
        kind="promotion",
        package=package,
    )


def _stockout(index: int) -> ExplanationScenario:
    p50 = 1000 + index * 25
    package = _baseline(
        store="store-02",
        sku=f"sku-{index + 1:03d}",
        p50=p50,
        p10=p50 - 10,
        p90=p50 + 10,
        previous=p50 - 40,
        year_over_year=2 + index,
        horizon_weeks=8,
    )
    stockout = Signal(name="stockout", kind="context")
    return ExplanationScenario(
        scenario_id=f"stockout-{index + 1:02d}",
        kind="stockout",
        package=package.model_copy(update={"signals": [*package.signals, stockout]}),
    )


def _wide(index: int) -> ExplanationScenario:
    p50 = 320 + index * 40
    p10 = 40
    p90 = p10 + (2 * p50) + 10
    return ExplanationScenario(
        scenario_id=f"wide-{index + 1:02d}",
        kind="wide_interval",
        package=_baseline(
            store="store-01",
            sku=f"sku-w{index + 1:02d}",
            p50=p50,
            p10=p10,
            p90=p90,
            previous=p50 - 20,
            year_over_year=6,
            horizon_weeks=2,
        ),
    )


def _flat(index: int) -> ExplanationScenario:
    p50 = 1500 + index * 5
    return ExplanationScenario(
        scenario_id=f"flat-{index + 1:02d}",
        kind="flat",
        package=_baseline(
            store="store-02",
            sku=f"sku-f{index + 1:02d}",
            p50=p50,
            p10=p50,
            p90=p50,
            previous=p50,
            year_over_year=1,
            horizon_weeks=1,
        ),
    )


def _baseline(
    *,
    store: str,
    sku: str,
    p50: int,
    p10: int,
    p90: int,
    previous: int,
    year_over_year: int,
    horizon_weeks: int,
) -> ExplanationPackage:
    return build_explanation_package(
        scope={"sku": sku, "store": store},
        model_family="seasonal_naive",
        horizon_weeks=horizon_weeks,
        p10=float(p10),
        p50=float(p50),
        p90=float(p90),
        previous_period_units=float(previous),
        year_over_year_pct=float(year_over_year),
    )


def _require_scenario_set(scenarios: tuple[ExplanationScenario, ...]) -> None:
    if len(scenarios) < MIN_EXPLANATION_SCENARIOS:
        raise ValueError(
            f"The explanation suite needs at least {MIN_EXPLANATION_SCENARIOS} scenarios."
        )
    ids = [item.scenario_id for item in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("Explanation scenario ids must be unique.")
    for scenario in scenarios:
        _require_kind(scenario)


def _require_kind(scenario: ExplanationScenario) -> None:
    names = {signal.name for signal in scenario.package.signals}
    wide = _wide_interval(scenario.package)
    flat = _flat_forecast(scenario.package)
    if scenario.kind == "promotion" and "promotion" not in names:
        raise ValueError(f"{scenario.scenario_id} needs a promotion signal.")
    if scenario.kind == "stockout" and not _stockout_context(scenario.package):
        raise ValueError(f"{scenario.scenario_id} needs a stockout context signal.")
    if scenario.kind == "wide_interval" and not wide:
        raise ValueError(f"{scenario.scenario_id} needs a wide prediction interval.")
    if scenario.kind == "flat" and not flat:
        raise ValueError(f"{scenario.scenario_id} needs a flat forecast.")
    if scenario.kind == "normal" and (wide or flat or "promotion" in names or "stockout" in names):
        raise ValueError(f"{scenario.scenario_id} must stay ordinary demand.")


def _wide_interval(package: ExplanationPackage) -> bool:
    forecast = package.forecast
    if forecast.p10 is None or forecast.p90 is None:
        return False
    return (forecast.p90 - forecast.p10) >= 2 * abs(forecast.p50)


def _flat_forecast(package: ExplanationPackage) -> bool:
    forecast = package.forecast
    return forecast.p10 == forecast.p50 == forecast.p90


def _stockout_context(package: ExplanationPackage) -> bool:
    return any(signal.name == "stockout" and signal.kind == "context" for signal in package.signals)
