"""Quality gate that decides whether a candidate may wait for a person.

The gate compares a candidate evaluation with a reference evaluation. It can
set ``PENDING_APPROVAL`` or ``REJECTED``. It never sets ``APPROVED``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from forecastops_ml.evaluation.backtest import EvaluationReport
from forecastops_ml.promotion.models import ModelStatus
from forecastops_ml.promotion.thresholds import PromotionThresholds

QUALITY_GATE = "quality_gate"
HUMAN_REJECTION = "human"


@dataclass(frozen=True)
class GateChecks:
    """Boolean result of each promotion clause.

    ``coverage_within_limit`` is null when the candidate has no ``p90`` and
    that clause is skipped. The other flags are always true or false.
    """

    wape_improved: bool
    bias_within_limit: bool
    coverage_within_limit: bool | None
    no_critical_segment_regression: bool

    def passed(self) -> bool:
        """Return whether every applied clause succeeded."""

        coverage_ok = self.coverage_within_limit is not False
        return (
            self.wape_improved
            and self.bias_within_limit
            and coverage_ok
            and self.no_critical_segment_regression
        )

    def to_dict(self) -> dict[str, bool | None]:
        """Return the JSON object for these checks."""

        return {
            "bias_within_limit": self.bias_within_limit,
            "coverage_within_limit": self.coverage_within_limit,
            "no_critical_segment_regression": self.no_critical_segment_regression,
            "wape_improved": self.wape_improved,
        }


@dataclass(frozen=True)
class SegmentRegression:
    """One category whose WAPE worsened by more than the allowed amount."""

    category_id: str
    candidate_wape: float
    reference_wape: float

    def to_dict(self) -> dict[str, object]:
        """Return the JSON object for this finding."""

        return {
            "candidate_wape": self.candidate_wape,
            "category_id": self.category_id,
            "reference_wape": self.reference_wape,
        }


@dataclass(frozen=True)
class PromotionDecision:
    """Persisted record of one gate run.

    ``p90_coverage`` is null when the candidate emits no ``p90``. ``reason``
    is ``quality_gate`` when the gate rejects the candidate, and null when
    the candidate is waiting for a person. ``regressed_categories`` names
    each category that failed the segment check.
    """

    candidate_id: str
    reference_id: str
    thresholds: PromotionThresholds
    checks: GateChecks
    status: ModelStatus
    reason: str | None
    created_at: datetime
    p90_coverage: float | None
    regressed_categories: tuple[SegmentRegression, ...]

    def __post_init__(self) -> None:
        if self.status not in {ModelStatus.PENDING_APPROVAL, ModelStatus.REJECTED}:
            raise ValueError("A gate decision status is pending approval or rejected.")
        if self.created_at.tzinfo is None:
            raise ValueError("Decision timestamp must include a timezone.")
        if self.status is ModelStatus.REJECTED and self.reason != QUALITY_GATE:
            raise ValueError("A gate rejection uses reason quality_gate.")
        if self.status is ModelStatus.PENDING_APPROVAL and self.reason is not None:
            raise ValueError("A pending decision has no rejection reason.")

    def to_dict(self) -> dict[str, object]:
        """Return the JSON document for this decision."""

        return {
            "candidate_id": self.candidate_id,
            "checks": self.checks.to_dict(),
            "created_at": self.created_at.isoformat(),
            "p90_coverage": self.p90_coverage,
            "reason": self.reason,
            "reference_id": self.reference_id,
            "regressed_categories": [item.to_dict() for item in self.regressed_categories],
            "status": self.status.value,
            "thresholds": self.thresholds.to_dict(),
        }


def empirical_p90_coverage(actual: Sequence[float], p90: Sequence[float]) -> float:
    """Return the share of actuals that fall at or below the P90 forecast.

    This is the coverage number a quantile model supplies to the gate. A
    model with no ``p90`` does not call this function.
    """

    actual_values = np.asarray(list(actual), dtype=np.float64)
    forecast_values = np.asarray(list(p90), dtype=np.float64)
    if actual_values.ndim != 1 or forecast_values.ndim != 1:
        raise ValueError("Actual and p90 must be one-dimensional sequences.")
    if actual_values.shape != forecast_values.shape:
        raise ValueError("Actual and p90 must have the same length.")
    if actual_values.size == 0:
        raise ValueError("Cannot compute P90 coverage for an empty forecast.")
    if not bool(np.isfinite(actual_values).all() and np.isfinite(forecast_values).all()):
        raise ValueError("P90 coverage requires finite actual and p90 values.")
    covered = actual_values <= forecast_values
    return float(np.mean(covered))


def gate(
    candidate: EvaluationReport,
    reference: EvaluationReport,
    *,
    candidate_id: str,
    reference_id: str,
    p90_coverage: float | None = None,
    thresholds: PromotionThresholds | None = None,
    now: datetime | None = None,
) -> PromotionDecision:
    """Compare ``candidate`` with ``reference`` and return a decision.

    Category WAPE comes from each report's category slices. Pass
    ``p90_coverage`` from :func:`empirical_p90_coverage` when the candidate
    report has a P90 pinball loss. Omit it when that loss is null. The
    returned status is ``PENDING_APPROVAL`` or ``REJECTED``.
    """

    limits = thresholds if thresholds is not None else PromotionThresholds()
    _require_id(candidate_id, "Candidate id")
    _require_id(reference_id, "Reference id")
    _require_same_dataset(candidate, reference)
    category_wape = _category_wape(candidate, reference)
    coverage = _coverage_input(candidate, p90_coverage)
    regressions = _segment_regressions(
        category_wape[0],
        category_wape[1],
        limits.max_segment_wape_regression,
    )
    checks = GateChecks(
        wape_improved=candidate.global_metrics.wape < reference.global_metrics.wape,
        bias_within_limit=abs(candidate.global_metrics.bias) < limits.max_bias,
        coverage_within_limit=_coverage_check(coverage, limits.min_p90_coverage),
        no_critical_segment_regression=not regressions,
    )
    rejected = not checks.passed()
    return PromotionDecision(
        candidate_id=candidate_id,
        reference_id=reference_id,
        thresholds=limits,
        checks=checks,
        status=ModelStatus.REJECTED if rejected else ModelStatus.PENDING_APPROVAL,
        reason=QUALITY_GATE if rejected else None,
        created_at=_timestamp(now),
        p90_coverage=coverage,
        regressed_categories=regressions,
    )


def _coverage_input(candidate: EvaluationReport, p90_coverage: float | None) -> float | None:
    has_p90 = candidate.global_metrics.pinball_loss_p90 is not None
    if not has_p90:
        if p90_coverage is not None:
            raise ValueError("A model without p90 does not supply P90 coverage.")
        return None
    if p90_coverage is None:
        raise ValueError("A model that emits p90 must supply empirical P90 coverage.")
    if isinstance(p90_coverage, bool) or not isinstance(p90_coverage, int | float):
        raise ValueError("P90 coverage must be a number.")
    coverage = float(p90_coverage)
    if not 0.0 <= coverage <= 1.0:
        raise ValueError("P90 coverage must be between 0 and 1.")
    return coverage


def _coverage_check(coverage: float | None, minimum: float) -> bool | None:
    if coverage is None:
        return None
    return coverage >= minimum


def _segment_regressions(
    candidate_wape: Mapping[str, float],
    reference_wape: Mapping[str, float],
    limit: float,
) -> tuple[SegmentRegression, ...]:
    findings: list[SegmentRegression] = []
    for category_id in sorted(set(candidate_wape) & set(reference_wape)):
        candidate_value = candidate_wape[category_id]
        reference_value = reference_wape[category_id]
        if candidate_value - reference_value > limit:
            findings.append(
                SegmentRegression(
                    category_id=category_id,
                    candidate_wape=candidate_value,
                    reference_wape=reference_value,
                )
            )
    return tuple(findings)


def _category_wape(
    candidate: EvaluationReport,
    reference: EvaluationReport,
) -> tuple[dict[str, float], dict[str, float]]:
    return _wape_by_category(candidate), _wape_by_category(reference)


def _wape_by_category(report: EvaluationReport) -> dict[str, float]:
    found: dict[str, float] = {}
    for item in report.by_category:
        if item.key in found:
            raise ValueError(f"Category {item.key} is repeated on the evaluation report.")
        found[item.key] = item.metrics.wape
    return found


def _require_same_dataset(candidate: EvaluationReport, reference: EvaluationReport) -> None:
    if candidate.dataset_version is None or reference.dataset_version is None:
        return
    if candidate.dataset_version != reference.dataset_version:
        raise ValueError("Candidate and reference must be scored on the same dataset version.")


def _require_id(value: str, label: str) -> None:
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{label} must be a non-empty string.")


def _timestamp(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("Decision timestamp must include a timezone.")
    return now
