"""Deterministic checks that keep an explanation inside the package."""

import math
import re
from dataclasses import dataclass
from typing import NoReturn

from pydantic import ValidationError

from forecastops_ml.explanations.models import ExplanationDraft, ExplanationPackage

_QUANTILE_LABEL = re.compile(r"\bP10\b|\bP50\b|\bP90\b", re.IGNORECASE)
_NUMBER = re.compile(r"(?<!\w)(-?\d{1,3}(?:,\d{3})+|-?\d+(?:\.\d+)?)(?!\w)")

_failures = 0


class ExplanationValidationError(Exception):
    """A draft failed a deterministic check and must not be shown."""

    def __init__(self, check: str, message: str) -> None:
        self.check = check
        self.message = message
        super().__init__(message)


def explanation_validation_failures() -> int:
    """Return how many drafts have failed validation in this process."""

    return _failures


def reject_draft(
    check: str,
    message: str,
    cause: BaseException | None = None,
) -> NoReturn:
    """Count one failed check and stop validation."""

    _fail(check, message, cause)


def parse_draft(payload: object) -> ExplanationDraft:
    """Parse a draft. A schema failure counts as a validation failure."""

    if not isinstance(payload, dict):
        _fail("schema", "The explanation failed the schema check.")
    try:
        return ExplanationDraft.model_validate(payload)
    except ValidationError:
        _fail("schema", "The explanation failed the schema check.")


@dataclass(frozen=True)
class ExplanationChecks:
    """Pass or fail for number preservation, known signals, and uncertainty."""

    number_preserved: bool
    signals_known: bool
    uncertainty_acknowledged: bool


def grade_explanation(package: ExplanationPackage, draft: ExplanationDraft) -> ExplanationChecks:
    """Record each deterministic check. One failure does not hide the others.

    Grading does not increment :func:`explanation_validation_failures`. That
    counter belongs to drafts rejected by :func:`validate_explanation`.
    """

    return ExplanationChecks(
        number_preserved=_forecast_numbers_ok(package, draft),
        signals_known=_signals_ok(package, draft),
        uncertainty_acknowledged=draft.uncertainty_note.strip() != "",
    )


def validate_explanation(package: ExplanationPackage, draft: ExplanationDraft) -> None:
    """Check number preservation, known signals, and a non-empty uncertainty note."""

    if not _forecast_numbers_ok(package, draft):
        _fail("forecast_numbers", "The explanation failed the forecast_numbers check.")
    if not _signals_ok(package, draft):
        _fail("known_signals", "The explanation failed the known_signals check.")
    if draft.uncertainty_note.strip() == "":
        _fail("uncertainty_note", "The explanation failed the uncertainty_note check.")


def format_number(value: float) -> str:
    """Format a package number so the same value can be read back."""

    if not math.isfinite(value):
        raise ValueError("A forecast number must be finite.")
    nearest = round(value)
    if abs(value - nearest) <= 1e-9:
        return str(int(nearest))
    text = f"{value:.12f}".rstrip("0").rstrip(".")
    return text


def _forecast_numbers_ok(package: ExplanationPackage, draft: ExplanationDraft) -> bool:
    allowed = _package_numbers(package)
    totals = _forecast_totals(package)
    prose = "\n".join(
        [
            draft.summary,
            *draft.risks,
            draft.uncertainty_note,
            *draft.recommended_checks,
        ]
    )
    for number in _numbers_in(prose):
        if not _looks_like_forecast_total(number, totals):
            continue
        if any(math.isclose(number, candidate, rel_tol=0.0, abs_tol=1e-6) for candidate in allowed):
            continue
        return False
    return True


def _signals_ok(package: ExplanationPackage, draft: ExplanationDraft) -> bool:
    known = {signal.name for signal in package.signals}
    known.update(package.positive_drivers)
    known.update(package.negative_drivers)
    return all(name in known for name in draft.drivers)


def _package_numbers(package: ExplanationPackage) -> list[float]:
    found: list[float] = []

    def walk(node: object) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, int | float):
            found.append(float(node))
            return
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
            return
        if isinstance(node, list):
            for value in node:
                walk(value)

    walk(package.model_dump(mode="json"))
    return found


def _forecast_totals(package: ExplanationPackage) -> list[float]:
    values = [
        package.forecast.p10,
        package.forecast.p50,
        package.forecast.p90,
        package.history.previous_period_units,
    ]
    return [float(value) for value in values if value is not None]


def _looks_like_forecast_total(number: float, totals: list[float]) -> bool:
    if not totals:
        return abs(number) >= 1
    floor = min(abs(value) for value in totals) * 0.5
    return abs(number) >= max(floor, 1.0)


def _numbers_in(text: str) -> list[float]:
    cleaned = _QUANTILE_LABEL.sub(" ", text)
    numbers: list[float] = []
    for match in _NUMBER.finditer(cleaned):
        token = match.group(1).replace(",", "")
        numbers.append(float(token))
    return numbers


def _fail(check: str, message: str, cause: BaseException | None = None) -> NoReturn:
    global _failures
    _failures += 1
    error = ExplanationValidationError(check, message)
    if cause is not None:
        raise error from cause
    raise error
