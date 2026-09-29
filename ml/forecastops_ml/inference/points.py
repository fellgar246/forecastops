"""Forecast points stored in a ``forecasts/`` artifact."""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from forecastops_ml.inference.errors import InferenceError


@dataclass(frozen=True)
class StoredForecastPoint:
    """One forecast point loaded from an artifact or a model.

    ``p50`` is always present. ``p10`` and ``p90`` are either both present or
    both absent. A single-value model leaves the outer quantiles empty.
    """

    series_id: str
    date: date
    p50: float
    p10: float | None = None
    p90: float | None = None
    actual: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.series_id, str) or self.series_id.strip() == "":
            raise InferenceError("Forecast series_id must be a non-empty string.")
        if type(self.date) is not date:
            raise InferenceError("Forecast dates must be calendar dates.")
        p50 = _finite(self.p50)
        object.__setattr__(self, "p50", p50)
        if (self.p10 is None) != (self.p90 is None):
            raise InferenceError("Forecast points must include both p10 and p90.")
        if self.p10 is not None and self.p90 is not None:
            p10 = _finite(self.p10)
            p90 = _finite(self.p90)
            if p10 > p50 or p50 > p90:
                raise InferenceError("Forecast quantiles must satisfy p10 <= p50 <= p90.")
            object.__setattr__(self, "p10", p10)
            object.__setattr__(self, "p90", p90)
        if self.actual is not None:
            object.__setattr__(self, "actual", _finite(self.actual))

    def to_dict(self) -> dict[str, object]:
        """Return the JSON object for this point."""

        return {
            "actual": self.actual,
            "date": self.date.isoformat(),
            "p10": self.p10,
            "p50": self.p50,
            "p90": self.p90,
            "series_id": self.series_id,
        }


def forecast_document(points: Sequence[StoredForecastPoint]) -> bytes:
    """Serialize ``points`` as the forecast artifact body."""

    if not points:
        raise InferenceError("Batch inference produced no forecast points.")
    payload = {"points": [point.to_dict() for point in points]}
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def load_forecast_document(body: bytes) -> tuple[StoredForecastPoint, ...]:
    """Load points from a forecast artifact.

    The artifact is the source for the rows written into metadata.
    """

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InferenceError("The forecast artifact is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise InferenceError("The forecast artifact must be a JSON object.")
    raw_points = payload.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        raise InferenceError("The forecast artifact has no points.")
    loaded: list[StoredForecastPoint] = []
    for item in raw_points:
        if not isinstance(item, Mapping):
            raise InferenceError("Each forecast point must be a JSON object.")
        loaded.append(_point_from_json(item))
    return tuple(loaded)


def _point_from_json(item: Mapping[str, object]) -> StoredForecastPoint:
    series_id = item.get("series_id")
    raw_date = item.get("date")
    if not isinstance(series_id, str) or not isinstance(raw_date, str):
        raise InferenceError("A forecast point needs a series_id and a date.")
    try:
        day = date.fromisoformat(raw_date)
    except ValueError as exc:
        raise InferenceError("Forecast point dates must be ISO calendar dates.") from exc
    return StoredForecastPoint(
        series_id=series_id,
        date=day,
        p50=_required_number(item.get("p50")),
        p10=_optional_number(item.get("p10")),
        p90=_optional_number(item.get("p90")),
        actual=_optional_number(item.get("actual")),
    )


def _required_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InferenceError("Forecast p50 must be a finite number.")
    return float(value)


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InferenceError("Forecast quantiles must be finite numbers or null.")
    return float(value)


def _finite(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InferenceError("Forecast quantiles must be finite numbers.")
    number = float(value)
    if not math.isfinite(number):
        raise InferenceError("Forecast quantiles must be finite numbers.")
    return number
