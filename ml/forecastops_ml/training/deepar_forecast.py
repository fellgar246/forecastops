"""Map DeepAR quantile output onto forecast points.

P10, P50, and P90 are required. A point is rejected when any value is
non-finite or when the quantiles are out of order. ``FakeDeepARRunner``
returns a checked-in forecast so local tests can run without a training
account.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from forecastops_ml.training.deepar_export import DeepARRecord, period_step

PREDICTION_QUANTILES = (0.1, 0.5, 0.9)
_QUANTILE_KEYS = ("0.1", "0.5", "0.9")
_FIXTURE = Path(__file__).parent / "fixtures" / "deepar_forecast.json"


@dataclass(frozen=True)
class ForecastPoint:
    """One probabilistic forecast for a series on a date."""

    series_id: str
    date: date
    p10: float
    p50: float
    p90: float
    actual: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.series_id, str) or self.series_id.strip() == "":
            raise ValueError("Forecast series_id must be a non-empty string.")
        if type(self.date) is not date:
            raise ValueError("Forecast dates must be calendar dates.")
        values = {
            "p10": _finite(self.p10, self.series_id, self.date),
            "p50": _finite(self.p50, self.series_id, self.date),
            "p90": _finite(self.p90, self.series_id, self.date),
        }
        if values["p10"] > values["p50"] or values["p50"] > values["p90"]:
            stamp = self.date.isoformat()
            raise ValueError(
                f"Series {self.series_id} on {stamp} has unordered quantiles. "
                "p10, p50, and p90 must satisfy p10 <= p50 <= p90."
            )
        object.__setattr__(self, "p10", values["p10"])
        object.__setattr__(self, "p50", values["p50"])
        object.__setattr__(self, "p90", values["p90"])
        if self.actual is not None:
            object.__setattr__(self, "actual", _finite(self.actual, self.series_id, self.date))

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


def map_deepar_predictions(
    series: Sequence[DeepARRecord],
    predictions: Sequence[Mapping[str, object]],
    *,
    grain: str,
    prediction_length: int,
) -> tuple[ForecastPoint, ...]:
    """Turn DeepAR quantile objects into forecast points, in series order.

    The first forecast date is the period after each series' last target
    timestamp. Quantile arrays must each have ``prediction_length`` values.
    """

    step = period_step(grain)
    if type(prediction_length) is not int or prediction_length < 1:
        raise ValueError("prediction_length must be a positive integer.")
    if len(series) != len(predictions):
        raise ValueError("Each series needs one prediction object.")
    points: list[ForecastPoint] = []
    for record, prediction in zip(series, predictions, strict=True):
        quantiles = _quantile_arrays(prediction, record.series_id, prediction_length)
        origin = record.start + len(record.target) * step
        for index in range(prediction_length):
            points.append(
                ForecastPoint(
                    series_id=record.series_id,
                    date=origin + index * step,
                    p10=quantiles["0.1"][index],
                    p50=quantiles["0.5"][index],
                    p90=quantiles["0.9"][index],
                )
            )
    if not points:
        raise ValueError("DeepAR predictions produced no forecast points.")
    return tuple(points)


def load_fixture_forecast() -> tuple[ForecastPoint, ...]:
    """Return the checked-in forecast after the same quantile checks."""

    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DeepAR fixture forecast must be a JSON object.")
    grain = payload.get("grain")
    prediction_length = payload.get("prediction_length")
    raw_series = payload.get("series")
    if (
        not isinstance(grain, str)
        or type(prediction_length) is not int
        or not isinstance(raw_series, list)
    ):
        raise ValueError("DeepAR fixture forecast is missing grain, prediction_length, or series.")
    records: list[DeepARRecord] = []
    predictions: list[dict[str, object]] = []
    for item in raw_series:
        if not isinstance(item, dict):
            raise ValueError("DeepAR fixture series entries must be JSON objects.")
        record, prediction = _fixture_series(item, grain)
        records.append(record)
        predictions.append(prediction)
    return map_deepar_predictions(
        records,
        predictions,
        grain=grain,
        prediction_length=prediction_length,
    )


def forecast_points_json(points: Sequence[ForecastPoint]) -> str:
    """Serialize forecast points for the ``models/`` artifact."""

    body = {"points": [point.to_dict() for point in points]}
    return json.dumps(body, indent=2, sort_keys=True) + "\n"


class FakeDeepARRunner:
    """Return the fixture forecast without calling a training service."""

    def run(self, channels: object = None) -> tuple[ForecastPoint, ...]:
        """Return the fixture forecast. ``channels`` is ignored."""

        del channels
        return load_fixture_forecast()

    def write_artifacts(self, root: Path) -> tuple[ForecastPoint, ...]:
        """Write the fixture under ``training/`` and ``models/``."""

        points = self.run()
        training = root / "training" / "deepar-fixture"
        models = root / "models" / "deepar-fixture"
        training.mkdir(parents=True, exist_ok=True)
        models.mkdir(parents=True, exist_ok=True)
        metrics = {
            "point_count": len(points),
            "quantiles": list(PREDICTION_QUANTILES),
            "source": "fixture",
        }
        (training / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (models / "forecast.json").write_text(forecast_points_json(points), encoding="utf-8")
        return points


def _fixture_series(
    item: Mapping[str, object], grain: str
) -> tuple[DeepARRecord, dict[str, object]]:
    series_id = item.get("series_id")
    start_text = item.get("start")
    target = item.get("target")
    quantiles = item.get("quantiles")
    if not isinstance(series_id, str) or not isinstance(start_text, str):
        raise ValueError("DeepAR fixture series need series_id and start.")
    if not isinstance(target, list) or not target or not isinstance(quantiles, dict):
        raise ValueError(f"DeepAR fixture series {series_id} needs target and quantiles.")
    start = date.fromisoformat(start_text)
    values = tuple(float(value) for value in target)
    zeros = tuple(0.0 for _ in values)
    record = DeepARRecord(
        series_id=series_id,
        start=start,
        target=values,
        cat=(0, 0),
        dynamic_feat=(zeros, zeros),
        target_end=start + (len(values) - 1) * period_step(grain),
    )
    return record, {"quantiles": quantiles}


def _quantile_arrays(
    prediction: Mapping[str, object],
    series_id: str,
    prediction_length: int,
) -> dict[str, list[float]]:
    raw = prediction.get("quantiles")
    if not isinstance(raw, Mapping):
        raise ValueError(f"Series {series_id} is missing quantile predictions.")
    parsed: dict[str, list[float]] = {}
    for key in _QUANTILE_KEYS:
        values = raw.get(key)
        if not isinstance(values, list) or len(values) != prediction_length:
            raise ValueError(
                f"Series {series_id} quantile {key} must have {prediction_length} values."
            )
        parsed[key] = [float(value) for value in values]
    return parsed


def _finite(value: float, series_id: str, day: date) -> float:
    number = float(value)
    if not math.isfinite(number):
        stamp = day.isoformat()
        raise ValueError(f"Series {series_id} on {stamp} has a non-finite quantile.")
    return number
