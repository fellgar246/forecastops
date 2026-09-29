"""Forecast run transitions shared by local and batch execution."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from forecastops_ml.inference.errors import (
    BatchCapacityError,
    InferenceError,
    public_inference_error,
)
from forecastops_ml.inference.points import StoredForecastPoint

ForecastEvent = Literal["start", "succeed", "fail"]


@dataclass(frozen=True)
class ForecastResult:
    """Points produced by one prediction, or a safe failure message."""

    status: Literal["SUCCEEDED", "FAILED"]
    points: tuple[StoredForecastPoint, ...]
    error_message: str | None


def advance_status(current: str, event: ForecastEvent) -> str:
    """Return the next status for ``event``.

    ``QUEUED`` starts, ``RUNNING`` succeeds or fails, and a failed start also
    ends as ``FAILED``.
    """

    if event == "start" and current == "QUEUED":
        return "RUNNING"
    if event == "succeed" and current == "RUNNING":
        return "SUCCEEDED"
    if event == "fail" and current in {"QUEUED", "RUNNING"}:
        return "FAILED"
    raise InferenceError(f"Cannot {event} a forecast in {current}.")


def admit_forecast(*, jobs_started_today: int, daily_limit: int, replay: bool) -> None:
    """Allow a new forecast, or a replay of one that already exists.

    A replay does not consume another slot. A new job at the ceiling raises
    :class:`BatchCapacityError`.
    """

    if replay:
        return
    if type(daily_limit) is not int or daily_limit < 0:
        raise InferenceError("The daily batch inference limit is invalid.")
    if type(jobs_started_today) is not int or jobs_started_today < 0:
        raise InferenceError("The daily batch inference count is invalid.")
    if jobs_started_today >= daily_limit:
        raise BatchCapacityError(daily_limit)


def execute_prediction(
    predict: Callable[[], Sequence[StoredForecastPoint]],
) -> ForecastResult:
    """Run ``predict`` and return points or a safe English failure."""

    try:
        points = tuple(predict())
    except Exception as exc:
        return ForecastResult(
            status="FAILED",
            points=(),
            error_message=public_inference_error(exc),
        )
    if not points:
        return ForecastResult(
            status="FAILED",
            points=(),
            error_message="Batch inference produced no forecast points.",
        )
    return ForecastResult(status="SUCCEEDED", points=points, error_message=None)
