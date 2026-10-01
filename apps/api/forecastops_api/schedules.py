"""Scheduled forecast refresh, error evaluation, and retrain requests.

These operations run from a local command or an HTTP route. They do not
construct a cloud scheduler client. A monthly run records a confirmation
request and does not start training.
"""

from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

from forecastops_api.errors import ApiError
from forecastops_api.persistence import (
    DataRefreshMarkerRow,
    ForecastErrorEvaluationRow,
    ForecastPointRow,
    RetrainRequestRow,
)
from forecastops_api.repositories import MetadataRepository
from forecastops_api.schemas import (
    ForecastCreate,
    ForecastErrorEvaluationResponse,
    RetrainRequestResponse,
    ScheduledForecastResponse,
)
from forecastops_api.services import ForecastService
from forecastops_ml.data import load_dataset
from forecastops_ml.evaluation.backtest import make_series_id
from forecastops_ml.evaluation.metrics import bias, wape

DAILY_FORECAST = "daily_forecast"
WEEKLY_EVALUATION = "weekly_evaluation"
MONTHLY_RETRAIN = "monthly_retrain"
SCHEDULE_ACTIONS = (DAILY_FORECAST, WEEKLY_EVALUATION, MONTHLY_RETRAIN)

PENDING = "PENDING"
CONFIRMED = "CONFIRMED"
DAILY_HORIZON_DAYS = 7


class ScheduleService:
    """Run the three clocks against the forecast service."""

    def __init__(self, forecasts: ForecastService, repository: MetadataRepository) -> None:
        self._forecasts = forecasts
        self._repository = repository

    def refresh_and_forecast(self) -> ScheduledForecastResponse:
        """Record a data refresh marker and generate the latest forecast."""

        production = self._repository.production_model()
        if production is None:
            raise ApiError(
                409,
                "no_production_model",
                "No production model is available for the scheduled forecast.",
            )
        dataset = self._forecasts.get_dataset(production.dataset_id)
        refreshed_at = _now()
        marker = DataRefreshMarkerRow(
            id=str(uuid4()),
            dataset_id=dataset.id,
            refreshed_at=refreshed_at,
            created_at=refreshed_at,
        )
        self._repository.add(marker)
        day = refreshed_at.date().isoformat()
        forecast = self._forecasts.start_forecast(
            _forecast_request(production.id, f"daily-forecast-{production.id}-{day}")
        )
        return ScheduledForecastResponse(
            marker_id=marker.id,
            dataset_id=dataset.id,
            refreshed_at=marker.refreshed_at,
            forecast_id=forecast.id,
            forecast_status=forecast.status,  # type: ignore[arg-type]
        )

    def evaluate_forecast_error(self) -> ForecastErrorEvaluationResponse:
        """Score the latest forecast on dates whose actuals are now in the dataset."""

        forecast = self._repository.latest_succeeded_forecast()
        if forecast is None:
            raise ApiError(
                409,
                "no_forecast",
                "No succeeded forecast is available to evaluate.",
            )
        dataset = self._forecasts.get_dataset(forecast.dataset_id)
        actuals = _arrived_actuals(Path(dataset.uri))
        points = self._repository.list_points(forecast.id)
        matched = [
            (point, actuals[(point.series_id, point.date)])
            for point in points
            if (point.series_id, point.date) in actuals
        ]
        wape_value, bias_value = _score(matched)
        for point, actual in matched:
            point.actual = actual
        created_at = _now()
        row = ForecastErrorEvaluationRow(
            id=str(uuid4()),
            forecast_run_id=forecast.id,
            compared_points=len(matched),
            wape=wape_value,
            bias=bias_value,
            created_at=created_at,
        )
        self._repository.add(row)
        self._repository.save()
        return _evaluation_response(row)

    def request_retrain(self) -> RetrainRequestResponse:
        """Record a retrain request and stop. Training waits for confirmation."""

        existing = self._repository.pending_retrain_request()
        if existing is not None:
            return _retrain_response(existing)
        production = self._repository.production_model()
        if production is None:
            raise ApiError(
                409,
                "no_production_model",
                "No production model is available to retrain.",
            )
        requested_at = _now()
        row = RetrainRequestRow(
            id=str(uuid4()),
            dataset_id=production.dataset_id,
            model_version_id=production.id,
            model_family=production.model_family,
            status=PENDING,
            training_run_id=None,
            requested_at=requested_at,
            confirmed_at=None,
            confirmed_by=None,
            created_at=requested_at,
        )
        self._repository.add(row)
        self._repository.save()
        return _retrain_response(row)

    def confirm_retrain(self, request_id: str, actor_id: str) -> RetrainRequestResponse:
        """Start training for a pending request. Do not approve the new model."""

        row = self._repository.get_retrain_request(request_id)
        if row is None:
            raise ApiError(404, "not_found", "Retrain request was not found.")
        if row.status != PENDING:
            raise ApiError(
                409,
                "invalid_transition",
                f"Retrain request cannot be confirmed from {row.status}.",
            )
        run = self._forecasts.retrain()
        row.status = CONFIRMED
        row.training_run_id = run.id
        row.confirmed_at = _now()
        row.confirmed_by = actor_id
        self._repository.save()
        return _retrain_response(row)

    def list_retrain_requests(self) -> list[RetrainRequestResponse]:
        """Return every retrain request."""

        return [_retrain_response(row) for row in self._repository.list_retrain_requests()]

    def execute(
        self,
        action: str,
    ) -> ScheduledForecastResponse | ForecastErrorEvaluationResponse | RetrainRequestResponse:
        """Run one named clock. Unknown names are rejected."""

        if action == DAILY_FORECAST:
            return self.refresh_and_forecast()
        if action == WEEKLY_EVALUATION:
            return self.evaluate_forecast_error()
        if action == MONTHLY_RETRAIN:
            return self.request_retrain()
        raise ApiError(404, "not_found", "Schedule action was not found.")


def _forecast_request(model_id: str, idempotency_key: str) -> ForecastCreate:
    return ForecastCreate(
        model_id=model_id,
        horizon=DAILY_HORIZON_DAYS,
        granularity="day",
        idempotency_key=idempotency_key,
    )


def _arrived_actuals(dataset: Path) -> dict[tuple[str, date], float]:
    try:
        frame, _dimensions = load_dataset(dataset)
    except ValueError as exc:
        raise ApiError(422, "invalid_dataset", str(exc)) from exc
    found: dict[tuple[str, date], float] = {}
    for day, store_id, sku_id, sold in zip(
        frame.column("date").to_pylist(),
        frame.column("store_id").to_pylist(),
        frame.column("sku_id").to_pylist(),
        frame.column("units_sold").to_pylist(),
        strict=True,
    ):
        if not isinstance(day, date) or sold is None:
            continue
        if not isinstance(store_id, str) or not isinstance(sku_id, str):
            continue
        if isinstance(sold, bool) or not isinstance(sold, int | float):
            continue
        found[(make_series_id(store_id, sku_id), day)] = float(sold)
    return found


def _score(matched: list[tuple[ForecastPointRow, float]]) -> tuple[float | None, float | None]:
    if not matched:
        return None, None
    actual = [value for _point, value in matched]
    forecast = [point.p50 for point, _value in matched]
    try:
        return wape(actual, forecast), bias(actual, forecast)
    except ValueError as exc:
        raise ApiError(422, "forecast_error_undefined", str(exc)) from exc


def _evaluation_response(row: ForecastErrorEvaluationRow) -> ForecastErrorEvaluationResponse:
    return ForecastErrorEvaluationResponse(
        id=row.id,
        forecast_run_id=row.forecast_run_id,
        compared_points=row.compared_points,
        wape=row.wape,
        bias=row.bias,
        created_at=row.created_at,
    )


def _retrain_response(row: RetrainRequestRow) -> RetrainRequestResponse:
    return RetrainRequestResponse(
        id=row.id,
        status=row.status,  # type: ignore[arg-type]
        dataset_id=row.dataset_id,
        model_version_id=row.model_version_id,
        model_family=row.model_family,
        training_run_id=row.training_run_id,
        requested_at=row.requested_at,
        confirmed_at=row.confirmed_at,
        confirmed_by=row.confirmed_by,
    )


def _now() -> datetime:
    return datetime.now(UTC)
