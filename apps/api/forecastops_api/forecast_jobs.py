"""Run a queued forecast through local or batch inference.

The worker resolves the approved model, writes
``tenant/{tenant_id}/forecasts/{id}/forecast.json``,
and loads those points back into metadata. A failure is stored as ``FAILED``
with an English message that does not include dataset rows.
"""

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import pyarrow as pa
import structlog

from forecastops_api.artifacts import ArtifactStore
from forecastops_api.observability import record_forecast_job
from forecastops_api.persistence import ForecastPointRow, ForecastRunRow
from forecastops_api.repositories import MetadataRepository
from forecastops_api.settings import ExecutionMode, Settings
from forecastops_ml.data import load_dataset
from forecastops_ml.inference import (
    BatchInference,
    InferenceError,
    StoredForecastPoint,
    advance_status,
    execute_prediction,
    forecast_document,
    load_forecast_document,
    local_family_forecast,
    public_inference_error,
    quantile_forecast,
)


class ForecastPredictor(Protocol):
    """Stand-in predictor used to drive a forecast in tests."""

    def predict(
        self,
        frame: pa.Table,
        *,
        horizon: int,
        granularity: str,
    ) -> Sequence[StoredForecastPoint]:
        """Return points for ``frame``."""


class ForecastExecutor:
    """Advance one ``QUEUED`` forecast run to ``SUCCEEDED`` or ``FAILED``."""

    def __init__(
        self,
        repository: MetadataRepository,
        artifacts: ArtifactStore,
        batch: BatchInference | None = None,
        predictor: ForecastPredictor | None = None,
    ) -> None:
        self._repository = repository
        self._artifacts = artifacts
        self._batch = batch
        self._predictor = predictor

    def run(self, forecast: ForecastRunRow) -> None:
        """Execute ``forecast`` when it is still queued."""

        if forecast.status != "QUEUED":
            return
        started = time.perf_counter()
        try:
            forecast.status = advance_status(forecast.status, "start")
            self._repository.save()
            result = execute_prediction(lambda: self._predict(forecast))
            if result.status != "SUCCEEDED":
                self._fail(forecast, result.error_message or "Batch inference failed.")
                return
            try:
                body = forecast_document(result.points)
                name = f"{forecast.id}/forecast.json"
                uri = self._artifacts.put("forecasts", name, body)
                loaded = load_forecast_document(self._artifacts.get(uri))
                self._repository.add_points([_point_row(forecast.id, point) for point in loaded])
                forecast.output_uri = uri
                forecast.error_message = None
                forecast.status = advance_status(forecast.status, "succeed")
                self._repository.save()
            except Exception as exc:
                self._fail(forecast, public_inference_error(exc))
        finally:
            if forecast.status in {"SUCCEEDED", "FAILED"}:
                outcome = "succeeded" if forecast.status == "SUCCEEDED" else "failed"
                duration = time.perf_counter() - started
                record_forecast_job(outcome=outcome, latency_seconds=duration)
                structlog.get_logger().info(
                    "forecast.finished",
                    forecast_run_id=forecast.id,
                    model_version_id=forecast.model_version_id,
                    status=forecast.status,
                    artifact_uri=forecast.output_uri,
                    duration_seconds=round(duration, 3),
                )

    def _fail(self, forecast: ForecastRunRow, message: str) -> None:
        forecast.status = advance_status(forecast.status, "fail")
        forecast.error_message = message
        self._repository.save()

    def _predict(self, forecast: ForecastRunRow) -> tuple[StoredForecastPoint, ...]:
        model = self._repository.get_model(forecast.model_version_id)
        dataset = self._repository.get_dataset(forecast.dataset_id)
        if model is None or dataset is None:
            raise InferenceError("The forecast is missing its model or dataset.")
        if self._predictor is not None:
            frame = _frame(dataset.uri)
            return tuple(
                self._predictor.predict(
                    frame,
                    horizon=forecast.horizon,
                    granularity=forecast.granularity,
                )
            )
        if model.model_family == "deepar" and self._batch is not None:
            return self._batch.run(
                run_id=forecast.id,
                model_name=model.id,
                horizon=forecast.horizon,
                granularity=forecast.granularity,
                object_prefix=f"tenant/{forecast.tenant_id}",
            )
        frame = _frame(dataset.uri)
        if model.model_family == "deepar":
            return quantile_forecast(
                frame,
                horizon=forecast.horizon,
                granularity=forecast.granularity,
            )
        return local_family_forecast(
            frame,
            model.model_family,
            horizon=forecast.horizon,
            granularity=forecast.granularity,
        )


def select_batch_inference(settings: Settings) -> BatchInference | None:
    """Return a batch adapter when cloud forecasts are enabled.

    Local mode returns none and does not construct a cloud client. Online
    inference does not open a client either: the adapter refuses the job
    instead of creating an endpoint.
    """

    if settings.execution_mode is not ExecutionMode.AWS or not settings.sagemaker_enabled:
        return None
    if settings.online_inference:
        return BatchInference(
            None,
            None,
            bucket=settings.artifacts_bucket,
            sagemaker_enabled=True,
            online_inference=True,
        )
    from forecastops_ml.inference import open_batch_inference

    return open_batch_inference(region=settings.aws_region, bucket=settings.artifacts_bucket)


def _frame(uri: str) -> pa.Table:
    try:
        frame, _dimensions = load_dataset(Path(uri))
    except ValueError as exc:
        raise InferenceError(public_inference_error(exc)) from exc
    return frame


def _point_row(forecast_run_id: str, point: StoredForecastPoint) -> ForecastPointRow:
    return ForecastPointRow(
        forecast_run_id=forecast_run_id,
        series_id=point.series_id,
        date=point.date,
        p10=point.p10,
        p50=point.p50,
        p90=point.p90,
        actual=point.actual,
    )
