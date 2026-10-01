"""Drift monitoring for the production model.

The report compares the recent window with the training baseline. When the
status is ``RETRAIN_RECOMMENDED`` it opens a confirmation request. It does
not start training and it does not approve a model.
"""

from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from forecastops_api.errors import ApiError
from forecastops_api.persistence import MonitoringReportRow
from forecastops_api.repositories import MetadataRepository
from forecastops_api.schedules import ScheduleService
from forecastops_api.schemas import MonitoringMetrics, MonitoringReportResponse
from forecastops_api.settings import Settings
from forecastops_ml.data import load_dataset
from forecastops_ml.monitoring import DriftThresholds, build_monitoring_report, split_recent_window
from forecastops_ml.monitoring.report import MonitoringReport


class TrainingClient(Protocol):
    """A training job submission. Monitoring accepts one and does not call it."""

    def create_training_job(self, request: Mapping[str, object]) -> str:
        """Submit one training job."""


class MonitoringService:
    """Build, store, and read drift monitoring reports."""

    def __init__(
        self,
        repository: MetadataRepository,
        schedules: ScheduleService,
        settings: Settings,
        training_client: TrainingClient | None = None,
    ) -> None:
        self._repository = repository
        self._schedules = schedules
        self._settings = settings
        self._training_client = training_client

    def latest(self) -> MonitoringReportResponse | None:
        """Return the newest stored report, if monitoring has run."""

        row = self._repository.latest_monitoring_report()
        if row is None:
            return None
        return _response(row)

    def run(self, as_of: date | None = None) -> MonitoringReportResponse:
        """Compare the recent window with the baseline and store the report.

        A ``RETRAIN_RECOMMENDED`` status records a confirmation request.
        ``training_client`` is not used.
        """

        production = self._repository.production_model()
        if production is None:
            raise ApiError(
                409,
                "no_production_model",
                "No production model is available to monitor.",
            )
        dataset = self._repository.get_dataset(production.dataset_id)
        if dataset is None or dataset.date_max is None:
            raise ApiError(
                422,
                "monitoring_failed",
                "The production dataset has no date_max.",
            )
        try:
            frame, _dimensions = load_dataset(Path(dataset.uri))
            limits = _thresholds(self._settings)
            baseline, recent = split_recent_window(
                frame,
                recent_window_days=limits.recent_window_days,
            )
            report = build_monitoring_report(
                baseline,
                recent,
                as_of=as_of or _today(),
                date_max=dataset.date_max,
                approved_wape=_approved_wape(production.metrics),
                recent_wape=_recent_wape(self._repository),
                thresholds=limits,
            )
        except ValueError as exc:
            raise ApiError(422, "monitoring_failed", str(exc)) from exc
        retrain_request_id = None
        if report.status == "RETRAIN_RECOMMENDED":
            retrain_request_id = self._schedules.request_retrain().id
        created_at = datetime.now(UTC)
        row = MonitoringReportRow(
            id=str(uuid4()),
            model_version_id=production.id,
            dataset_id=dataset.id,
            status=report.status,
            as_of=report.as_of,
            date_max=report.date_max,
            baseline_start=report.baseline_start,
            baseline_end=report.baseline_end,
            recent_start=report.recent_start,
            recent_end=report.recent_end,
            metrics=_metrics(report),
            retrain_request_id=retrain_request_id,
            created_at=created_at,
        )
        self._repository.add(row)
        return _response(row)


def _thresholds(settings: Settings) -> DriftThresholds:
    try:
        return DriftThresholds(
            psi_warning=settings.psi_warning_threshold,
            psi_retrain=settings.psi_retrain_threshold,
            wape_degradation_limit=settings.wape_degradation_limit,
            max_dataset_age_days=settings.max_dataset_age_days,
            frequency_warning_delta=settings.frequency_warning_delta,
            recent_window_days=settings.recent_window_days,
        )
    except ValueError as exc:
        raise ApiError(422, "monitoring_failed", str(exc)) from exc


def _approved_wape(metrics: dict[str, object] | None) -> float | None:
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("wape")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _recent_wape(repository: MetadataRepository) -> float | None:
    row = repository.latest_forecast_error()
    if row is None or row.wape is None:
        return None
    return float(row.wape)


def _metrics(report: MonitoringReport) -> dict[str, object]:
    return {
        "approved_wape": report.approved_wape,
        "coverage": [item.to_dict() for item in report.coverage],
        "dataset_age_days": report.dataset_age_days,
        "frequencies": [item.to_dict() for item in report.frequencies],
        "numeric": [item.to_dict() for item in report.numeric],
        "reasons": list(report.reasons),
        "recent_wape": report.recent_wape,
        "thresholds": report.thresholds.to_dict(),
        "wape_degradation": report.wape_degradation,
    }


def _response(row: MonitoringReportRow) -> MonitoringReportResponse:
    metrics = MonitoringMetrics.model_validate(row.metrics)
    return MonitoringReportResponse(
        id=row.id,
        status=row.status,  # type: ignore[arg-type]
        model_version_id=row.model_version_id,
        dataset_id=row.dataset_id,
        as_of=row.as_of,
        date_max=row.date_max,
        baseline_start=row.baseline_start,
        baseline_end=row.baseline_end,
        recent_start=row.recent_start,
        recent_end=row.recent_end,
        dataset_age_days=metrics.dataset_age_days,
        approved_wape=metrics.approved_wape,
        recent_wape=metrics.recent_wape,
        wape_degradation=metrics.wape_degradation,
        numeric=metrics.numeric,
        frequencies=metrics.frequencies,
        coverage=metrics.coverage,
        reasons=metrics.reasons,
        thresholds=metrics.thresholds,
        retrain_request_id=row.retrain_request_id,
        created_at=row.created_at,
    )


def _today() -> date:
    return datetime.now(UTC).date()
