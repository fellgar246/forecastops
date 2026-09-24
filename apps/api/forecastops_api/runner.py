"""In-process training and forecast jobs.

Jobs call the local forecasting library and write artifacts on disk. They do
not import a cloud SDK.
"""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pyarrow as pa

from forecastops_api.persistence import (
    ForecastPointRow,
    ForecastRunRow,
    ModelVersionRow,
    PromotionDecisionRow,
    TrainingRunRow,
)
from forecastops_api.repositories import Repository
from forecastops_ml.baselines.catalog import local_catalog
from forecastops_ml.evaluation.backtest import (
    Backtester,
    EvaluationReport,
    ForecastFrame,
    UnscoredModelError,
)
from forecastops_ml.promotion.gate import PromotionDecision, gate
from forecastops_ml.promotion.models import ModelStatus
from forecastops_ml.promotion.reference import reference_report

_LOCAL_FAMILIES = {"naive", "seasonal_naive", "holt_winters", "gradient_boosting"}
_DEFAULT_FOLDS = 3
_DEFAULT_HORIZON = 7


class LocalJobRunner:
    """Move training and forecast rows through their statuses."""

    def __init__(self, repository: Repository, artifact_dir: Path) -> None:
        self._repository = repository
        self._artifact_dir = artifact_dir

    def train(self, run: TrainingRunRow, frame: pa.Table) -> ModelVersionRow | None:
        """Fit, score, and register ``run``. Return the model, or none on failure."""

        self._training_status(run, "PREPROCESSING", started=True)
        if run.model_family not in _LOCAL_FAMILIES:
            message = f"{run.model_family} is not available in the local profile."
            self._fail_training(run, message)
            return None
        folds = _positive_int(run.configuration.get("folds"), _DEFAULT_FOLDS)
        horizon = _positive_int(run.configuration.get("horizon"), _DEFAULT_HORIZON)
        try:
            forecaster = local_catalog().get(run.model_family)
            self._training_status(run, "TRAINING")
            self._training_status(run, "EVALUATING")
            report = Backtester().evaluate(
                forecaster,
                frame,
                folds,
                horizon,
                dataset_version=run.dataset_version,
            )
        except (UnscoredModelError, ValueError) as exc:
            self._fail_training(run, str(exc))
            return None

        self._training_status(run, "REGISTERING")
        artifact = self._write_training_artifact(run, report)
        run.artifact_uri = str(artifact)
        run.metrics = report.to_dict()
        model = ModelVersionRow(
            id=_new_id(),
            model_family=run.model_family,
            version=self._repository.next_model_version(run.model_family),
            training_run_id=run.id,
            dataset_id=run.dataset_id,
            dataset_version=run.dataset_version,
            registry_arn="",
            status="TRAINED",
            metrics=report.global_metrics.to_dict(),
            approved_at=None,
            approved_by=None,
            rejected_by=None,
            rejection_reason=None,
            created_at=_now(),
        )
        self._repository.add(model)
        decision = self._decide(model, frame, report, folds, horizon)
        model.status = decision.status.value
        metrics = dict(model.metrics or {})
        metrics["p90_coverage"] = decision.p90_coverage
        model.metrics = metrics
        if decision.status is ModelStatus.REJECTED:
            model.rejection_reason = decision.reason
        self._repository.add(
            PromotionDecisionRow(
                id=_new_id(),
                candidate_id=model.id,
                reference_id=decision.reference_id,
                thresholds=decision.thresholds.to_dict(),
                checks=decision.checks.to_dict(),
                status=decision.status.value,
                reason=decision.reason,
                p90_coverage=decision.p90_coverage,
                regressed_categories=[item.to_dict() for item in decision.regressed_categories],
                actor_id=None,
                created_at=decision.created_at,
            )
        )
        self._repository.save()
        self._training_status(run, "COMPLETED", finished=True)
        return model

    def forecast(self, forecast: ForecastRunRow, frame: pa.Table, family: str) -> None:
        """Fit ``family`` on ``frame`` and store the next horizon as points."""

        forecast.status = "RUNNING"
        self._repository.save()
        try:
            forecaster = local_catalog().get(family)
            forecaster.fit(frame, None)
            future = _future_frame(frame, forecast.horizon, forecast.granularity)
            predicted = forecaster.predict(forecast.horizon, future)
        except (UnscoredModelError, ValueError) as exc:
            forecast.status = "FAILED"
            forecast.error_message = str(exc)
            self._repository.save()
            return
        artifact = self._write_forecast_artifact(forecast, predicted)
        points = [
            ForecastPointRow(
                forecast_run_id=forecast.id,
                series_id=series_id,
                date=day,
                p10=None if predicted.p10 is None else predicted.p10[index],
                p50=predicted.p50[index],
                p90=None if predicted.p90 is None else predicted.p90[index],
                actual=None,
            )
            for index, (series_id, day) in enumerate(
                zip(predicted.series_id, predicted.date, strict=True)
            )
        ]
        self._repository.add_points(points)
        forecast.output_uri = str(artifact)
        forecast.status = "SUCCEEDED"
        self._repository.save()

    def _decide(
        self,
        model: ModelVersionRow,
        frame: pa.Table,
        report: EvaluationReport,
        folds: int,
        horizon: int,
    ) -> PromotionDecision:
        """Gate ``report`` against production, or against seasonal naive."""

        model.status = "EVALUATED"
        self._repository.save()
        production = self._repository.production_model()
        production_report = _production_report(production, frame, folds, horizon)
        reference = reference_report(
            frame,
            folds=folds,
            horizon=horizon,
            dataset_version=model.dataset_version,
            production=production_report,
        )
        reference_id = production.id if production is not None else "seasonal_naive"
        decision = gate(
            report,
            reference,
            candidate_id=model.id,
            reference_id=reference_id,
        )
        if _baseline_may_wait(model.model_family, production is None, report, reference, decision):
            return PromotionDecision(
                candidate_id=decision.candidate_id,
                reference_id=decision.reference_id,
                thresholds=decision.thresholds,
                checks=decision.checks,
                status=ModelStatus.PENDING_APPROVAL,
                reason=None,
                created_at=decision.created_at,
                p90_coverage=decision.p90_coverage,
                regressed_categories=decision.regressed_categories,
            )
        return decision

    def _training_status(
        self,
        run: TrainingRunRow,
        status: str,
        *,
        started: bool = False,
        finished: bool = False,
    ) -> None:
        run.status = status
        if started:
            run.started_at = _now()
        if finished:
            run.finished_at = _now()
        self._repository.save()

    def _fail_training(self, run: TrainingRunRow, message: str) -> None:
        run.status = "FAILED"
        run.error_message = message
        run.finished_at = _now()
        self._repository.save()
        return None

    def _write_training_artifact(self, run: TrainingRunRow, report: EvaluationReport) -> Path:
        directory = self._artifact_dir / "training" / run.id
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "evaluation.json"
        destination.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return directory

    def _write_forecast_artifact(self, forecast: ForecastRunRow, predicted: ForecastFrame) -> Path:
        directory = self._artifact_dir / "forecasts" / forecast.id
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "dates": [day.isoformat() for day in predicted.date],
            "p50": list(predicted.p50),
            "series_id": list(predicted.series_id),
        }
        destination = directory / "forecast.json"
        document = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        destination.write_text(document, encoding="utf-8")
        return directory


def _baseline_may_wait(
    family: str,
    no_production: bool,
    report: EvaluationReport,
    reference: EvaluationReport,
    decision: PromotionDecision,
) -> bool:
    """Let seasonal naive wait for a person when it is the reference.

    The gate requires a strictly better WAPE. The first seasonal naive model
    is scored against itself, so an equal WAPE still waits for approval when
    bias, coverage, and segments already pass.
    """

    if family != "seasonal_naive" or not no_production:
        return False
    if decision.status is not ModelStatus.REJECTED:
        return False
    checks = decision.checks
    if not checks.bias_within_limit or checks.coverage_within_limit is False:
        return False
    if not checks.no_critical_segment_regression:
        return False
    return report.global_metrics.wape <= reference.global_metrics.wape


def _production_report(
    production: ModelVersionRow | None,
    frame: pa.Table,
    folds: int,
    horizon: int,
) -> EvaluationReport | None:
    if production is None or production.model_family not in _LOCAL_FAMILIES:
        return None
    forecaster = local_catalog().get(production.model_family)
    return Backtester().evaluate(
        forecaster,
        frame,
        folds,
        horizon,
        dataset_version=production.dataset_version,
    )


def _future_frame(frame: pa.Table, horizon: int, granularity: str) -> pa.Table:
    dates = [value for value in frame.column("date").to_pylist() if isinstance(value, date)]
    if not dates:
        raise ValueError("The dataset has no observation dates to forecast from.")
    origin = max(dates)
    step = timedelta(weeks=1) if granularity == "week" else timedelta(days=1)
    stores = frame.column("store_id").to_pylist()
    skus = frame.column("sku_id").to_pylist()
    series = sorted({(str(store), str(sku)) for store, sku in zip(stores, skus, strict=True)})
    future_stores: list[str] = []
    future_skus: list[str] = []
    future_dates: list[date] = []
    for store_id, sku_id in series:
        for offset in range(1, horizon + 1):
            future_stores.append(store_id)
            future_skus.append(sku_id)
            future_dates.append(origin + step * offset)
    return pa.table(
        {
            "store_id": pa.array(future_stores, type=pa.string()),
            "sku_id": pa.array(future_skus, type=pa.string()),
            "date": pa.array(future_dates, type=pa.date32()),
        }
    )


def _positive_int(value: object, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    if value < 1:
        return default
    return value


def _now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    from uuid import uuid4

    return str(uuid4())
