"""In-process training jobs.

Jobs call the local forecasting library and write artifacts through the store.
They do not import a cloud SDK. Forecast runs are executed by the forecast worker.
"""

import json
import time
from datetime import UTC, datetime

import pyarrow as pa
import structlog

from forecastops_api.artifacts import ArtifactStore
from forecastops_api.observability import record_metric, record_model_metrics
from forecastops_api.persistence import (
    ModelVersionRow,
    PromotionDecisionRow,
    TrainingRunRow,
)
from forecastops_api.repositories import MetadataRepository
from forecastops_ml.baselines.catalog import local_catalog
from forecastops_ml.evaluation.backtest import (
    Backtester,
    EvaluationReport,
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

    def __init__(self, repository: MetadataRepository, artifacts: ArtifactStore) -> None:
        self._repository = repository
        self._artifacts = artifacts

    def train(self, run: TrainingRunRow, frame: pa.Table) -> ModelVersionRow | None:
        """Fit, score, and register ``run``. Return the model, or none on failure."""

        started = time.perf_counter()
        record_metric("training_job_count", 1)
        self._training_status(run, "PREPROCESSING", started=True)
        if run.model_family not in _LOCAL_FAMILIES:
            message = f"{run.model_family} is not available in the local profile."
            self._fail_training(run, message)
            self._observe_training(run, started, failed=True)
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
            self._observe_training(run, started, failed=True)
            return None

        self._training_status(run, "REGISTERING")
        run.artifact_uri = self._write_training_artifact(run, report)
        run.metrics = report.to_dict()
        model = ModelVersionRow(
            id=_new_id(),
            model_family=run.model_family,
            version=self._repository.next_model_version(run.model_family),
            training_run_id=run.id,
            dataset_id=run.dataset_id,
            dataset_version=run.dataset_version,
            registry_arn="",
            registry_status="",
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
        self._observe_training(
            run,
            started,
            failed=False,
            model=model,
            report=report,
            horizon=horizon,
        )
        return model

    def _observe_training(
        self,
        run: TrainingRunRow,
        started: float,
        *,
        failed: bool,
        model: ModelVersionRow | None = None,
        report: EvaluationReport | None = None,
        horizon: int | None = None,
    ) -> None:
        duration = time.perf_counter() - started
        record_metric("training_duration_seconds", duration)
        if failed:
            record_metric("training_failures", 1)
        if model is not None and report is not None and horizon is not None:
            record_model_metrics(
                model_version=model.version,
                forecast_horizon=horizon,
                report=report,
                p90_coverage=_coverage(model),
            )
        structlog.get_logger().info(
            "training.finished",
            training_run_id=run.id,
            dataset_id=run.dataset_id,
            model_version_id=None if model is None else model.id,
            status=run.status,
            artifact_uri=run.artifact_uri,
            duration_seconds=round(duration, 3),
        )

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

    def _write_training_artifact(self, run: TrainingRunRow, report: EvaluationReport) -> str:
        document = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
        return self._artifacts.put(
            "training",
            f"{run.id}/evaluation.json",
            document.encode("utf-8"),
        )


def _coverage(model: ModelVersionRow) -> float | None:
    metrics = model.metrics
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("p90_coverage")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


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
