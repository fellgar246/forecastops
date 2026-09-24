"""In-memory promotion service.

``gate``, ``approve``, and ``reject`` do not depend on HTTP. A later API
calls them. Training success registers a model as ``TRAINED`` and does not
approve it.
"""

from collections.abc import Callable
from datetime import UTC, datetime

from forecastops_ml.evaluation.backtest import EvaluationReport
from forecastops_ml.promotion.gate import (
    HUMAN_REJECTION,
    PromotionDecision,
    gate,
)
from forecastops_ml.promotion.models import ModelStatus, ModelVersion
from forecastops_ml.promotion.thresholds import PromotionThresholds

Clock = Callable[[], datetime]


class PromotionError(ValueError):
    """A promotion transition is not allowed from the current status."""


class PromotionService:
    """Store model versions and the decisions that change their status."""

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))
        self._models: dict[str, ModelVersion] = {}
        self._reports: dict[str, EvaluationReport] = {}
        self._decisions: list[PromotionDecision] = []

    def register_trained(
        self,
        *,
        model_id: str,
        model_family: str,
        version: str,
        training_run_id: str,
        dataset_version: str | None = None,
    ) -> ModelVersion:
        """Record a finished training job as ``TRAINED``.

        The returned version is not approved. Approval happens only through
        :meth:`approve` after the gate leaves the model pending.
        """

        if model_id in self._models:
            raise PromotionError(f"Model {model_id} is already registered.")
        model = ModelVersion(
            id=model_id,
            model_family=model_family,
            version=version,
            training_run_id=training_run_id,
            status=ModelStatus.TRAINED,
            metrics={},
            registry_arn="",
            approved_at=None,
            created_at=self._now(),
            dataset_version=dataset_version,
        )
        self._models[model_id] = model
        return model

    def record_evaluation(self, model_id: str, report: EvaluationReport) -> ModelVersion:
        """Store ``report`` and move a trained model to ``EVALUATED``.

        Scoring does not approve the model.
        """

        model = self._require(model_id)
        if model.status is not ModelStatus.TRAINED:
            raise PromotionError(f"Model {model_id} cannot be evaluated from {model.status.value}.")
        if (
            model.dataset_version is not None
            and report.dataset_version is not None
            and model.dataset_version != report.dataset_version
        ):
            raise PromotionError("Evaluation dataset version does not match the model.")
        updated = _replace(
            model,
            status=ModelStatus.EVALUATED,
            metrics=dict(report.global_metrics.to_dict()),
        )
        self._models[model_id] = updated
        self._reports[model_id] = report
        return updated

    def gate(
        self,
        candidate_id: str,
        reference: EvaluationReport,
        *,
        reference_id: str,
        p90_coverage: float | None = None,
        thresholds: PromotionThresholds | None = None,
    ) -> PromotionDecision:
        """Run the quality gate for an evaluated candidate.

        A passing candidate becomes ``PENDING_APPROVAL``. A failing candidate
        becomes ``REJECTED`` with reason ``quality_gate``. Neither outcome
        approves the model, and this method does not take an actor id.
        """

        model = self._require(candidate_id)
        if model.status is not ModelStatus.EVALUATED:
            raise PromotionError(f"Model {candidate_id} cannot be gated from {model.status.value}.")
        try:
            report = self._reports[candidate_id]
        except KeyError as exc:
            raise PromotionError(f"Model {candidate_id} has no evaluation report.") from exc
        decision = gate(
            report,
            reference,
            candidate_id=candidate_id,
            reference_id=reference_id,
            p90_coverage=p90_coverage,
            thresholds=thresholds,
            now=self._now(),
        )
        metrics = dict(model.metrics)
        metrics["p90_coverage"] = decision.p90_coverage
        self._models[candidate_id] = _replace(
            model,
            status=decision.status,
            metrics=metrics,
            rejection_reason=decision.reason,
        )
        self._decisions.append(decision)
        return decision

    def approve(self, model_id: str, actor_id: str) -> ModelVersion:
        """Move ``PENDING_APPROVAL`` to ``APPROVED`` for ``actor_id``."""

        actor = _actor(actor_id)
        model = self._require(model_id)
        if model.status is not ModelStatus.PENDING_APPROVAL:
            raise PromotionError(f"Model {model_id} cannot be approved from {model.status.value}.")
        updated = _replace(
            model,
            status=ModelStatus.APPROVED,
            approved_at=self._now(),
            approved_by=actor,
        )
        self._models[model_id] = updated
        return updated

    def reject(self, model_id: str, actor_id: str, reason: str) -> ModelVersion:
        """Move ``PENDING_APPROVAL`` to ``REJECTED`` for ``actor_id``.

        ``reason`` must be ``human``. The quality gate uses a different
        reason and does not call this method.
        """

        actor = _actor(actor_id)
        if reason != HUMAN_REJECTION:
            raise PromotionError("Human rejection uses reason 'human'.")
        model = self._require(model_id)
        if model.status is not ModelStatus.PENDING_APPROVAL:
            raise PromotionError(f"Model {model_id} cannot be rejected from {model.status.value}.")
        updated = _replace(
            model,
            status=ModelStatus.REJECTED,
            rejected_by=actor,
            rejection_reason=HUMAN_REJECTION,
        )
        self._models[model_id] = updated
        return updated

    def promote(self, model_id: str, actor_id: str) -> ModelVersion:
        """Move ``APPROVED`` to ``PRODUCTION`` for ``actor_id``.

        The previous production model of the same family returns to
        ``APPROVED``. No other status can enter production.
        """

        _actor(actor_id)
        model = self._require(model_id)
        if model.status is not ModelStatus.APPROVED:
            raise PromotionError(
                f"Model {model_id} cannot enter production from {model.status.value}."
            )
        for other in list(self._models.values()):
            if other.id == model_id:
                continue
            if other.model_family != model.model_family:
                continue
            if other.status is ModelStatus.PRODUCTION:
                self._models[other.id] = _replace(other, status=ModelStatus.APPROVED)
        updated = _replace(model, status=ModelStatus.PRODUCTION)
        self._models[model_id] = updated
        return updated

    def get(self, model_id: str) -> ModelVersion:
        """Return the stored model version."""

        return self._require(model_id)

    def decisions(self) -> tuple[PromotionDecision, ...]:
        """Return gate decisions in the order they were recorded."""

        return tuple(self._decisions)

    def _require(self, model_id: str) -> ModelVersion:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise PromotionError(f"Model {model_id} is not registered.") from exc

    def _now(self) -> datetime:
        current = self._clock()
        if current.tzinfo is None:
            raise PromotionError("Promotion timestamps must include a timezone.")
        return current


def _actor(actor_id: str) -> str:
    if not isinstance(actor_id, str) or actor_id.strip() == "":
        raise PromotionError("An actor id is required.")
    return actor_id


def _replace(model: ModelVersion, **changes: object) -> ModelVersion:
    values: dict[str, object] = {
        "id": model.id,
        "model_family": model.model_family,
        "version": model.version,
        "training_run_id": model.training_run_id,
        "status": model.status,
        "metrics": dict(model.metrics),
        "registry_arn": model.registry_arn,
        "approved_at": model.approved_at,
        "created_at": model.created_at,
        "dataset_version": model.dataset_version,
        "approved_by": model.approved_by,
        "rejected_by": model.rejected_by,
        "rejection_reason": model.rejection_reason,
    }
    values.update(changes)
    return ModelVersion(
        id=str(values["id"]),
        model_family=str(values["model_family"]),
        version=str(values["version"]),
        training_run_id=str(values["training_run_id"]),
        status=_status(values["status"]),
        metrics=_metrics(values["metrics"]),
        registry_arn=str(values["registry_arn"]),
        approved_at=_optional_datetime(values["approved_at"]),
        created_at=_datetime(values["created_at"]),
        dataset_version=_optional_str(values["dataset_version"]),
        approved_by=_optional_str(values["approved_by"]),
        rejected_by=_optional_str(values["rejected_by"]),
        rejection_reason=_optional_str(values["rejection_reason"]),
    )


def _status(value: object) -> ModelStatus:
    if not isinstance(value, ModelStatus):
        raise PromotionError("Model status must be a ModelStatus value.")
    return value


def _metrics(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PromotionError("Model metrics must be a mapping.")
    return dict(value)


def _datetime(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise PromotionError("Promotion timestamps must be datetimes.")
    return value


def _optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    return _datetime(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PromotionError("Expected a string.")
    return value
