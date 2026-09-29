"""Promotion gate, human approval, and the seasonal-naive reference."""

from datetime import UTC, date, datetime, timedelta

import pyarrow as pa
import pytest

from forecastops_ml.evaluation import EvaluationReport, MetricSummary, SliceMetrics
from forecastops_ml.promotion import (
    DEFAULT_MAX_BIAS,
    DEFAULT_MAX_SEGMENT_WAPE_REGRESSION,
    DEFAULT_MIN_P90_COVERAGE,
    HUMAN_REJECTION,
    QUALITY_GATE,
    ModelStatus,
    PromotionError,
    PromotionService,
    RegistryStatus,
    empirical_p90_coverage,
    reference_report,
)
from forecastops_ml.registry import MemoryModelRegistry

DATASET = "retail-demand-v1"
NOW = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)


def test_worse_wape_is_rejected_without_an_approver() -> None:
    service = PromotionService(clock=lambda: NOW)
    _evaluated(service, "candidate", wape=0.20, bias=0.0)
    decision = service.gate("candidate", _report(wape=0.15), reference_id="production-1")

    assert decision.status is ModelStatus.REJECTED
    assert decision.reason == QUALITY_GATE
    assert decision.checks.wape_improved is False
    assert decision.created_at == NOW
    assert decision.thresholds.max_bias == DEFAULT_MAX_BIAS
    model = service.get("candidate")
    assert model.status is ModelStatus.REJECTED
    assert model.approved_by is None
    with pytest.raises(PromotionError, match="actor id"):
        service.approve("candidate", " ")


def test_excess_bias_is_rejected() -> None:
    service = _service()
    _evaluated(service, "candidate", wape=0.10, bias=0.05)
    decision = service.gate("candidate", _report(wape=0.15), reference_id="production-1")

    assert decision.status is ModelStatus.REJECTED
    assert decision.reason == QUALITY_GATE
    assert decision.checks.bias_within_limit is False
    assert decision.checks.wape_improved is True


def test_coverage_failure_is_rejected() -> None:
    service = _service()
    _evaluated(service, "candidate", wape=0.10, bias=0.0, with_p90=True)
    decision = service.gate(
        "candidate",
        _report(wape=0.15, with_p90=True),
        reference_id="production-1",
        p90_coverage=0.84,
    )

    assert decision.status is ModelStatus.REJECTED
    assert decision.checks.coverage_within_limit is False
    assert decision.p90_coverage == 0.84
    assert decision.thresholds.min_p90_coverage == DEFAULT_MIN_P90_COVERAGE


def test_segment_regression_names_the_category() -> None:
    service = _service()
    _evaluated(
        service,
        "candidate",
        wape=0.10,
        bias=0.0,
        categories={"grocery": 0.10, "electronics": 0.18},
    )
    decision = service.gate(
        "candidate",
        _report(wape=0.15, categories={"grocery": 0.12, "electronics": 0.15}),
        reference_id="production-1",
    )

    assert decision.status is ModelStatus.REJECTED
    assert decision.reason == QUALITY_GATE
    assert decision.checks.wape_improved is True
    assert [item.category_id for item in decision.regressed_categories] == ["electronics"]
    assert decision.regressed_categories[0].candidate_wape == 0.18
    assert DEFAULT_MAX_SEGMENT_WAPE_REGRESSION == 0.02


def test_missing_quantiles_skip_coverage_and_wait_for_approval() -> None:
    service = _service()
    trained = service.register_trained(
        model_id="gbm-1",
        model_family="gradient_boosting",
        version="1",
        training_run_id="run-1",
        dataset_version=DATASET,
    )
    assert trained.status is ModelStatus.TRAINED
    assert trained.approved_at is None

    evaluated = service.record_evaluation("gbm-1", _report(wape=0.10, bias=0.01))
    assert evaluated.status is ModelStatus.EVALUATED
    assert evaluated.approved_at is None

    decision = service.gate("gbm-1", _report(wape=0.15, bias=0.0), reference_id="seasonal-naive")

    assert decision.status is ModelStatus.PENDING_APPROVAL
    assert decision.reason is None
    assert decision.p90_coverage is None
    assert decision.checks.coverage_within_limit is None
    assert service.get("gbm-1").status is ModelStatus.PENDING_APPROVAL
    assert service.get("gbm-1").status is not ModelStatus.APPROVED
    assert service.decisions() == (decision,)


def test_happy_path_stays_pending_until_a_person_approves() -> None:
    service = _service()
    _evaluated(
        service,
        "candidate",
        wape=0.10,
        bias=0.01,
        with_p90=True,
        categories={"grocery": 0.11},
    )
    decision = service.gate(
        "candidate",
        _report(wape=0.15, with_p90=True, categories={"grocery": 0.12}),
        reference_id="production-1",
        p90_coverage=0.85,
    )

    assert decision.status is ModelStatus.PENDING_APPROVAL
    assert decision.checks.passed() is True
    assert service.get("candidate").status is not ModelStatus.APPROVED

    approved = service.approve("candidate", "analyst-7")
    assert approved.status is ModelStatus.APPROVED
    assert approved.approved_by == "analyst-7"
    assert approved.approved_at == NOW


def test_approve_and_reject_require_an_actor_id() -> None:
    pending = _pending()

    with pytest.raises(PromotionError, match="actor id"):
        pending.approve("candidate", "")
    with pytest.raises(PromotionError, match="actor id"):
        pending.reject("candidate", "", HUMAN_REJECTION)

    rejected = pending.reject("candidate", "analyst-7", HUMAN_REJECTION)
    assert rejected.status is ModelStatus.REJECTED
    assert rejected.rejection_reason == HUMAN_REJECTION
    assert rejected.rejected_by == "analyst-7"


def test_promote_demotes_the_previous_production_model() -> None:
    service = _service()
    current = _pending(service=service, model_id="current", family="gradient_boosting")
    service.approve("current", "analyst-7")
    service.promote("current", "analyst-7")

    nxt = _pending(service=service, model_id="next", family="gradient_boosting")
    service.approve("next", "analyst-8")
    _pending(service=service, model_id="seasonal", family="seasonal_naive")
    service.approve("seasonal", "analyst-8")
    service.promote("seasonal", "analyst-8")

    promoted = nxt.promote("next", "analyst-8")

    assert promoted.status is ModelStatus.PRODUCTION
    assert current.get("current").status is ModelStatus.APPROVED
    assert nxt.get("seasonal").status is ModelStatus.PRODUCTION


def test_only_approved_models_enter_production() -> None:
    service = _pending()
    with pytest.raises(PromotionError, match="production"):
        service.promote("candidate", "analyst-7")


def test_human_reject_does_not_accept_another_reason() -> None:
    service = _pending()
    with pytest.raises(PromotionError, match="human"):
        service.reject("candidate", "analyst-7", QUALITY_GATE)


def test_quantile_model_must_supply_coverage() -> None:
    service = _service()
    _evaluated(service, "candidate", wape=0.10, with_p90=True)
    with pytest.raises(ValueError, match="empirical P90 coverage"):
        service.gate("candidate", _report(wape=0.15, with_p90=True), reference_id="production-1")


def test_gate_registers_a_pending_model_as_pending_manual_approval() -> None:
    registry = MemoryModelRegistry()
    service = PromotionService(clock=lambda: NOW, registry=registry)
    _evaluated(service, "candidate", wape=0.10)
    service.gate("candidate", _report(wape=0.20), reference_id="production-1")

    model = service.get("candidate")
    assert model.status is ModelStatus.PENDING_APPROVAL
    assert model.registry_status is RegistryStatus.PENDING_MANUAL_APPROVAL
    entry = registry.get(model.registry_arn)
    assert entry.status is RegistryStatus.PENDING_MANUAL_APPROVAL
    assert entry.actor_id is None
    assert model.version == entry.version


def test_gate_records_a_rejected_candidate_in_the_registry() -> None:
    registry = MemoryModelRegistry()
    service = PromotionService(clock=lambda: NOW, registry=registry)
    _evaluated(service, "candidate", wape=0.20)
    service.gate("candidate", _report(wape=0.10), reference_id="production-1")

    model = service.get("candidate")
    assert model.status is ModelStatus.REJECTED
    assert model.registry_status is RegistryStatus.REJECTED
    assert model.approved_by is None
    assert registry.get(model.registry_arn).status is RegistryStatus.REJECTED


def test_approve_updates_the_internal_row_and_the_registry() -> None:
    registry = MemoryModelRegistry()
    service = PromotionService(clock=lambda: NOW, registry=registry)
    _evaluated(service, "candidate", wape=0.10)
    service.gate("candidate", _report(wape=0.20), reference_id="production-1")

    approved = service.approve("candidate", "analyst-7")

    assert approved.status is ModelStatus.APPROVED
    assert approved.registry_status is RegistryStatus.APPROVED
    assert approved.approved_by == "analyst-7"
    entry = registry.get(approved.registry_arn)
    assert entry.status is RegistryStatus.APPROVED
    assert entry.actor_id == "analyst-7"
    assert approved.version == entry.version


def test_approve_without_a_registry_updates_only_the_internal_row() -> None:
    approved = _pending().approve("candidate", "analyst-7")

    assert approved.status is ModelStatus.APPROVED
    assert approved.approved_by == "analyst-7"
    assert approved.registry_arn == ""
    assert approved.registry_status is None


def test_reject_updates_the_registry_and_stores_the_actor() -> None:
    registry = MemoryModelRegistry()
    service = PromotionService(clock=lambda: NOW, registry=registry)
    _evaluated(service, "candidate", wape=0.10)
    service.gate("candidate", _report(wape=0.20), reference_id="production-1")

    rejected = service.reject("candidate", "analyst-7", HUMAN_REJECTION)

    assert rejected.status is ModelStatus.REJECTED
    assert rejected.rejected_by == "analyst-7"
    assert rejected.registry_status is RegistryStatus.REJECTED
    assert registry.get(rejected.registry_arn).actor_id == "analyst-7"


def test_promote_leaves_the_registry_status_approved() -> None:
    registry = MemoryModelRegistry()
    service = PromotionService(clock=lambda: NOW, registry=registry)
    _evaluated(service, "candidate", wape=0.10)
    service.gate("candidate", _report(wape=0.20), reference_id="production-1")
    service.approve("candidate", "analyst-7")

    promoted = service.promote("candidate", "analyst-7")

    assert promoted.status is ModelStatus.PRODUCTION
    assert promoted.registry_status is RegistryStatus.APPROVED
    assert registry.get(promoted.registry_arn).status is RegistryStatus.APPROVED


def test_empirical_p90_coverage_is_the_share_at_or_below_p90() -> None:
    assert empirical_p90_coverage([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 3.0, 5.0]) == 0.75


def test_absent_production_uses_seasonal_naive_on_the_same_dataset() -> None:
    frame = _daily_frame()
    produced = reference_report(
        frame,
        folds=3,
        horizon=1,
        dataset_version=DATASET,
        production=None,
    )
    assert produced.model_family == "seasonal_naive"
    assert produced.dataset_version == DATASET
    assert produced.global_metrics.pinball_loss_p90 is None

    existing = _report(wape=0.2, dataset_version=DATASET)
    assert (
        reference_report(
            frame,
            folds=3,
            horizon=1,
            dataset_version=DATASET,
            production=existing,
        )
        is existing
    )

    service = _service()
    _evaluated(service, "candidate", wape=produced.global_metrics.wape - 0.01, bias=0.0)
    decision = service.gate("candidate", produced, reference_id="seasonal_naive")
    assert decision.status is ModelStatus.PENDING_APPROVAL
    assert decision.reference_id == "seasonal_naive"


def _service() -> PromotionService:
    return PromotionService(clock=lambda: NOW)


def _pending(
    service: PromotionService | None = None,
    model_id: str = "candidate",
    family: str = "gradient_boosting",
) -> PromotionService:
    store = service if service is not None else _service()
    _evaluated(store, model_id, wape=0.10, bias=0.0, family=family)
    store.gate(model_id, _report(wape=0.20), reference_id="production-1")
    return store


def _evaluated(
    service: PromotionService,
    model_id: str,
    *,
    wape: float,
    bias: float = 0.0,
    with_p90: bool = False,
    categories: dict[str, float] | None = None,
    family: str = "gradient_boosting",
) -> None:
    service.register_trained(
        model_id=model_id,
        model_family=family,
        version="1",
        training_run_id=f"run-{model_id}",
        dataset_version=DATASET,
    )
    service.record_evaluation(
        model_id,
        _report(wape=wape, bias=bias, with_p90=with_p90, categories=categories),
    )


def _report(
    *,
    wape: float,
    bias: float = 0.0,
    with_p90: bool = False,
    categories: dict[str, float] | None = None,
    dataset_version: str | None = DATASET,
) -> EvaluationReport:
    summary = _metrics(wape, bias, with_p90=with_p90)
    slices = categories or {"grocery": wape}
    return EvaluationReport(
        model_family="candidate",
        fold_count=3,
        horizon=1,
        dataset_version=dataset_version,
        global_metrics=summary,
        by_category=tuple(
            SliceMetrics(key=key, metrics=_metrics(value, bias, with_p90=with_p90))
            for key, value in sorted(slices.items())
        ),
        by_store=(),
        by_horizon_step=(),
        by_demand_quartile=(),
        runtime_ms=1,
        wape_by_horizon=((1, wape),),
        skipped_series=(),
    )


def _metrics(wape: float, bias: float, *, with_p90: bool) -> MetricSummary:
    return MetricSummary(
        row_count=4,
        mae=1.0,
        rmse=1.0,
        wape=wape,
        smape=0.1,
        bias=bias,
        pinball_loss_p10=0.2 if with_p90 else None,
        pinball_loss_p90=0.3 if with_p90 else None,
    )


def _daily_frame() -> pa.Table:
    start = date(2026, 1, 5)
    days = [start + timedelta(days=offset) for offset in range(28)]
    return pa.table(
        {
            "date": pa.array(days, type=pa.date32()),
            "store_id": pa.array(["store-01"] * len(days), type=pa.string()),
            "sku_id": pa.array(["sku-1"] * len(days), type=pa.string()),
            "category_id": pa.array(["grocery"] * len(days), type=pa.string()),
            "units_sold": pa.array(
                [10 + (offset % 7) for offset in range(len(days))],
                type=pa.int64(),
            ),
        }
    )
