"""Contract tests for the SQL and DynamoDB metadata adapters."""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from forecastops_api.db import Base
from forecastops_api.documents import as_utc
from forecastops_api.dynamodb import DynamoRepository, MemoryMetadataTable
from forecastops_api.persistence import (
    AIExplanationRow,
    DatasetRow,
    ForecastErrorEvaluationRow,
    ForecastPointRow,
    ForecastRunRow,
    ModelVersionRow,
    MonitoringReportRow,
    PromotionDecisionRow,
    RetrainRequestRow,
    TrainingRunRow,
)
from forecastops_api.repositories import MetadataRepository, Repository


class _Adapter:
    def repository(self) -> MetadataRepository:
        raise NotImplementedError

    def reopen(self) -> MetadataRepository:
        raise NotImplementedError

    def for_tenant(self, tenant_id: str) -> MetadataRepository:
        raise NotImplementedError


class _SqlAdapter(_Adapter):
    def __init__(self, database: Path) -> None:
        engine = create_engine(f"sqlite+pysqlite:///{database}")
        Base.metadata.create_all(engine)
        self._factory = sessionmaker(bind=engine, expire_on_commit=False)
        self._session: Session = self._factory()
        self._repository: MetadataRepository = Repository(self._session)

    def repository(self) -> MetadataRepository:
        return self._repository

    def reopen(self) -> MetadataRepository:
        self._session.commit()
        self._session.close()
        self._session = self._factory()
        self._repository = Repository(self._session)
        return self._repository

    def for_tenant(self, tenant_id: str) -> MetadataRepository:
        return Repository(self._session, tenant_id=tenant_id)


class _DynamoAdapter(_Adapter):
    def __init__(self) -> None:
        self._table = MemoryMetadataTable()
        self._repository: MetadataRepository = DynamoRepository(self._table)

    def repository(self) -> MetadataRepository:
        return self._repository

    def reopen(self) -> MetadataRepository:
        self._repository.save()
        self._repository = DynamoRepository(self._table)
        return self._repository

    def for_tenant(self, tenant_id: str) -> MetadataRepository:
        return DynamoRepository(self._table, tenant_id=tenant_id)


@pytest.fixture(params=["sql", "dynamodb"])
def adapter(request: pytest.FixtureRequest, tmp_path: Path) -> _Adapter:
    if request.param == "sql":
        return _SqlAdapter(tmp_path / "metadata.db")
    return _DynamoAdapter()


def test_dataset_round_trip_list_order_and_next_version(adapter: _Adapter) -> None:
    first = _dataset("d1", name="sales", version="1", created_at=_at(1))
    second = _dataset("d2", name="sales", version="2", created_at=_at(2))
    other = _dataset("d3", name="traffic", version="9", created_at=_at(3))
    repo = adapter.repository()
    repo.add(first)
    repo.add(second)
    repo.add(other)
    first.status = "valid"
    repo.save()

    fresh = adapter.reopen()
    found = fresh.get_dataset("d1")
    assert found is not None
    assert found.status == "valid"
    assert found.date_min == date(2024, 1, 1)
    assert found.quality_report == {"status": "valid"}
    assert as_utc(found.created_at) == _at(1)
    assert [row.id for row in fresh.list_datasets()] == ["d1", "d2", "d3"]
    assert fresh.next_dataset_version("sales") == "3"
    assert fresh.next_dataset_version("traffic") == "10"
    assert fresh.get_dataset("missing") is None


def test_a_tenant_cannot_read_another_tenants_rows(adapter: _Adapter) -> None:
    alpha = adapter.for_tenant("alpha")
    beta = adapter.for_tenant("beta")
    alpha.add(_dataset("alpha-1", name="sales", version="1", created_at=_at(1)))
    beta.add(_dataset("beta-1", name="sales", version="1", created_at=_at(2)))

    assert [row.id for row in alpha.list_datasets()] == ["alpha-1"]
    assert [row.id for row in beta.list_datasets()] == ["beta-1"]
    assert alpha.get_dataset("beta-1") is None
    assert beta.get_dataset("alpha-1") is None
    owned = alpha.get_dataset("alpha-1")
    assert owned is not None
    assert owned.tenant_id == "alpha"
    assert alpha.next_dataset_version("sales") == "2"
    assert beta.next_dataset_version("sales") == "2"


def test_training_run_update_is_visible_to_a_new_adapter(adapter: _Adapter) -> None:
    run = _training_run("run-1")
    repo = adapter.repository()
    repo.add(run)
    run.status = "COMPLETED"
    run.metrics = {"wape": 0.2}
    repo.save()

    fresh = adapter.reopen()
    found = fresh.get_training_run("run-1")
    assert found is not None
    assert found.status == "COMPLETED"
    assert found.metrics == {"wape": 0.2}
    assert [row.id for row in fresh.list_training_runs()] == ["run-1"]


def test_production_model_and_decisions(adapter: _Adapter) -> None:
    older = _model("m1", family="seasonal_naive", version="1", status="APPROVED", created_at=_at(1))
    current = _model(
        "m2",
        family="seasonal_naive",
        version="2",
        status="PRODUCTION",
        created_at=_at(2),
    )
    other = _model("m3", family="naive", version="1", status="PRODUCTION", created_at=_at(3))
    decision = _decision("decision-1", "m2")
    repo = adapter.repository()
    repo.add(older)
    repo.add(current)
    repo.add(other)
    repo.add(decision)

    fresh = adapter.reopen()
    production = fresh.production_model()
    assert production is not None
    assert production.id == "m3"
    assert [row.id for row in fresh.production_for_family("seasonal_naive")] == ["m2"]
    assert fresh.next_model_version("seasonal_naive") == "3"
    assert [row.id for row in fresh.list_decisions("m2")] == ["decision-1"]
    assert fresh.list_decisions("m1") == []


def test_forecast_idempotency_points_and_daily_count(adapter: _Adapter) -> None:
    forecast = _forecast("f1", idempotency_key="client-1", status="SUCCEEDED", created_at=_at(4))
    earlier = _forecast(
        "f0",
        idempotency_key=None,
        status="FAILED",
        created_at=datetime(2024, 5, 1, 1, tzinfo=UTC),
    )
    point = ForecastPointRow(
        forecast_run_id="f1",
        series_id="sku-1",
        date=date(2024, 2, 1),
        p10=1.0,
        p50=2.5,
        p90=4.0,
        actual=None,
    )
    later_point = ForecastPointRow(
        forecast_run_id="f1",
        series_id="sku-1",
        date=date(2024, 2, 2),
        p10=None,
        p50=3.0,
        p90=None,
        actual=3.0,
    )
    repo = adapter.repository()
    repo.add(earlier)
    repo.add(forecast)
    repo.add_points([later_point, point])

    fresh = adapter.reopen()
    found = fresh.get_forecast_by_idempotency_key("client-1")
    assert found is not None
    assert found.id == "f1"
    assert fresh.get_forecast_by_idempotency_key("missing") is None
    listed = fresh.list_points("f1")
    assert [(row.series_id, row.date, row.p50) for row in listed] == [
        ("sku-1", date(2024, 2, 1), 2.5),
        ("sku-1", date(2024, 2, 2), 3.0),
    ]
    assert fresh.count_forecasts_created_on(date(2024, 5, 2)) == 1
    assert fresh.count_forecasts_created_on(date(2024, 5, 1)) == 1
    assert [row.id for row in fresh.list_forecasts()] == ["f0", "f1"]
    succeeded = fresh.latest_succeeded_forecast()
    assert succeeded is not None
    assert succeeded.id == "f1"


def test_explanations_errors_monitoring_and_retrain(adapter: _Adapter) -> None:
    stale = _explanation("e1", created_at=_at(1), prompt_version="v1")
    current = _explanation("e2", created_at=_at(2), prompt_version="v2")
    invalid = _explanation("e3", created_at=_at(3), prompt_version="v2", status="invalid")
    error = ForecastErrorEvaluationRow(
        id="err-1",
        forecast_run_id="f1",
        compared_points=4,
        wape=0.18,
        bias=0.01,
        created_at=_at(2),
    )
    report = _monitoring_report("report-1")
    pending = _retrain("req-1", status="PENDING", requested_at=_at(2))
    confirmed = _retrain("req-2", status="CONFIRMED", requested_at=_at(1))
    repo = adapter.repository()
    for row in (stale, current, invalid, error, report, pending, confirmed):
        repo.add(row)

    fresh = adapter.reopen()
    found = fresh.find_explanation("f1", "scope", "v2")
    assert found is not None
    assert found.id == "e2"
    latest = fresh.latest_explanation("f1", "scope")
    assert latest is not None
    assert latest.id == "e2"
    assert fresh.count_explanation_calls_on(_at(1).date()) == 3
    usage = fresh.explanation_usage_on(_at(1).date())
    assert usage.calls == 3
    assert usage.input_tokens == 30
    assert usage.output_tokens == 60
    scored = fresh.latest_forecast_error()
    assert scored is not None
    assert scored.wape == 0.18
    stored = fresh.latest_monitoring_report()
    assert stored is not None
    assert stored.status == "WARNING"
    assert stored.as_of == date(2024, 3, 1)
    open_request = fresh.pending_retrain_request()
    assert open_request is not None
    assert open_request.id == "req-1"
    assert [row.id for row in fresh.list_retrain_requests()] == ["req-2", "req-1"]


def _at(hour: int) -> datetime:
    return datetime(2024, 5, 2, hour, tzinfo=UTC)


def _dataset(row_id: str, *, name: str, version: str, created_at: datetime) -> DatasetRow:
    return DatasetRow(
        id=row_id,
        name=name,
        version=version,
        source="synthetic",
        status="registered",
        uri="/tmp/data",
        schema_version="1",
        row_count=10,
        date_min=date(2024, 1, 1),
        date_max=date(2024, 1, 2),
        quality_report={"status": "valid"},
        created_at=created_at,
    )


def _training_run(row_id: str) -> TrainingRunRow:
    return TrainingRunRow(
        id=row_id,
        dataset_id="d1",
        dataset_version="1",
        model_family="seasonal_naive",
        configuration={"folds": 3},
        status="QUEUED",
        started_at=None,
        finished_at=None,
        artifact_uri="",
        metrics=None,
        git_sha="abc",
        pipeline_execution_arn="",
        error_message=None,
        created_at=_at(1),
    )


def _model(
    row_id: str,
    *,
    family: str,
    version: str,
    status: str,
    created_at: datetime,
) -> ModelVersionRow:
    return ModelVersionRow(
        id=row_id,
        model_family=family,
        version=version,
        training_run_id="run-1",
        dataset_id="d1",
        dataset_version="1",
        registry_arn="",
        registry_status="",
        status=status,
        metrics={"wape": 0.2},
        approved_at=None,
        approved_by=None,
        rejected_by=None,
        rejection_reason=None,
        created_at=created_at,
    )


def _decision(row_id: str, candidate_id: str) -> PromotionDecisionRow:
    return PromotionDecisionRow(
        id=row_id,
        candidate_id=candidate_id,
        reference_id="reference",
        thresholds={"wape": 0.25},
        checks={"wape": True},
        status="APPROVED",
        reason=None,
        p90_coverage=0.9,
        regressed_categories=[],
        actor_id="analyst-1",
        created_at=_at(2),
    )


def _forecast(
    row_id: str,
    *,
    idempotency_key: str | None,
    status: str,
    created_at: datetime,
) -> ForecastRunRow:
    return ForecastRunRow(
        id=row_id,
        model_version_id="m1",
        dataset_id="d1",
        dataset_version="1",
        horizon=7,
        granularity="day",
        status=status,
        output_uri="",
        error_message=None,
        idempotency_key=idempotency_key,
        created_at=created_at,
    )


def _explanation(
    row_id: str,
    *,
    created_at: datetime,
    prompt_version: str,
    status: str = "valid",
) -> AIExplanationRow:
    return AIExplanationRow(
        id=row_id,
        forecast_run_id="f1",
        scope_key="scope",
        scope={"sku": "sku-1"},
        model_id="local",
        prompt_version=prompt_version,
        input_tokens=10,
        output_tokens=20,
        latency_ms=5,
        explanation={"text": "steady"} if status == "valid" else None,
        package={"total": 1},
        status=status,
        created_at=created_at,
    )


def _monitoring_report(row_id: str) -> MonitoringReportRow:
    return MonitoringReportRow(
        id=row_id,
        model_version_id="m1",
        dataset_id="d1",
        status="WARNING",
        as_of=date(2024, 3, 1),
        date_max=date(2024, 2, 28),
        baseline_start=date(2024, 1, 1),
        baseline_end=date(2024, 1, 31),
        recent_start=date(2024, 2, 1),
        recent_end=date(2024, 2, 28),
        metrics={"psi": 0.12},
        retrain_request_id=None,
        created_at=_at(3),
    )


def _retrain(row_id: str, *, status: str, requested_at: datetime) -> RetrainRequestRow:
    return RetrainRequestRow(
        id=row_id,
        dataset_id="d1",
        model_version_id="m1",
        model_family="seasonal_naive",
        status=status,
        training_run_id=None,
        requested_at=requested_at,
        confirmed_at=None,
        confirmed_by=None,
        created_at=requested_at + timedelta(seconds=1),
    )
