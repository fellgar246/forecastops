"""Domain tables stored in PostgreSQL.

Services reach these rows only through repositories.
"""

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from forecastops_api.db import Base
from forecastops_api.tenancy import LOCAL_TENANT


class TenantScoped:
    """Every domain row belongs to one tenant.

    Existing local rows default to ``local``. Queries filter on this column.
    """

    tenant_id: Mapped[str] = mapped_column(
        String(64),
        default=LOCAL_TENANT,
        server_default=text("'local'"),
        index=True,
    )


class DatasetRow(TenantScoped, Base):
    """One immutable dataset snapshot."""

    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    uri: Mapped[str] = mapped_column(Text)
    schema_version: Mapped[str] = mapped_column(String(64))
    row_count: Mapped[int] = mapped_column(Integer)
    date_min: Mapped[date | None] = mapped_column(Date, nullable=True)
    date_max: Mapped[date | None] = mapped_column(Date, nullable=True)
    quality_report: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class TrainingRunRow(TenantScoped, Base):
    """One training job and the metrics it produced."""

    __tablename__ = "training_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    dataset_version: Mapped[str] = mapped_column(String(64))
    model_family: Mapped[str] = mapped_column(String(64))
    configuration: Mapped[dict[str, object]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    artifact_uri: Mapped[str] = mapped_column(Text, default="")
    metrics: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    git_sha: Mapped[str] = mapped_column(String(64), default="")
    pipeline_execution_arn: Mapped[str] = mapped_column(Text, default="")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ModelVersionRow(TenantScoped, Base):
    """One registered model version and its promotion status."""

    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_family: Mapped[str] = mapped_column(String(64))
    version: Mapped[str] = mapped_column(String(64))
    training_run_id: Mapped[str] = mapped_column(ForeignKey("training_runs.id"))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    dataset_version: Mapped[str] = mapped_column(String(64))
    registry_arn: Mapped[str] = mapped_column(Text, default="")
    registry_status: Mapped[str] = mapped_column(String(32), default="", server_default="")
    status: Mapped[str] = mapped_column(String(32))
    metrics: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    approved_by: Mapped[str | None] = mapped_column(String(200), nullable=True)
    rejected_by: Mapped[str | None] = mapped_column(String(200), nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class PromotionDecisionRow(TenantScoped, Base):
    """One quality-gate or human promotion decision."""

    __tablename__ = "promotion_decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    candidate_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    reference_id: Mapped[str] = mapped_column(String(64))
    thresholds: Mapped[dict[str, object]] = mapped_column(JSON)
    checks: Mapped[dict[str, object]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    p90_coverage: Mapped[float | None] = mapped_column(Float, nullable=True)
    regressed_categories: Mapped[list[dict[str, object]]] = mapped_column(JSON)
    actor_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ForecastRunRow(TenantScoped, Base):
    """One batch forecast job."""

    __tablename__ = "forecast_runs"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_forecast_runs_tenant_idempotency",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    dataset_version: Mapped[str] = mapped_column(String(64))
    horizon: Mapped[int] = mapped_column(Integer)
    granularity: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(32))
    output_uri: Mapped[str] = mapped_column(Text, default="")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AIExplanationRow(TenantScoped, Base):
    """One explanation attempt for a forecast scope.

    ``status`` is ``valid`` only after the draft passes the deterministic checks.
    Invalid attempts are kept so the daily call ceiling still counts them.
    """

    __tablename__ = "ai_explanations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    forecast_run_id: Mapped[str] = mapped_column(ForeignKey("forecast_runs.id"), index=True)
    scope_key: Mapped[str] = mapped_column(String(500))
    scope: Mapped[dict[str, str]] = mapped_column(JSON)
    model_id: Mapped[str] = mapped_column(String(200))
    prompt_version: Mapped[str] = mapped_column(String(32))
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[int] = mapped_column(Integer)
    explanation: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    package: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class DataRefreshMarkerRow(TenantScoped, Base):
    """One record that a scheduled refresh touched a dataset."""

    __tablename__ = "data_refresh_markers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    refreshed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ForecastErrorEvaluationRow(TenantScoped, Base):
    """Forecast error for points whose actuals have arrived."""

    __tablename__ = "forecast_error_evaluations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    forecast_run_id: Mapped[str] = mapped_column(ForeignKey("forecast_runs.id"), index=True)
    compared_points: Mapped[int] = mapped_column(Integer)
    wape: Mapped[float | None] = mapped_column(Float, nullable=True)
    bias: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class RetrainRequestRow(TenantScoped, Base):
    """A request to train again. Training starts only after a person confirms it."""

    __tablename__ = "retrain_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    model_family: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), index=True)
    training_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("training_runs.id"),
        nullable=True,
    )
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    confirmed_by: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class MonitoringReportRow(TenantScoped, Base):
    """One comparison of a recent window with the training baseline.

    ``status`` is ``HEALTHY``, ``WARNING``, or ``RETRAIN_RECOMMENDED``.
    A recommendation stores ``retrain_request_id`` and does not start training.
    """

    __tablename__ = "monitoring_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    status: Mapped[str] = mapped_column(String(32))
    as_of: Mapped[date] = mapped_column(Date)
    date_max: Mapped[date] = mapped_column(Date)
    baseline_start: Mapped[date] = mapped_column(Date)
    baseline_end: Mapped[date] = mapped_column(Date)
    recent_start: Mapped[date] = mapped_column(Date)
    recent_end: Mapped[date] = mapped_column(Date)
    metrics: Mapped[dict[str, object]] = mapped_column(JSON)
    retrain_request_id: Mapped[str | None] = mapped_column(
        ForeignKey("retrain_requests.id"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class ForecastPointRow(TenantScoped, Base):
    """One quantile forecast for a series on a date."""

    __tablename__ = "forecast_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    forecast_run_id: Mapped[str] = mapped_column(ForeignKey("forecast_runs.id"), index=True)
    series_id: Mapped[str] = mapped_column(String(200))
    date: Mapped[date] = mapped_column(Date)
    p10: Mapped[float | None] = mapped_column(Float, nullable=True)
    p50: Mapped[float] = mapped_column(Float)
    p90: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual: Mapped[float | None] = mapped_column(Float, nullable=True)
