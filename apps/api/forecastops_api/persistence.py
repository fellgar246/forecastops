"""Domain tables stored in PostgreSQL.

Services reach these rows only through repositories.
"""

from datetime import date, datetime

from sqlalchemy import JSON, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from forecastops_api.db import Base


class DatasetRow(Base):
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


class TrainingRunRow(Base):
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


class ModelVersionRow(Base):
    """One registered model version and its promotion status."""

    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_family: Mapped[str] = mapped_column(String(64))
    version: Mapped[str] = mapped_column(String(64))
    training_run_id: Mapped[str] = mapped_column(ForeignKey("training_runs.id"))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    dataset_version: Mapped[str] = mapped_column(String(64))
    registry_arn: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32))
    metrics: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    approved_by: Mapped[str | None] = mapped_column(String(200), nullable=True)
    rejected_by: Mapped[str | None] = mapped_column(String(200), nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class PromotionDecisionRow(Base):
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


class ForecastRunRow(Base):
    """One batch forecast job."""

    __tablename__ = "forecast_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    dataset_version: Mapped[str] = mapped_column(String(64))
    horizon: Mapped[int] = mapped_column(Integer)
    granularity: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(32))
    output_uri: Mapped[str] = mapped_column(Text, default="")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ForecastPointRow(Base):
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
