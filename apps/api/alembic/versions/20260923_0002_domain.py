"""Domain tables for datasets, training, models, and forecasts."""

import sqlalchemy as sa
from alembic import op

revision = "0002_domain"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("uri", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("date_min", sa.Date(), nullable=True),
        sa.Column("date_max", sa.Date(), nullable=True),
        sa.Column("quality_report", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_datasets"),
    )
    op.create_table(
        "training_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("dataset_id", sa.String(length=36), nullable=False),
        sa.Column("dataset_version", sa.String(length=64), nullable=False),
        sa.Column("model_family", sa.String(length=64), nullable=False),
        sa.Column("configuration", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("artifact_uri", sa.Text(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("git_sha", sa.String(length=64), nullable=False),
        sa.Column("pipeline_execution_arn", sa.Text(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name="fk_training_runs_dataset_id_datasets",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_training_runs"),
    )
    op.create_table(
        "model_versions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("model_family", sa.String(length=64), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("training_run_id", sa.String(length=36), nullable=False),
        sa.Column("dataset_id", sa.String(length=36), nullable=False),
        sa.Column("dataset_version", sa.String(length=64), nullable=False),
        sa.Column("registry_arn", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("approved_by", sa.String(length=200), nullable=True),
        sa.Column("rejected_by", sa.String(length=200), nullable=True),
        sa.Column("rejection_reason", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name="fk_model_versions_dataset_id_datasets",
        ),
        sa.ForeignKeyConstraint(
            ["training_run_id"],
            ["training_runs.id"],
            name="fk_model_versions_training_run_id_training_runs",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_model_versions"),
    )
    op.create_table(
        "promotion_decisions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("candidate_id", sa.String(length=36), nullable=False),
        sa.Column("reference_id", sa.String(length=64), nullable=False),
        sa.Column("thresholds", sa.JSON(), nullable=False),
        sa.Column("checks", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("reason", sa.String(length=64), nullable=True),
        sa.Column("p90_coverage", sa.Float(), nullable=True),
        sa.Column("regressed_categories", sa.JSON(), nullable=False),
        sa.Column("actor_id", sa.String(length=200), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["candidate_id"],
            ["model_versions.id"],
            name="fk_promotion_decisions_candidate_id_model_versions",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_promotion_decisions"),
    )
    op.create_table(
        "forecast_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("model_version_id", sa.String(length=36), nullable=False),
        sa.Column("dataset_id", sa.String(length=36), nullable=False),
        sa.Column("dataset_version", sa.String(length=64), nullable=False),
        sa.Column("horizon", sa.Integer(), nullable=False),
        sa.Column("granularity", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("output_uri", sa.Text(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name="fk_forecast_runs_dataset_id_datasets",
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name="fk_forecast_runs_model_version_id_model_versions",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_forecast_runs"),
    )
    op.create_table(
        "forecast_points",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("forecast_run_id", sa.String(length=36), nullable=False),
        sa.Column("series_id", sa.String(length=200), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("p10", sa.Float(), nullable=True),
        sa.Column("p50", sa.Float(), nullable=False),
        sa.Column("p90", sa.Float(), nullable=True),
        sa.Column("actual", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["forecast_run_id"],
            ["forecast_runs.id"],
            name="fk_forecast_points_forecast_run_id_forecast_runs",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_forecast_points"),
    )
    op.create_index(
        "ix_forecast_points_forecast_run_id",
        "forecast_points",
        ["forecast_run_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_forecast_points_forecast_run_id", table_name="forecast_points")
    op.drop_table("forecast_points")
    op.drop_table("forecast_runs")
    op.drop_table("promotion_decisions")
    op.drop_table("model_versions")
    op.drop_table("training_runs")
    op.drop_table("datasets")
