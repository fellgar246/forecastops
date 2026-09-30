"""Store data refresh markers, forecast error scores, and retrain requests."""

import sqlalchemy as sa
from alembic import op

revision = "0006_schedules"
down_revision = "0005_ai_explanations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "data_refresh_markers",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("dataset_id", sa.String(length=36), nullable=False),
        sa.Column("refreshed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name="fk_data_refresh_markers_dataset_id_datasets",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_data_refresh_markers"),
    )
    op.create_table(
        "forecast_error_evaluations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("forecast_run_id", sa.String(length=36), nullable=False),
        sa.Column("compared_points", sa.Integer(), nullable=False),
        sa.Column("wape", sa.Float(), nullable=True),
        sa.Column("bias", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["forecast_run_id"],
            ["forecast_runs.id"],
            name="fk_forecast_error_evaluations_forecast_run_id_forecast_runs",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_forecast_error_evaluations"),
    )
    op.create_index(
        "ix_forecast_error_evaluations_forecast_run_id",
        "forecast_error_evaluations",
        ["forecast_run_id"],
    )
    op.create_table(
        "retrain_requests",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("dataset_id", sa.String(length=36), nullable=False),
        sa.Column("model_version_id", sa.String(length=36), nullable=False),
        sa.Column("model_family", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("training_run_id", sa.String(length=36), nullable=True),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confirmed_by", sa.String(length=200), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name="fk_retrain_requests_dataset_id_datasets",
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name="fk_retrain_requests_model_version_id_model_versions",
        ),
        sa.ForeignKeyConstraint(
            ["training_run_id"],
            ["training_runs.id"],
            name="fk_retrain_requests_training_run_id_training_runs",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_retrain_requests"),
    )
    op.create_index("ix_retrain_requests_status", "retrain_requests", ["status"])


def downgrade() -> None:
    op.drop_index("ix_retrain_requests_status", table_name="retrain_requests")
    op.drop_table("retrain_requests")
    op.drop_index(
        "ix_forecast_error_evaluations_forecast_run_id",
        table_name="forecast_error_evaluations",
    )
    op.drop_table("forecast_error_evaluations")
    op.drop_table("data_refresh_markers")
