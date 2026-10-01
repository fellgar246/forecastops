"""Store drift monitoring reports."""

import sqlalchemy as sa
from alembic import op

revision = "0007_monitoring"
down_revision = "0006_schedules"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "monitoring_reports",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("model_version_id", sa.String(length=36), nullable=False),
        sa.Column("dataset_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("date_max", sa.Date(), nullable=False),
        sa.Column("baseline_start", sa.Date(), nullable=False),
        sa.Column("baseline_end", sa.Date(), nullable=False),
        sa.Column("recent_start", sa.Date(), nullable=False),
        sa.Column("recent_end", sa.Date(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("retrain_request_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name="fk_monitoring_reports_dataset_id_datasets",
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name="fk_monitoring_reports_model_version_id_model_versions",
        ),
        sa.ForeignKeyConstraint(
            ["retrain_request_id"],
            ["retrain_requests.id"],
            name="fk_monitoring_reports_retrain_request_id_retrain_requests",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_monitoring_reports"),
    )
    op.create_index(
        "ix_monitoring_reports_created_at",
        "monitoring_reports",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_monitoring_reports_created_at", table_name="monitoring_reports")
    op.drop_table("monitoring_reports")
