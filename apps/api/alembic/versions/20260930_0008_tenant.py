"""Add tenant_id to domain rows and scope forecast idempotency keys."""

import sqlalchemy as sa
from alembic import op

revision = "0008_tenant"
down_revision = "0007_monitoring"
branch_labels = None
depends_on = None

_TABLES = (
    "datasets",
    "training_runs",
    "model_versions",
    "promotion_decisions",
    "forecast_runs",
    "ai_explanations",
    "data_refresh_markers",
    "forecast_error_evaluations",
    "retrain_requests",
    "monitoring_reports",
    "forecast_points",
)


def upgrade() -> None:
    for table in _TABLES:
        op.add_column(
            table,
            sa.Column(
                "tenant_id",
                sa.String(length=64),
                nullable=False,
                server_default=sa.text("'local'"),
            ),
        )
        op.create_index(f"ix_{table}_tenant_id", table, ["tenant_id"])
    op.drop_index("uq_forecast_runs_idempotency_key", table_name="forecast_runs")
    op.create_index(
        "uq_forecast_runs_tenant_idempotency",
        "forecast_runs",
        ["tenant_id", "idempotency_key"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_forecast_runs_tenant_idempotency", table_name="forecast_runs")
    op.create_index(
        "uq_forecast_runs_idempotency_key",
        "forecast_runs",
        ["idempotency_key"],
        unique=True,
    )
    for table in reversed(_TABLES):
        op.drop_index(f"ix_{table}_tenant_id", table_name=table)
        op.drop_column(table, "tenant_id")
