"""Store an optional idempotency key on a forecast run."""

import sqlalchemy as sa
from alembic import op

revision = "0004_forecast_idempotency"
down_revision = "0003_registry_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "forecast_runs",
        sa.Column("idempotency_key", sa.String(length=200), nullable=True),
    )
    op.create_index(
        "uq_forecast_runs_idempotency_key",
        "forecast_runs",
        ["idempotency_key"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_forecast_runs_idempotency_key", table_name="forecast_runs")
    op.drop_column("forecast_runs", "idempotency_key")
