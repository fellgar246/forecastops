"""Store an explanation attempt for a forecast run."""

import sqlalchemy as sa
from alembic import op

revision = "0005_ai_explanations"
down_revision = "0004_forecast_idempotency"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ai_explanations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("forecast_run_id", sa.String(length=36), nullable=False),
        sa.Column("scope_key", sa.String(length=500), nullable=False),
        sa.Column("scope", sa.JSON(), nullable=False),
        sa.Column("model_id", sa.String(length=200), nullable=False),
        sa.Column("prompt_version", sa.String(length=32), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("explanation", sa.JSON(), nullable=True),
        sa.Column("package", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["forecast_run_id"],
            ["forecast_runs.id"],
            name="fk_ai_explanations_forecast_run_id_forecast_runs",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_ai_explanations"),
    )
    op.create_index(
        "ix_ai_explanations_forecast_run_id",
        "ai_explanations",
        ["forecast_run_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_ai_explanations_forecast_run_id", table_name="ai_explanations")
    op.drop_table("ai_explanations")
