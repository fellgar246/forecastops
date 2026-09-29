"""Store the cloud registry status on a model version."""

import sqlalchemy as sa
from alembic import op

revision = "0003_registry_status"
down_revision = "0002_domain"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "model_versions",
        sa.Column("registry_status", sa.String(length=32), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("model_versions", "registry_status")
