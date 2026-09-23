"""Alembic migration environment."""

from pathlib import Path

from alembic import context
from sqlalchemy import create_engine, pool

from forecastops_api.db import Base
from forecastops_api.settings import get_settings

config = context.config
config.set_main_option("script_location", str(Path(__file__).resolve().parent))
target_metadata = Base.metadata


def database_url() -> str:
    """Read the current database URL, ignoring a previously cached value."""

    get_settings.cache_clear()
    return get_settings().database_url


def run_migrations_offline() -> None:
    """Emit SQL without opening a connection."""

    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against the configured database."""

    connectable = create_engine(database_url(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
