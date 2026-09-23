"""Migration harness tests."""

from alembic import command
from alembic.config import Config

from forecastops_api.db import Base
from forecastops_api.settings import get_settings


def test_metadata_starts_without_domain_tables() -> None:
    assert Base.metadata.tables == {}


def test_initial_migration_applies(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "app.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{database_path}")
    get_settings.cache_clear()

    config = Config("apps/api/alembic.ini")
    command.upgrade(config, "head")

    assert database_path.exists()
