"""Migration harness tests."""

from alembic import command
from alembic.config import Config

from forecastops_api.db import Base
from forecastops_api.persistence import DatasetRow, ForecastPointRow  # noqa: F401
from forecastops_api.settings import get_settings


def test_metadata_includes_domain_tables() -> None:
    names = set(Base.metadata.tables)
    assert {
        "datasets",
        "training_runs",
        "model_versions",
        "promotion_decisions",
        "forecast_runs",
        "forecast_points",
    } <= names


def test_initial_migration_applies(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "app.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+pysqlite:///{database_path}")
    get_settings.cache_clear()

    config = Config("apps/api/alembic.ini")
    command.upgrade(config, "head")

    assert database_path.exists()
    from sqlalchemy import create_engine, inspect

    engine = create_engine(f"sqlite+pysqlite:///{database_path}")
    tables = set(inspect(engine).get_table_names())
    assert "datasets" in tables
    assert "forecast_points" in tables
    columns = {column["name"] for column in inspect(engine).get_columns("forecast_runs")}
    assert "idempotency_key" in columns
