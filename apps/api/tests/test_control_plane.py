"""Cloud control plane wiring. Local mode stays on PostgreSQL."""

import re
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest
from envfile import ROOT, parse_env_file
from fastapi.testclient import TestClient

from forecastops_api.auth import Caller
from forecastops_api.dynamodb import DynamoRepository, MemoryMetadataTable, open_metadata_table
from forecastops_api.main import create_app
from forecastops_api.observability import LogMetricPublisher, configure_metrics
from forecastops_api.persistence import DatasetRow
from forecastops_api.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> object:
    yield
    get_settings.cache_clear()
    configure_metrics(get_settings(), publisher=LogMetricPublisher())
    get_settings.cache_clear()


def test_local_profile_uses_postgresql_and_does_not_open_dynamodb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("cloud metadata client constructed")

    monkeypatch.setattr("forecastops_api.dynamodb.open_metadata_table", forbidden)
    application = create_app(_settings(monkeypatch, tmp_path))

    assert application.state.metadata_backend == "sql"
    assert application.state.session_factory is not None
    assert application.state.metadata_table is None


def test_local_startup_does_not_construct_a_metadata_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def resource(service_name: str, **kwargs: object) -> object:
        calls.append(service_name)
        assert "aws_access_key_id" not in kwargs
        assert "aws_secret_access_key" not in kwargs
        raise AssertionError(kwargs)

    fake = types.ModuleType("boto3")
    fake.resource = resource  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake)

    create_app(_settings(monkeypatch, tmp_path))
    assert calls == []


def test_cloud_profile_serves_routes_from_dynamodb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "forecastops_api.observability.cloudwatch_client",
        lambda _region: object(),
    )
    table = MemoryMetadataTable()
    application = create_app(
        _settings(monkeypatch, tmp_path, execution_mode="aws", aws_enabled=False),
        metadata_table=table,
    )
    assert application.state.metadata_backend == "dynamodb"
    assert application.state.session_factory is None
    created_at = datetime(2024, 5, 2, tzinfo=UTC)
    DynamoRepository(table).add(
        DatasetRow(
            id="dataset-1",
            name="sales",
            version="1",
            source="synthetic",
            status="registered",
            uri=str(tmp_path),
            schema_version="1",
            row_count=4,
            date_min=None,
            date_max=None,
            quality_report=None,
            created_at=created_at,
        )
    )
    application.state.token_verifier = _LocalTenantVerifier()
    client = TestClient(application)
    headers = {"Authorization": "Bearer local-token"}

    health = client.get("/health")
    listed = client.get("/datasets", headers=headers)
    one = client.get("/datasets/dataset-1", headers=headers)

    assert health.status_code == 200
    assert health.json()["execution_mode"] == "aws"
    assert health.json()["status"] == "healthy"
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()["items"]] == ["dataset-1"]
    assert one.status_code == 200
    assert one.json()["name"] == "sales"


def test_metadata_client_uses_the_execution_role_without_access_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class _Resource:
        def Table(self, name: str) -> dict[str, str]:
            return {"name": name}

    def resource(service_name: str, **kwargs: object) -> _Resource:
        captured["service"] = service_name
        captured["kwargs"] = kwargs
        return _Resource()

    fake = types.ModuleType("boto3")
    fake.resource = resource  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake)
    settings = _settings(
        monkeypatch,
        tmp_path,
        execution_mode="aws",
        aws_enabled=False,
        table_name="forecastops-dev-metadata",
    )

    table = open_metadata_table(settings)

    assert captured["service"] == "dynamodb"
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["region_name"] == "us-east-1"
    assert "aws_access_key_id" not in kwargs
    assert "aws_secret_access_key" not in kwargs
    assert table is not None


def test_gateway_routes_match_the_http_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    get_settings.cache_clear()
    client = TestClient(create_app(_settings(monkeypatch, tmp_path)))
    paths = client.get("/openapi.json").json()["paths"]
    expected = {
        f"{method.upper()} {path}"
        for path, operations in paths.items()
        for method in operations
        if method in {"get", "post", "put", "patch", "delete"}
    }
    gateway = (ROOT / "infra" / "modules" / "api_gateway" / "main.tf").read_text(encoding="utf-8")
    declared = set(re.findall(r"(?:GET|POST|PUT|PATCH|DELETE) /[^\"\n]+", gateway))
    assert expected == declared


class _LocalTenantVerifier:
    def verify(self, token: str) -> Caller:
        assert token == "local-token"
        return Caller(tenant_id="local")


def _settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    execution_mode: str = "local",
    aws_enabled: bool = False,
    table_name: str = "",
) -> Settings:
    for key, value in parse_env_file(ROOT / ".env.example").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("EXECUTION_MODE", execution_mode)
    monkeypatch.setenv("AWS_ENABLED", "true" if aws_enabled else "false")
    if execution_mode == "aws":
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("COGNITO_USER_POOL_ID", "us-east-1_testpool")
        monkeypatch.setenv("COGNITO_APP_CLIENT_ID", "test-client")
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    if table_name:
        monkeypatch.setenv("METADATA_TABLE_NAME", table_name)
    else:
        monkeypatch.delenv("METADATA_TABLE_NAME", raising=False)
    get_settings.cache_clear()
    return Settings(_env_file=None)
