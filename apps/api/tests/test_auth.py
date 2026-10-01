"""Bearer tokens, tenant claims, and cross-tenant isolation."""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey
from envfile import ROOT, parse_env_file
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from forecastops_api.artifacts import LocalArtifactStore
from forecastops_api.auth import CognitoTokenVerifier
from forecastops_api.db import Base
from forecastops_api.dynamodb import DynamoRepository, MemoryMetadataTable
from forecastops_api.errors import ApiError
from forecastops_api.main import create_app
from forecastops_api.persistence import DatasetRow, ForecastPointRow, ForecastRunRow
from forecastops_api.repositories import Repository
from forecastops_api.settings import Settings, get_settings
from forecastops_ml.data.synthetic import DEFAULT_MAX_ROWS, DEFAULT_SEED, generate_dataset

_POOL = "us-east-1_testpool"
_CLIENT = "test-client"
_ISSUER = f"https://cognito-idp.us-east-1.amazonaws.com/{_POOL}"


class _FixedKeys:
    def __init__(self, key: RSAPublicKey) -> None:
        self._key = key

    def key_for(self, token: str) -> RSAPublicKey:
        _ = token
        return self._key


def test_cloud_mode_rejects_a_missing_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _cloud_client(monkeypatch, tmp_path)

    missing = client.get("/datasets")
    health = client.get("/health")

    assert missing.status_code == 401
    assert missing.json()["code"] == "unauthorized"
    assert health.status_code == 200


def test_cloud_mode_rejects_a_token_without_a_tenant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    private, _public = _keypair()
    client = _cloud_client(monkeypatch, tmp_path, private_key=private)
    token = _access_token(private, tenant_id=None)

    response = client.get("/datasets", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 403
    assert response.json()["code"] == "forbidden"


def test_a_tenant_reads_its_rows_and_cannot_read_another(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    private, _public = _keypair()
    application, client = _cloud_app(monkeypatch, tmp_path, private_key=private)
    table = application.state.metadata_table
    _seed(DynamoRepository(table, tenant_id="alpha"), "alpha", tmp_path)
    _seed(DynamoRepository(table, tenant_id="beta"), "beta", tmp_path)
    alpha = {"Authorization": f"Bearer {_access_token(private, tenant_id='alpha')}"}
    beta = {"Authorization": f"Bearer {_access_token(private, tenant_id='beta')}"}

    listed = client.get("/datasets", headers=alpha)
    own = client.get("/datasets/alpha-dataset", headers=alpha)
    foreign = client.get("/datasets/beta-dataset", headers=alpha)
    own_series = client.get("/forecasts/alpha-forecast/series", headers=alpha)
    foreign_series = client.get("/forecasts/beta-forecast/series", headers=alpha)
    beta_list = client.get("/datasets", headers=beta)

    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()["items"]] == ["alpha-dataset"]
    assert own.status_code == 200
    assert own.json()["name"] == "alpha-sales"
    assert foreign.status_code == 404
    assert own_series.status_code == 200
    assert own_series.json()["items"][0]["p50"] == 2
    assert foreign_series.status_code == 404
    assert [item["id"] for item in beta_list.json()["items"]] == ["beta-dataset"]


def test_local_mode_stamps_tenant_local_without_a_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    application, client = _local_app(monkeypatch, tmp_path)
    directory = tmp_path / "data"
    generate_dataset(directory, profile="test", seed=DEFAULT_SEED, max_rows=DEFAULT_MAX_ROWS)

    created = client.post(
        "/datasets",
        json={"name": "retail-test", "source": "synthetic", "uri": str(directory)},
    )
    uploaded = client.post(
        "/datasets/uploads",
        files={"file": ("observations.csv", b"date,units_sold\n", "text/csv")},
    )

    assert created.status_code == 202
    assert uploaded.status_code == 201
    assert "/tenant/local/raw/" in uploaded.json()["uri"]
    session = application.state.session_factory()
    row = Repository(session).get_dataset(created.json()["job_id"])
    session.close()
    assert row is not None
    assert row.tenant_id == "local"
    assert client.get("/datasets").status_code == 200


def _seed(repository: DynamoRepository, tenant: str, tmp_path: Path) -> None:
    created_at = datetime(2024, 5, 2, tzinfo=UTC)
    repository.add(
        DatasetRow(
            id=f"{tenant}-dataset",
            name=f"{tenant}-sales",
            version="1",
            source="synthetic",
            status="registered",
            uri=str(tmp_path / tenant),
            schema_version="1",
            row_count=1,
            date_min=None,
            date_max=None,
            quality_report=None,
            created_at=created_at,
        )
    )
    repository.add(
        ForecastRunRow(
            id=f"{tenant}-forecast",
            model_version_id=f"{tenant}-model",
            dataset_id=f"{tenant}-dataset",
            dataset_version="1",
            horizon=1,
            granularity="day",
            status="SUCCEEDED",
            output_uri="",
            error_message=None,
            idempotency_key=None,
            created_at=created_at,
        )
    )
    repository.add(
        ForecastPointRow(
            forecast_run_id=f"{tenant}-forecast",
            series_id="store|sku",
            date=date(2024, 5, 3),
            p10=1,
            p50=2,
            p90=3,
            actual=None,
        )
    )


def _cloud_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    private_key: RSAPrivateKey | None = None,
) -> TestClient:
    _application, client = _cloud_app(monkeypatch, tmp_path, private_key=private_key)
    return client


def _cloud_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    private_key: RSAPrivateKey | None = None,
):
    monkeypatch.setattr(
        "forecastops_api.observability.cloudwatch_client",
        lambda _region: object(),
    )
    application = create_app(
        _settings(monkeypatch, tmp_path, execution_mode="aws"),
        metadata_table=MemoryMetadataTable(),
    )
    if private_key is not None:
        application.state.token_verifier = CognitoTokenVerifier(
            issuer=_ISSUER,
            client_id=_CLIENT,
            keys=_FixedKeys(private_key.public_key()),
        )
    return application, TestClient(application)


def _local_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    application = create_app(_settings(monkeypatch, tmp_path))
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'api.db'}")
    application.state.engine = engine
    application.state.session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    application.state.artifact_store = LocalArtifactStore(tmp_path / "artifacts")
    Base.metadata.create_all(engine)
    return application, TestClient(application)


def _settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    execution_mode: str = "local",
) -> Settings:
    for key, value in parse_env_file(ROOT / ".env.example").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("EXECUTION_MODE", execution_mode)
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    if execution_mode == "aws":
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("COGNITO_USER_POOL_ID", _POOL)
        monkeypatch.setenv("COGNITO_APP_CLIENT_ID", _CLIENT)
        monkeypatch.setenv("METADATA_TABLE_NAME", "forecastops-dev-metadata")
    get_settings.cache_clear()
    return Settings(_env_file=None)


def _keypair() -> tuple[RSAPrivateKey, RSAPublicKey]:
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private, private.public_key()


def _access_token(private: RSAPrivateKey, *, tenant_id: str | None) -> str:
    now = datetime.now(UTC)
    payload: dict[str, object] = {
        "sub": "user-1",
        "iss": _ISSUER,
        "client_id": _CLIENT,
        "token_use": "access",
        "exp": now + timedelta(minutes=5),
        "iat": now,
    }
    if tenant_id is not None:
        payload["tenant_id"] = tenant_id
    encoded = jwt.encode(payload, private, algorithm="RS256")
    assert isinstance(encoded, str)
    return encoded


def test_verifier_rejects_a_token_for_another_client() -> None:
    private, public = _keypair()
    verifier = CognitoTokenVerifier(issuer=_ISSUER, client_id=_CLIENT, keys=_FixedKeys(public))
    now = datetime.now(UTC)
    token = jwt.encode(
        {
            "sub": "user-1",
            "iss": _ISSUER,
            "client_id": "other-client",
            "token_use": "access",
            "tenant_id": "alpha",
            "exp": now + timedelta(minutes=5),
            "iat": now,
        },
        private,
        algorithm="RS256",
    )

    with pytest.raises(ApiError) as caught:
        verifier.verify(token)

    assert caught.value.status_code == 401
