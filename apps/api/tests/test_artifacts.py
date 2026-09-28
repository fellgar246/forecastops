"""Contract tests for local and remote artifact stores."""

import sys
import types
from pathlib import Path

import pytest
from envfile import ROOT, parse_env_file
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from forecastops_api.artifacts import (
    ArtifactNotFound,
    ArtifactStore,
    InvalidArtifactPrefix,
    LocalArtifactStore,
    S3ArtifactStore,
    normalize_prefix,
    require_allowed_key,
    select_artifact_store,
)
from forecastops_api.db import Base
from forecastops_api.main import create_app
from forecastops_api.settings import Settings


class _ObjectBlob:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _MissingObject(Exception):
    def __init__(self) -> None:
        self.response = {"Error": {"Code": "NoSuchKey", "Message": "The object was not found."}}
        super().__init__("The object was not found.")


class MemoryObjectClient:
    """In-memory double for the remote object client."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.presign_params: list[dict[str, str]] = []

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> dict[str, object]:
        self.objects[(Bucket, Key)] = bytes(Body)
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        payload = self.objects.get((Bucket, Key))
        if payload is None:
            raise _MissingObject
        return {"Body": _ObjectBlob(payload)}

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str,
        ContinuationToken: str | None = None,
    ) -> dict[str, object]:
        keys = sorted(
            key for bucket, key in self.objects if bucket == Bucket and key.startswith(Prefix)
        )
        start = 0 if ContinuationToken is None else int(ContinuationToken)
        page = keys[start : start + 1]
        truncated = start + 1 < len(keys)
        result: dict[str, object] = {
            "Contents": [{"Key": key} for key in page],
            "IsTruncated": truncated,
        }
        if truncated:
            result["NextContinuationToken"] = str(start + 1)
        return result

    def delete_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        self.objects.pop((Bucket, Key), None)
        return {}

    def generate_presigned_url(
        self,
        ClientMethod: str,
        *,
        Params: dict[str, str],
        ExpiresIn: int,
    ) -> str:
        self.presign_params.append(dict(Params))
        key = Params["Key"]
        return f"https://example.invalid/{key}?method={ClientMethod}&expires={ExpiresIn}"


@pytest.fixture
def artifact_store(request: pytest.FixtureRequest, tmp_path: Path) -> ArtifactStore:
    """Build one store implementation for the shared contract."""

    if request.param == "local":
        return LocalArtifactStore(tmp_path / "disk")
    return S3ArtifactStore("forecastops-artifacts", MemoryObjectClient())


@pytest.mark.parametrize("artifact_store", ["local", "s3"], indirect=True)
def test_put_get_list_and_delete_follow_the_same_contract(artifact_store: ArtifactStore) -> None:
    first = artifact_store.put("raw", "batch/observations.csv", b"one")
    replaced = artifact_store.put("raw", "batch/observations.csv", b"two")
    second = artifact_store.put("raw", "other/dimensions.csv", b"dims")
    forecast = artifact_store.put("forecasts", "run-1/forecast.json", b"{}")

    assert first == replaced
    assert artifact_store.get(replaced) == b"two"
    assert artifact_store.list("raw") == sorted([replaced, second])
    assert forecast not in artifact_store.list("raw")
    assert artifact_store.list("forecasts") == [forecast]
    assert artifact_store.list("models") == []

    artifact_store.delete(second)
    artifact_store.delete(second)
    assert second not in artifact_store.list("raw")
    with pytest.raises(ArtifactNotFound):
        artifact_store.get(second)

    with pytest.raises(InvalidArtifactPrefix):
        artifact_store.put("scratch", "notes.txt", b"nope")
    with pytest.raises(InvalidArtifactPrefix):
        artifact_store.list("scratch")
    with pytest.raises(InvalidArtifactPrefix):
        artifact_store.delete(_outside_uri(artifact_store))


def test_prefix_helper_rejects_a_key_outside_the_allowed_list() -> None:
    with pytest.raises(InvalidArtifactPrefix):
        require_allowed_key("scratch/file.csv")
    with pytest.raises(InvalidArtifactPrefix):
        require_allowed_key("file.csv")
    with pytest.raises(InvalidArtifactPrefix):
        normalize_prefix("raw/nested")
    assert require_allowed_key("evaluations/run/report.json") == "evaluations/run/report.json"
    assert normalize_prefix("forecasts/") == "forecasts"


def test_local_store_rejects_a_uri_outside_its_root(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    outside = tmp_path / "other" / "raw" / "secret.csv"
    with pytest.raises(InvalidArtifactPrefix):
        store.get(str(outside))


def test_selector_uses_the_local_store_when_aws_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setitem(sys.modules, "boto3", _boom_boto3())
    settings = _settings(monkeypatch, tmp_path, aws_enabled=False)

    selected = select_artifact_store(settings)

    assert isinstance(selected, LocalArtifactStore)


def test_selector_builds_a_remote_client_only_when_aws_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    constructed: list[tuple[str, str | None]] = []
    fake = types.ModuleType("boto3")

    def client(service_name: str, region_name: str | None = None) -> MemoryObjectClient:
        constructed.append((service_name, region_name))
        return MemoryObjectClient()

    fake.client = client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake)

    local = select_artifact_store(_settings(monkeypatch, tmp_path, aws_enabled=False))
    assert isinstance(local, LocalArtifactStore)
    assert constructed == []

    remote = select_artifact_store(
        _settings(monkeypatch, tmp_path, aws_enabled=True, bucket="forecastops-artifacts")
    )
    assert isinstance(remote, S3ArtifactStore)
    assert constructed == [("s3", "us-east-1")]


def test_remote_store_requires_a_bucket(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(monkeypatch, tmp_path, aws_enabled=True)

    with pytest.raises(ValueError, match="ARTIFACTS_BUCKET"):
        select_artifact_store(settings)


def test_local_mode_stores_a_dataset_without_a_remote_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def forbidden(region: str) -> MemoryObjectClient:
        raise AssertionError(f"remote client constructed for {region}")

    monkeypatch.setattr("forecastops_api.artifacts.build_s3_client", forbidden)
    application, client = _api(tmp_path)
    payload = b"date,units_sold\n"
    response = client.post(
        "/datasets/uploads",
        files={"file": ("observations.csv", payload, "text/csv")},
    )

    assert response.status_code == 201
    body = response.json()
    store = application.state.artifact_store
    assert isinstance(store, LocalArtifactStore)
    assert store.get(body["uri"]) == payload
    assert body["uri"] in store.list("raw")
    assert "boto3" not in sys.modules


def test_aws_mode_issues_a_presigned_url_and_does_not_store_the_body(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    memory = MemoryObjectClient()
    monkeypatch.setattr("forecastops_api.artifacts.build_s3_client", lambda _region: memory)
    settings = _settings(monkeypatch, tmp_path, aws_enabled=True, bucket="forecastops-artifacts")
    _application, client = _api(tmp_path, settings)
    payload = b"private-dataset-bytes"

    denied = client.post("/datasets/uploads", files={"file": ("obs.csv", payload, "text/csv")})
    assert denied.status_code == 409
    assert denied.json()["code"] == "presigned_upload_required"
    assert payload.decode() not in denied.text
    assert memory.objects == {}

    issued = client.post("/datasets/upload-url", json={"name": "incoming/observations.csv"})
    assert issued.status_code == 200
    issued_body = issued.json()
    assert issued_body["uri"] == "s3://forecastops-artifacts/raw/observations.csv"
    assert issued_body["expires_in"] == 900
    assert "observations.csv" in issued_body["url"]
    assert memory.objects == {}
    assert memory.presign_params == [
        {"Bucket": "forecastops-artifacts", "Key": "raw/observations.csv"}
    ]
    assert payload.decode() not in issued.text


def test_local_mode_refuses_a_presigned_upload(tmp_path: Path) -> None:
    _application, client = _api(tmp_path)
    response = client.post("/datasets/upload-url", json={"name": "observations.csv"})
    assert response.status_code == 409
    assert response.json()["code"] == "aws_disabled"


def test_openapi_lists_dataset_upload_routes() -> None:
    client = TestClient(create_app())
    paths = set(client.get("/openapi.json").json()["paths"])
    assert "/datasets/uploads" in paths
    assert "/datasets/upload-url" in paths


def _outside_uri(store: ArtifactStore) -> str:
    if isinstance(store, LocalArtifactStore):
        return "/tmp/forecastops-outside/raw/file.csv"
    return "s3://other-bucket/raw/file.csv"


def _settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    aws_enabled: bool,
    bucket: str = "",
) -> Settings:
    for key, value in parse_env_file(ROOT / ".env.example").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("AWS_ENABLED", "true" if aws_enabled else "false")
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    if bucket:
        monkeypatch.setenv("ARTIFACTS_BUCKET", bucket)
    else:
        monkeypatch.delenv("ARTIFACTS_BUCKET", raising=False)
    return Settings(_env_file=None)


def _api(
    tmp_path: Path,
    settings: Settings | None = None,
) -> tuple[FastAPI, TestClient]:
    application = create_app() if settings is None else create_app(settings)
    if settings is None:
        root = tmp_path / "artifacts"
        application.state.artifact_dir = root
        application.state.artifact_store = LocalArtifactStore(root)
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'api.db'}")
    application.state.engine = engine
    application.state.session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.create_all(engine)
    return application, TestClient(application)


def _boom_boto3() -> types.ModuleType:
    module = types.ModuleType("boto3")

    def client(*_args: object, **_kwargs: object) -> MemoryObjectClient:
        raise AssertionError("remote client constructed")

    module.client = client  # type: ignore[attr-defined]
    return module
