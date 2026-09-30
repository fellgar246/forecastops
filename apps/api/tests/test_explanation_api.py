"""Explanation routes: mock adapter, cache, quotas, and validation."""

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from forecastops_api.artifacts import LocalArtifactStore
from forecastops_api.db import Base
from forecastops_api.main import create_app
from forecastops_api.persistence import (
    AIExplanationRow,
    DatasetRow,
    ForecastPointRow,
    ForecastRunRow,
    ModelVersionRow,
)
from forecastops_api.settings import get_settings
from forecastops_ml.explanations import (
    ExplanationDraft,
    ExplanationUsage,
    explanation_validation_failures,
)


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_get_is_missing_until_a_mock_explanation_is_stored(tmp_path: Path) -> None:
    client = _client(tmp_path)
    forecast_id = _seed_forecast(client)
    missing = client.get(f"/forecasts/{forecast_id}/explanation")
    assert missing.status_code == 404

    created = client.post(f"/forecasts/{forecast_id}/explanation")
    assert created.status_code == 202
    body = created.json()
    assert body["model_id"] == "mock-explainer"
    assert "1840" in body["summary"]
    assert body["uncertainty"]
    assert body["package"]["forecast"]["p50"] == 1840
    assert "boto3" not in sys.modules

    cached = client.post(f"/forecasts/{forecast_id}/explanation")
    assert cached.status_code == 200
    assert cached.json()["id"] == body["id"]
    stored = client.get(f"/forecasts/{forecast_id}/explanation")
    assert stored.status_code == 200
    assert stored.json()["summary"] == body["summary"]


def test_mock_mode_does_not_construct_the_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[object] = []

    class Spy:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            constructed.append(True)
            raise AssertionError("provider constructed")

    monkeypatch.setattr("forecastops_ml.explanations.clients.BedrockExplanationClient", Spy)
    client = _client(tmp_path)
    forecast_id = _seed_forecast(client)
    response = client.post(f"/forecasts/{forecast_id}/explanation")
    assert response.status_code == 202
    assert constructed == []


def test_a_draft_that_changes_p50_is_not_stored(tmp_path: Path) -> None:
    client = _client(tmp_path)
    forecast_id = _seed_forecast(client)
    client.app.state.explanation_client = _MutatingClient()
    before = explanation_validation_failures()
    response = client.post(f"/forecasts/{forecast_id}/explanation")
    assert response.status_code == 422
    assert response.json()["details"]["check"] == "forecast_numbers"
    assert "summary" not in response.json()
    assert explanation_validation_failures() == before + 1
    assert client.get(f"/forecasts/{forecast_id}/explanation").status_code == 404


def test_daily_limit_does_not_call_the_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BEDROCK_ENABLED", "true")
    monkeypatch.setenv("MAX_BEDROCK_CALLS_PER_DAY", "1")
    get_settings.cache_clear()
    constructed: list[object] = []

    class Spy:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            constructed.append(True)
            raise AssertionError("provider constructed")

    monkeypatch.setattr("forecastops_ml.explanations.clients.BedrockExplanationClient", Spy)
    client = _client(tmp_path)
    forecast_id = _seed_forecast(client)
    _seed_call(client, forecast_id)
    response = client.post(f"/forecasts/{forecast_id}/explanation")
    assert response.status_code == 429
    assert response.json()["code"] == "explanation_quota"
    assert "30" not in response.json()["message"]
    assert "1 call" in response.json()["message"]
    assert "00:00 UTC" in response.json()["message"]
    assert constructed == []
    assert "summary" not in response.json()


def test_disabled_explanations_stay_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_ENABLED", "false")
    get_settings.cache_clear()
    client = _client(tmp_path)
    forecast_id = _seed_forecast(client)
    response = client.post(f"/forecasts/{forecast_id}/explanation")
    assert response.status_code == 409
    assert response.json()["message"] == "Explanations are not enabled."
    assert "summary" not in response.json()


def test_input_ceiling_is_enforced_before_the_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_BEDROCK_INPUT_TOKENS", "1")
    get_settings.cache_clear()
    client = _client(tmp_path)
    forecast_id = _seed_forecast(client)

    class Boom:
        model_id = "unused"

        def explain(self, package: object) -> ExplanationDraft:
            raise AssertionError(package)

        @property
        def usage(self) -> ExplanationUsage:
            return ExplanationUsage(input_tokens=0, output_tokens=0, latency_ms=0)

    client.app.state.explanation_client = Boom()
    response = client.post(f"/forecasts/{forecast_id}/explanation")
    assert response.status_code == 429
    assert "input token ceiling" in response.json()["message"]


class _MutatingClient:
    model_id = "bad-explainer"

    def __init__(self) -> None:
        self._usage = ExplanationUsage(input_tokens=4, output_tokens=4, latency_ms=1)

    def explain(self, package: object) -> ExplanationDraft:
        _ = package
        return ExplanationDraft(
            summary="The median forecast is 1900.",
            drivers=["historical_context"],
            risks=["The range may be wide."],
            uncertainty_note="The forecast remains uncertain.",
            recommended_checks=["Compare the median forecast with history."],
        )

    @property
    def usage(self) -> ExplanationUsage:
        return self._usage


def _client(tmp_path: Path) -> TestClient:
    database = tmp_path / "api.db"
    get_settings.cache_clear()
    application = create_app()
    engine = create_engine(f"sqlite+pysqlite:///{database}")
    application.state.engine = engine
    application.state.session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    application.state.artifact_dir = tmp_path / "artifacts"
    application.state.artifact_store = LocalArtifactStore(application.state.artifact_dir)
    Base.metadata.create_all(engine)
    return TestClient(application)


def _seed_forecast(client: TestClient) -> str:
    now = datetime.now(UTC)
    session = client.app.state.session_factory()
    dataset = DatasetRow(
        id="dataset-1",
        name="retail-test",
        version="1",
        source="synthetic",
        status="valid",
        uri="missing",
        schema_version="retail-demand-v1",
        row_count=1,
        date_min=None,
        date_max=None,
        quality_report=None,
        created_at=now,
    )
    model = ModelVersionRow(
        id="model-1",
        model_family="seasonal_naive",
        version="1",
        training_run_id="run-1",
        dataset_id=dataset.id,
        dataset_version="1",
        registry_arn="",
        status="APPROVED",
        metrics={"wape": 0.132, "bias": 0.004},
        approved_at=now,
        created_at=now,
    )
    forecast = ForecastRunRow(
        id="forecast-1",
        model_version_id=model.id,
        dataset_id=dataset.id,
        dataset_version="1",
        horizon=56,
        granularity="day",
        status="SUCCEEDED",
        output_uri="",
        error_message=None,
        created_at=now,
    )
    point = ForecastPointRow(
        forecast_run_id=forecast.id,
        series_id="CDMX-01|sku-1",
        date=datetime(2026, 9, 21, tzinfo=UTC).date(),
        p10=1530,
        p50=1840,
        p90=2130,
        actual=None,
    )
    session.add(dataset)
    session.add(model)
    session.add(forecast)
    session.add(point)
    session.commit()
    session.close()
    return forecast.id


def _seed_call(client: TestClient, forecast_id: str) -> None:
    session = client.app.state.session_factory()
    session.add(
        AIExplanationRow(
            id="prior-call",
            forecast_run_id=forecast_id,
            scope_key='{"series":"other"}',
            scope={"series": "other"},
            model_id="mock-explainer",
            prompt_version="v1",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            explanation=None,
            package=None,
            status="invalid",
            created_at=datetime.now(UTC),
        )
    )
    session.commit()
    session.close()
