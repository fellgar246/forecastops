"""Local forecast workflow without cloud calls."""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

from forecastops_api.artifacts import LocalArtifactStore
from forecastops_api.db import Base
from forecastops_api.main import create_app
from forecastops_api.settings import get_settings
from forecastops_ml.data.synthetic import DEFAULT_MAX_ROWS, DEFAULT_SEED, generate_dataset

ROUTES = {
    "/health",
    "/health/aws",
    "/datasets",
    "/datasets/{dataset_id}",
    "/datasets/{dataset_id}/validate",
    "/training-runs",
    "/training-runs/{run_id}",
    "/models",
    "/models/{model_id}",
    "/models/{model_id}/approve",
    "/models/{model_id}/reject",
    "/forecasts",
    "/forecasts/{forecast_id}",
    "/forecasts/{forecast_id}/series",
    "/forecasts/{forecast_id}/explanation",
    "/metrics/model-performance",
    "/metrics/data-quality",
    "/admin/retrain",
}


def test_openapi_lists_the_local_routes() -> None:
    client = TestClient(create_app())
    paths = set(client.get("/openapi.json").json()["paths"])
    assert paths >= ROUTES


def test_local_workflow_trains_seasonal_naive_and_returns_p50(tmp_path: Path) -> None:
    client = _client(tmp_path)
    dataset_id = _valid_dataset(client, tmp_path / "data")

    training = client.post(
        "/training-runs",
        json={
            "dataset_id": dataset_id,
            "model_family": "seasonal_naive",
            "configuration": {"folds": 3, "horizon": 7},
        },
    )
    assert training.status_code == 202
    run_id = training.json()["job_id"]
    run = client.get(f"/training-runs/{run_id}")
    assert run.status_code == 200
    assert run.json()["status"] == "COMPLETED"
    assert run.json()["pipeline_execution_arn"] == ""
    artifact_uri = run.json()["artifact_uri"]
    assert artifact_uri.endswith(f"training/{run_id}/evaluation.json")
    assert Path(artifact_uri).is_file()

    model_id = client.get("/models").json()["items"][0]["id"]
    pending = client.post(
        "/forecasts",
        json={"model_id": model_id, "horizon": 7, "granularity": "day"},
    )
    assert pending.status_code == 409
    assert pending.json()["code"] == "model_not_approved"

    approved = client.post("/models/" + model_id + "/approve", json={"actor_id": "analyst-1"})
    assert approved.status_code == 200
    assert approved.json()["status"] == "APPROVED"

    forecast = client.post(
        "/forecasts",
        json={"model_id": model_id, "horizon": 7, "granularity": "day"},
    )
    assert forecast.status_code == 202
    forecast_id = forecast.json()["job_id"]
    stored = client.get(f"/forecasts/{forecast_id}").json()
    assert stored["status"] == "SUCCEEDED"
    assert stored["output_uri"].endswith(f"forecasts/{forecast_id}/forecast.json")
    assert Path(stored["output_uri"]).is_file()
    series = client.get(f"/forecasts/{forecast_id}/series")
    assert series.status_code == 200
    body = series.json()
    points = body["items"]
    assert points
    assert all(point["p50"] is not None for point in points)
    assert body["p50_total"] == sum(point["p50"] for point in points)
    assert body["series"]
    store_id = points[0]["series_id"].split("|", 1)[0]
    filtered = client.get(f"/forecasts/{forecast_id}/series", params={"store": store_id})
    assert filtered.status_code == 200
    assert filtered.json()["items"]
    assert all(item["series_id"].startswith(f"{store_id}|") for item in filtered.json()["items"])
    missing = client.get(
        f"/forecasts/{forecast_id}/series",
        params={"store": "missing-store", "category": "missing-category"},
    )
    assert missing.json()["items"] == []
    assert missing.json()["p50_total"] is None
    dataset_id = client.get(f"/forecasts/{forecast_id}").json()["dataset_id"]
    catalog = client.get(f"/datasets/{dataset_id}/catalog")
    assert catalog.status_code == 200
    assert catalog.json()["categories"]
    category_id = catalog.json()["categories"][0]["id"]
    scoped = client.get(f"/forecasts/{forecast_id}/series", params={"category": category_id})
    assert scoped.status_code == 200
    assert scoped.json()["items"]
    assert "boto3" not in sys.modules

    explanation = client.post(f"/forecasts/{forecast_id}/explanation")
    assert explanation.status_code == 409
    assert explanation.json()["message"] == "Explanations are not enabled."
    assert "summary" not in explanation.json()


def test_rejected_model_cannot_forecast(tmp_path: Path) -> None:
    client = _client(tmp_path)
    dataset_id = _valid_dataset(client, tmp_path / "data")
    training = client.post(
        "/training-runs",
        json={"dataset_id": dataset_id, "model_family": "seasonal_naive"},
    )
    assert training.status_code == 202
    model_id = client.get("/models").json()["items"][0]["id"]
    rejected = client.post(
        f"/models/{model_id}/reject",
        json={"actor_id": "analyst-1", "reason": "human"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "REJECTED"

    forecast = client.post(
        "/forecasts",
        json={"model_id": model_id, "horizon": 7, "granularity": "day"},
    )
    assert forecast.status_code == 409
    assert forecast.json()["code"] == "model_not_approved"


def test_retrain_does_not_approve_the_new_model(tmp_path: Path) -> None:
    client = _client(tmp_path)
    dataset_id = _valid_dataset(client, tmp_path / "data")
    client.post(
        "/training-runs",
        json={"dataset_id": dataset_id, "model_family": "seasonal_naive"},
    )
    model_id = client.get("/models").json()["items"][0]["id"]
    approved = client.post(
        f"/models/{model_id}/approve",
        json={"actor_id": "analyst-1", "promote": True},
    )
    assert approved.json()["status"] == "PRODUCTION"

    retrained = client.post("/admin/retrain")
    assert retrained.status_code == 202
    models = client.get("/models").json()["items"]
    assert len(models) == 2
    new_model = next(item for item in models if item["id"] != model_id)
    assert new_model["status"] not in {"APPROVED", "PRODUCTION"}


def test_horizon_cannot_exceed_the_configured_limit(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.post(
        "/forecasts",
        json={"model_id": "missing", "horizon": 91, "granularity": "day"},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "horizon_too_long"


def _client(tmp_path: Path) -> TestClient:
    database = tmp_path / "api.db"
    get_settings.cache_clear()
    application = create_app()
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite+pysqlite:///{database}")
    application.state.engine = engine
    application.state.session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    application.state.artifact_dir = tmp_path / "artifacts"
    application.state.artifact_store = LocalArtifactStore(application.state.artifact_dir)
    Base.metadata.create_all(engine)
    return TestClient(application)


def _valid_dataset(client: TestClient, directory: Path) -> str:
    generate_dataset(directory, profile="test", seed=DEFAULT_SEED, max_rows=DEFAULT_MAX_ROWS)
    created = client.post(
        "/datasets",
        json={"name": "retail-test", "source": "synthetic", "uri": str(directory)},
    )
    assert created.status_code == 202
    dataset_id = created.json()["job_id"]
    validated = client.post(
        f"/datasets/{dataset_id}/validate",
        json={"as_of": "2026-09-30"},
    )
    assert validated.status_code == 200
    assert validated.json()["status"] == "valid"
    return str(dataset_id)
