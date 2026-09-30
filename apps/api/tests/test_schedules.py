"""Scheduled refresh, evaluation, and retrain confirmation."""

import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from forecastops_api.artifacts import LocalArtifactStore
from forecastops_api.db import Base
from forecastops_api.main import create_app
from forecastops_api.persistence import ForecastPointRow
from forecastops_api.schedules import SCHEDULE_ACTIONS
from forecastops_api.settings import get_settings
from forecastops_ml.data import load_dataset
from forecastops_ml.data.synthetic import DEFAULT_MAX_ROWS, DEFAULT_SEED, generate_dataset
from forecastops_ml.evaluation.backtest import make_series_id

ROOT = Path(__file__).resolve().parents[3]


def test_monthly_action_stops_at_a_confirmation_record(tmp_path: Path) -> None:
    client = _client(tmp_path)
    _production(client, tmp_path / "data")
    runs_before = client.get("/training-runs").json()["items"]

    monthly = client.post("/admin/schedules/monthly_retrain")
    assert monthly.status_code == 202
    body = monthly.json()
    assert body["status"] == "PENDING"
    assert body["training_run_id"] is None
    assert body["confirmed_by"] is None
    assert client.get("/training-runs").json()["items"] == runs_before

    repeated = client.post("/admin/schedules/monthly_retrain")
    assert repeated.status_code == 202
    assert repeated.json()["id"] == body["id"]
    assert client.get("/training-runs").json()["items"] == runs_before


def test_confirmation_starts_training_and_does_not_approve_the_model(tmp_path: Path) -> None:
    client = _client(tmp_path)
    _production(client, tmp_path / "data")
    production_id = next(
        item["id"]
        for item in client.get("/models").json()["items"]
        if item["status"] == "PRODUCTION"
    )
    runs_before = len(client.get("/training-runs").json()["items"])
    request_id = client.post("/admin/schedules/monthly_retrain").json()["id"]

    confirmed = client.post(
        f"/admin/retrain-requests/{request_id}/confirm",
        json={"actor_id": "analyst-1"},
    )
    assert confirmed.status_code == 202
    assert confirmed.json()["status"] == "CONFIRMED"
    assert confirmed.json()["confirmed_by"] == "analyst-1"
    assert confirmed.json()["training_run_id"]

    assert len(client.get("/training-runs").json()["items"]) == runs_before + 1
    models = client.get("/models").json()["items"]
    new_model = next(item for item in models if item["id"] != production_id)
    assert new_model["status"] not in {"APPROVED", "PRODUCTION"}
    production = [item for item in models if item["status"] == "PRODUCTION"]
    assert [item["id"] for item in production] == [production_id]

    again = client.post(
        f"/admin/retrain-requests/{request_id}/confirm",
        json={"actor_id": "analyst-1"},
    )
    assert again.status_code == 409
    assert again.json()["code"] == "invalid_transition"


def test_daily_refresh_generates_a_forecast(tmp_path: Path) -> None:
    client = _client(tmp_path)
    dataset_id = _production(client, tmp_path / "data")

    daily = client.post("/admin/schedules/daily_forecast")
    assert daily.status_code == 202
    body = daily.json()
    assert body["dataset_id"] == dataset_id
    assert body["forecast_status"] == "SUCCEEDED"
    forecast = client.get(f"/forecasts/{body['forecast_id']}")
    assert forecast.status_code == 200
    assert forecast.json()["horizon"] == 7

    again = client.post("/admin/schedules/daily_forecast")
    assert again.json()["forecast_id"] == body["forecast_id"]
    assert again.json()["marker_id"] != body["marker_id"]


def test_weekly_evaluation_scores_only_arrived_actuals(tmp_path: Path) -> None:
    client = _client(tmp_path)
    dataset = tmp_path / "data"
    _production(client, dataset)
    forecast_id = client.post("/admin/schedules/daily_forecast").json()["forecast_id"]

    waiting = client.post("/admin/schedules/weekly_evaluation")
    assert waiting.status_code == 200
    assert waiting.json()["forecast_run_id"] == forecast_id
    assert waiting.json()["compared_points"] == 0
    assert waiting.json()["wape"] is None

    _move_one_point_onto_an_observation(client, forecast_id, dataset)
    scored = client.post("/admin/schedules/weekly_evaluation")
    assert scored.status_code == 200
    assert scored.json()["compared_points"] == 1
    assert scored.json()["wape"] is not None
    assert scored.json()["bias"] is not None


def test_local_schedule_command_does_not_construct_a_scheduler() -> None:
    script = (ROOT / "scripts" / "schedule" / "run_schedule.py").read_text(encoding="utf-8")
    service = (ROOT / "apps" / "api" / "forecastops_api" / "schedules.py").read_text(
        encoding="utf-8"
    )
    for text in (script, service):
        assert "boto3" not in text
        assert "scheduler.amazonaws" not in text
    assert "daily_forecast" in script
    assert set(SCHEDULE_ACTIONS) == {
        "daily_forecast",
        "weekly_evaluation",
        "monthly_retrain",
    }


def test_schedule_handler_leaves_monthly_retrain_pending() -> None:
    path = ROOT / "infra" / "modules" / "eventbridge" / "src" / "handler.py"
    spec = importlib.util.spec_from_file_location("schedule_handler", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pending = module.handler({"action": "monthly_retrain"}, None)
    assert pending == {
        "action": "monthly_retrain",
        "status": "PENDING",
        "training_started": False,
    }
    accepted = module.handler({"action": "daily_forecast"}, None)
    assert accepted["training_started"] is False
    assert set(SCHEDULE_ACTIONS) == module.ACTIONS


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


def _production(client: TestClient, directory: Path) -> str:
    generate_dataset(directory, profile="test", seed=DEFAULT_SEED, max_rows=DEFAULT_MAX_ROWS)
    created = client.post(
        "/datasets",
        json={"name": "retail-test", "source": "synthetic", "uri": str(directory)},
    )
    assert created.status_code == 202
    dataset_id = str(created.json()["job_id"])
    validated = client.post(
        f"/datasets/{dataset_id}/validate",
        json={"as_of": "2026-09-30"},
    )
    assert validated.status_code == 200
    training = client.post(
        "/training-runs",
        json={"dataset_id": dataset_id, "model_family": "seasonal_naive"},
    )
    assert training.status_code == 202
    model_id = client.get("/models").json()["items"][0]["id"]
    approved = client.post(
        f"/models/{model_id}/approve",
        json={"actor_id": "analyst-1", "promote": True},
    )
    assert approved.json()["status"] == "PRODUCTION"
    return dataset_id


def _move_one_point_onto_an_observation(
    client: TestClient,
    forecast_id: str,
    dataset: Path,
) -> None:
    frame, _dimensions = load_dataset(dataset)
    dates = frame.column("date").to_pylist()
    stores = frame.column("store_id").to_pylist()
    skus = frame.column("sku_id").to_pylist()
    units = frame.column("units_sold").to_pylist()
    chosen: tuple[object, str, str] | None = None
    for day, store_id, sku_id, sold in zip(dates, stores, skus, units, strict=True):
        if isinstance(sold, bool) or not isinstance(sold, int | float) or float(sold) <= 0:
            continue
        if not isinstance(store_id, str) or not isinstance(sku_id, str):
            continue
        chosen = (day, store_id, sku_id)
        break
    assert chosen is not None
    day, store_id, sku_id = chosen
    factory = client.app.state.session_factory
    session: Session = factory()
    try:
        point = session.scalars(
            select(ForecastPointRow).where(ForecastPointRow.forecast_run_id == forecast_id)
        ).first()
        assert point is not None
        point.series_id = make_series_id(store_id, sku_id)
        point.date = day
        session.commit()
    finally:
        session.close()
