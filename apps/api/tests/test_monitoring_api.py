"""Drift reports on the metrics API, without starting training."""

from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from forecastops_api.artifacts import LocalArtifactStore
from forecastops_api.db import Base
from forecastops_api.main import create_app
from forecastops_api.persistence import (
    ForecastErrorEvaluationRow,
    ForecastRunRow,
    ModelVersionRow,
    TrainingRunRow,
)
from forecastops_api.services import ForecastService
from forecastops_api.settings import get_settings

START = date(2026, 1, 1)
QUIET_PRICES = [1.0] * 14 + [5.0] * 14
WARNING_PRICES = [1.0] * 8 + [5.0] * 20


class SpyTrainingClient:
    """Records training submissions. Monitoring must leave this empty."""

    def __init__(self) -> None:
        self.calls: list[Mapping[str, object]] = []

    def create_training_job(self, request: Mapping[str, object]) -> str:
        self.calls.append(dict(request))
        raise AssertionError("training client was called")


def test_model_performance_has_no_report_until_monitoring_runs(tmp_path: Path) -> None:
    client = _client(tmp_path)

    body = client.get("/metrics/model-performance").json()

    assert body["items"] == []
    assert body["monitoring"] is None


def test_quiet_window_is_healthy_and_published(tmp_path: Path) -> None:
    client, spy = _ready(tmp_path, QUIET_PRICES, approved_wape=0.10, recent_wape=0.12)
    dataset = client.get("/datasets").json()["items"][0]
    assert dataset["date_max"] == "2026-02-25"

    report = client.post("/admin/monitoring", params={"as_of": dataset["date_max"]})

    assert report.status_code == 200
    body = report.json()
    assert body["status"] == "HEALTHY"
    assert body["reasons"] == []
    assert body["dataset_age_days"] == 0
    assert body["as_of"] == "2026-02-25"
    assert body["date_max"] == "2026-02-25"
    assert body["baseline_end"] == "2026-01-28"
    assert body["recent_start"] == "2026-01-29"
    assert body["retrain_request_id"] is None
    assert body["wape_degradation"] == pytest.approx(0.2)
    assert {item["feature"] for item in body["numeric"]} == {"price", "units_sold"}
    assert {item["feature"] for item in body["frequencies"]} == {
        "promotion",
        "zero_demand",
        "stockout",
    }
    assert client.get("/admin/retrain-requests").json()["items"] == []
    published = client.get("/metrics/model-performance").json()
    assert published["monitoring"]["id"] == body["id"]
    assert published["monitoring"]["status"] == "HEALTHY"
    assert spy.calls == []


def test_moderate_price_shift_is_a_warning(tmp_path: Path) -> None:
    client, spy = _ready(tmp_path, WARNING_PRICES, approved_wape=0.10, recent_wape=0.11)

    report = client.post("/admin/monitoring", params={"as_of": "2026-02-25"})

    assert report.status_code == 200
    body = report.json()
    price = next(item for item in body["numeric"] if item["feature"] == "price")
    assert body["status"] == "WARNING"
    assert body["reasons"] == ["PSI for price is above 0.1 and at or below 0.2."]
    assert 0.1 < price["psi"] <= 0.2
    assert body["retrain_request_id"] is None
    assert client.get("/admin/retrain-requests").json()["items"] == []
    assert spy.calls == []


def test_retrain_recommendation_does_not_start_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, spy = _ready(tmp_path, QUIET_PRICES, approved_wape=0.10, recent_wape=0.13)
    calls: list[str] = []
    monkeypatch.setattr(ForecastService, "start_training", _blocked(calls, "start_training"))
    monkeypatch.setattr(ForecastService, "retrain", _blocked(calls, "retrain"))
    monkeypatch.setattr(ForecastService, "approve", _blocked(calls, "approve"))
    monkeypatch.setattr(
        "forecastops_ml.training.deepar_job.BotoTrainingJobClient.create_training_job",
        _blocked(calls, "create_training_job"),
    )
    runs_before = client.get("/training-runs").json()["items"]
    production_id = client.get("/models").json()["items"][0]["id"]

    report = client.post("/admin/monitoring", params={"as_of": "2026-02-25"})

    assert report.status_code == 200
    body = report.json()
    assert body["status"] == "RETRAIN_RECOMMENDED"
    assert body["reasons"] == [
        "Recent WAPE is more than 20% worse than the approved model's WAPE.",
    ]
    assert body["wape_degradation"] == pytest.approx(0.3)
    requests = client.get("/admin/retrain-requests").json()["items"]
    assert len(requests) == 1
    assert requests[0]["id"] == body["retrain_request_id"]
    assert requests[0]["status"] == "PENDING"
    assert requests[0]["training_run_id"] is None
    assert requests[0]["confirmed_by"] is None
    assert client.get("/training-runs").json()["items"] == runs_before
    assert client.get(f"/models/{production_id}").json()["status"] == "PRODUCTION"
    assert calls == []
    assert spy.calls == []


def test_dataset_age_uses_date_max_against_as_of(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, spy = _ready(tmp_path, QUIET_PRICES, approved_wape=0.10, recent_wape=0.10)
    calls: list[str] = []
    monkeypatch.setattr(ForecastService, "start_training", _blocked(calls, "start_training"))
    monkeypatch.setattr(ForecastService, "retrain", _blocked(calls, "retrain"))
    monkeypatch.setattr(ForecastService, "approve", _blocked(calls, "approve"))
    date_max = date.fromisoformat(client.get("/datasets").json()["items"][0]["date_max"])

    boundary = client.post(
        "/admin/monitoring",
        params={"as_of": (date_max + timedelta(days=30)).isoformat()},
    )
    stale = client.post(
        "/admin/monitoring",
        params={"as_of": (date_max + timedelta(days=31)).isoformat()},
    )

    assert boundary.status_code == 200
    assert boundary.json()["dataset_age_days"] == 30
    assert boundary.json()["status"] == "HEALTHY"
    assert boundary.json()["as_of"] == (date_max + timedelta(days=30)).isoformat()
    assert boundary.json()["date_max"] == date_max.isoformat()
    assert stale.status_code == 200
    assert stale.json()["dataset_age_days"] == (stale_as_of(date_max) - date_max).days == 31
    assert stale.json()["status"] == "RETRAIN_RECOMMENDED"
    assert stale.json()["reasons"] == ["Dataset age is greater than 30 days."]
    assert stale.json()["retrain_request_id"]
    assert client.get("/admin/retrain-requests").json()["items"][0]["training_run_id"] is None
    assert calls == []
    assert spy.calls == []


def stale_as_of(date_max: date) -> date:
    return date_max + timedelta(days=31)


def _blocked(calls: list[str], name: str):
    def reject(*_args: object, **_kwargs: object) -> None:
        calls.append(name)
        raise AssertionError(f"{name} was called")

    return reject


def _ready(
    tmp_path: Path,
    recent_prices: list[float],
    *,
    approved_wape: float,
    recent_wape: float,
) -> tuple[TestClient, SpyTrainingClient]:
    client = _client(tmp_path)
    directory = tmp_path / "data"
    _write_dataset(directory, recent_prices)
    created = client.post(
        "/datasets",
        json={"name": "retail-monitor", "source": "synthetic", "uri": str(directory)},
    )
    assert created.status_code == 202
    dataset_id = str(created.json()["job_id"])
    dataset = client.get(f"/datasets/{dataset_id}").json()
    _production(
        client,
        dataset_id,
        dataset["version"],
        approved_wape=approved_wape,
        recent_wape=recent_wape,
    )
    spy = SpyTrainingClient()
    client.app.state.training_client = spy
    return client, spy


def _write_dataset(directory: Path, recent_prices: list[float]) -> None:
    directory.mkdir()
    days = [START + timedelta(days=offset) for offset in range(56)]
    count = len(days)
    pq.write_table(
        pa.table(
            {
                "date": pa.array(days, type=pa.date32()),
                "store_id": ["s1"] * count,
                "sku_id": ["k1"] * count,
                "category_id": ["c1"] * count,
                "units_sold": pa.array([4] * count, type=pa.int64()),
                "price": pa.array(QUIET_PRICES + recent_prices, type=pa.float64()),
                "promotion": pa.array([False] * count, type=pa.bool_()),
                "stockout": pa.array([False] * count, type=pa.bool_()),
            }
        ),
        directory / "observations.parquet",
    )
    pq.write_table(
        pa.table({"store_id": ["s1"], "store_region": ["north"]}),
        directory / "stores.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "sku_id": ["k1"],
                "category_id": ["c1"],
                "product_brand": ["brand"],
                "product_lifecycle": ["mature"],
            }
        ),
        directory / "skus.parquet",
    )
    pq.write_table(
        pa.table({"category_id": ["c1"], "category_name": ["Grocery"]}),
        directory / "categories.parquet",
    )


def _production(
    client: TestClient,
    dataset_id: str,
    version: str,
    *,
    approved_wape: float,
    recent_wape: float,
) -> None:
    now = datetime.now(UTC)
    training_id = str(uuid4())
    model_id = str(uuid4())
    forecast_id = str(uuid4())
    session = client.app.state.session_factory()
    try:
        session.add(
            TrainingRunRow(
                id=training_id,
                dataset_id=dataset_id,
                dataset_version=version,
                model_family="seasonal_naive",
                configuration={},
                status="COMPLETED",
                started_at=now,
                finished_at=now,
                artifact_uri="",
                metrics={"wape": approved_wape},
                git_sha="",
                pipeline_execution_arn="",
                error_message=None,
                created_at=now,
            )
        )
        session.add(
            ModelVersionRow(
                id=model_id,
                model_family="seasonal_naive",
                version="1",
                training_run_id=training_id,
                dataset_id=dataset_id,
                dataset_version=version,
                registry_arn="",
                registry_status="",
                status="PRODUCTION",
                metrics={"wape": approved_wape},
                approved_at=now,
                approved_by="analyst-1",
                rejected_by=None,
                rejection_reason=None,
                created_at=now,
            )
        )
        session.add(
            ForecastRunRow(
                id=forecast_id,
                model_version_id=model_id,
                dataset_id=dataset_id,
                dataset_version=version,
                horizon=7,
                granularity="day",
                status="SUCCEEDED",
                output_uri="",
                error_message=None,
                idempotency_key=None,
                created_at=now,
            )
        )
        session.add(
            ForecastErrorEvaluationRow(
                id=str(uuid4()),
                forecast_run_id=forecast_id,
                compared_points=4,
                wape=recent_wape,
                bias=0.0,
                created_at=now,
            )
        )
        session.commit()
    finally:
        session.close()


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
