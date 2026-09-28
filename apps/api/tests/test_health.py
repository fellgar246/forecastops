"""Health endpoint tests."""

from fastapi.testclient import TestClient

from forecastops_api.main import app
from forecastops_api.settings import get_settings


def test_health_reports_local_profile_with_cloud_flags_off() -> None:
    get_settings.cache_clear()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["execution_mode"] == "local"
    assert body["aws_enabled"] is False
    assert body["bedrock_enabled"] is False
    assert body["sagemaker_enabled"] is False
    assert body["training_enabled"] is True
    assert body["max_forecast_horizon_days"] == 90
    assert body["max_training_jobs_per_day"] == 2


def test_aws_health_reports_each_cloud_integration_disabled() -> None:
    get_settings.cache_clear()
    client = TestClient(app)

    response = client.get("/health/aws")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "aws_enabled": False,
        "aws_ml_enabled": False,
        "bedrock_enabled": False,
        "sagemaker_enabled": False,
        "online_inference": False,
    }
