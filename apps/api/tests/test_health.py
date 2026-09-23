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
