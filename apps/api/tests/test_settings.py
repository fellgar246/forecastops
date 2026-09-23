"""Settings contract tests."""

import pytest
from envfile import ROOT, parse_env_file
from pydantic import ValidationError

from forecastops_api.settings import ExecutionMode, Settings, get_settings, load_settings

CONTRACT = {
    "EXECUTION_MODE": "local",
    "AWS_ENABLED": "false",
    "BEDROCK_ENABLED": "false",
    "SAGEMAKER_ENABLED": "false",
    "ONLINE_INFERENCE": "false",
    "AI_ENABLED": "true",
    "TRAINING_ENABLED": "true",
    "AWS_ML_ENABLED": "false",
    "MAX_BEDROCK_CALLS_PER_DAY": "30",
    "MAX_BEDROCK_INPUT_TOKENS": "5000",
    "MAX_BEDROCK_OUTPUT_TOKENS": "700",
    "MAX_TRAINING_JOBS_PER_DAY": "2",
    "MAX_TRAINING_RUNTIME_MINUTES": "45",
    "ALLOW_GPU_TRAINING": "false",
    "MAX_BATCH_INFERENCE_JOBS_PER_DAY": "5",
    "MAX_DATASET_ROWS_DEMO": "2000000",
    "MAX_FORECAST_HORIZON_DAYS": "90",
    "RANDOM_SEED": "20260921",
}


def test_example_environment_matches_the_contract() -> None:
    values = parse_env_file(ROOT / ".env.example")
    for key, expected in CONTRACT.items():
        assert values[key] == expected


def test_local_example_leaves_cloud_flags_off(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in parse_env_file(ROOT / ".env.example").items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()

    settings = Settings(_env_file=None)

    assert settings.execution_mode is ExecutionMode.LOCAL
    assert settings.aws_enabled is False
    assert settings.bedrock_enabled is False
    assert settings.sagemaker_enabled is False
    assert settings.online_inference is False
    assert settings.aws_ml_enabled is False
    assert settings.ai_enabled is True
    assert settings.training_enabled is True
    assert settings.allow_gpu_training is False
    assert settings.random_seed == 20260921


def test_unknown_execution_mode_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in parse_env_file(ROOT / ".env.example").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("EXECUTION_MODE", "cluster")

    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=None)

    assert "execution_mode" in str(caught.value)


def test_missing_required_value_fails_in_english(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in CONTRACT:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(SystemExit) as caught:
        load_settings()

    message = str(caught.value)
    assert "invalid" in message.lower()
    assert "execution_mode" in message
