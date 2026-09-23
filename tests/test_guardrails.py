"""Guardrails for the default environment and product wording."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFRA = ROOT / "infra"

FORBIDDEN_RESOURCES = (
    "aws_sagemaker_training_job",
    "aws_sagemaker_endpoint",
    "aws_sagemaker_endpoint_configuration",
    "aws_sagemaker_notebook_instance",
    "aws_sagemaker_domain",
    "aws_sagemaker_app",
    "aws_sagemaker_hyperparameter_tuning_job",
    "aws_cloudwatch_event_rule",
    "aws_cloudwatch_event_target",
    "aws_scheduler_schedule",
)

REQUIRED_TAGS = (
    "Project",
    "Environment",
    "Owner",
    "ManagedBy",
    "CostCenter",
    "AutoCleanup",
)

PRODUCT_SUFFIXES = {
    ".css",
    ".csv",
    ".example",
    ".ini",
    ".mako",
    ".md",
    ".py",
    ".sh",
    ".tf",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".terraform",
    ".venv",
    "_origen",
    "node_modules",
}


def test_default_plan_has_no_training_job_endpoint_notebook_or_schedule() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in INFRA.rglob("*.tf"))
    for resource in FORBIDDEN_RESOURCES:
        assert resource not in combined


def test_environment_flags_default_off() -> None:
    for name in ("dev", "demo"):
        text = (INFRA / "environments" / name / "variables.tf").read_text(encoding="utf-8")
        for flag in ("enable_serverless_endpoint", "enable_bedrock", "enable_schedules"):
            pattern = rf'variable "{flag}"[\s\S]*?default\s*=\s*false'
            assert re.search(pattern, text), f"{name} {flag}"
        assert "enable_serverless_endpoint = true" not in text
        assert "enable_bedrock = true" not in text
        assert "enable_schedules = true" not in text


def test_budget_and_log_retention() -> None:
    budget = (INFRA / "modules" / "budget" / "main.tf").read_text(encoding="utf-8")
    assert re.search(r'limit_amount\s*=\s*"5"', budget)
    for amount in (1, 3, 5):
        assert re.search(rf"threshold\s*=\s*{amount}\b", budget)
    assert "FORECASTED" in budget

    for name in ("dev", "demo"):
        text = (INFRA / "environments" / name / "main.tf").read_text(encoding="utf-8")
        assert 'var.environment == "demo" ? 14 : 7' in text
        variables = (INFRA / "environments" / name / "variables.tf").read_text(encoding="utf-8")
        assert re.search(rf'variable "environment"[\s\S]*?default\s*=\s*"{name}"', variables)


def test_resources_use_required_tags() -> None:
    for name in ("dev", "demo"):
        text = (INFRA / "environments" / name / "providers.tf").read_text(encoding="utf-8")
        for tag in REQUIRED_TAGS:
            assert tag in text
        assert "ml-demand-forecasting" in text
        assert "portfolio" in text
        assert "terraform" in text
        assert "learning" in text
        assert 'AutoCleanup = "true"' in text
        assert "Environment = var.environment" in text


def test_architecture_records_exist() -> None:
    adr_dir = ROOT / "docs" / "adr"
    assert len(list(adr_dir.glob("*.md"))) == 6
    assert (ROOT / "docs" / "architecture" / "overview.md").is_file()


def test_product_files_do_not_name_planning_directories() -> None:
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.name in {"package-lock.json", "uv.lock", "test_guardrails.py"}:
            continue
        if path.suffix not in PRODUCT_SUFFIXES and path.name not in {"Makefile", "Dockerfile"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "_origen" in text or "specs/" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
