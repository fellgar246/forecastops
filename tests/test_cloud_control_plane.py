"""Guardrails for the cloud control plane commands and IAM."""

import importlib.util
import os
import re
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFRA = ROOT / "infra"
AWS = ROOT / "scripts" / "aws"

FORBIDDEN_SNIPPETS = (
    "aws_iam_access_key",
    "aws_iam_user",
    "aws_access_key_id",
    "aws_secret_access_key",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)


def test_configuration_rejects_embedded_access_keys() -> None:
    offenders: list[str] = []
    for path in _configuration_files():
        text = path.read_text(encoding="utf-8")
        for snippet in FORBIDDEN_SNIPPETS:
            if snippet in text:
                offenders.append(f"{path.relative_to(ROOT)}:{snippet}")
    assert offenders == []


def test_default_deploy_has_no_realtime_endpoint() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in INFRA.rglob("*.tf"))
    assert "aws_sagemaker_endpoint" not in combined
    assert "aws_sagemaker_endpoint_configuration" not in combined
    for name in ("dev", "demo"):
        text = (INFRA / "environments" / name / "variables.tf").read_text(encoding="utf-8")
        assert 'variable "enable_serverless_endpoint"' in text
        assert "default     = false" in text
        main = (INFRA / "environments" / name / "main.tf").read_text(encoding="utf-8")
        assert "enable_serverless_endpoint = var.enable_serverless_endpoint" in main
        assert re.search(r'SAGEMAKER_ENABLED\s*=\s*"false"', main)
        assert re.search(r'ONLINE_INFERENCE\s*=\s*"false"', main)


def test_purpose_roles_are_resource_scoped() -> None:
    iam = (INFRA / "modules" / "iam" / "main.tf").read_text(encoding="utf-8")
    for role in ("api", "pipeline", "training", "inference", "explanation"):
        assert f'resource "aws_iam_role" "{role}"' in iam
    deploy = (INFRA / "modules" / "github_oidc" / "main.tf").read_text(encoding="utf-8")
    assert 'resource "aws_iam_role" "deploy"' in deploy
    assert "AssumeRoleWithWebIdentity" in deploy
    for name in ("pipeline.json", "training.json", "inference.json"):
        text = (INFRA / "policies" / name).read_text(encoding="utf-8")
        assert '"Resource": "*"' not in text
    api_policy = (INFRA / "policies" / "api.json").read_text(encoding="utf-8")
    assert "${metadata_table_arn}" in api_policy
    assert "${artifacts_bucket_arn}" in api_policy
    assert "${forecasts_bucket_arn}" in api_policy


def test_cloud_commands_select_dev_unless_demo_is_requested(tmp_path: Path) -> None:
    log = tmp_path / "commands.log"
    env = _fake_path(tmp_path, log)
    subprocess.run(["bash", str(AWS / "deploy.sh")], check=True, env=env, cwd=ROOT)
    subprocess.run(["bash", str(AWS / "status.sh")], check=True, env=env, cwd=ROOT)
    subprocess.run(["bash", str(AWS / "destroy.sh")], check=True, env=env, cwd=ROOT)
    text = log.read_text(encoding="utf-8")
    assert "infra/environments/dev" in text
    assert "infra/environments/demo" not in text
    assert "apply" in text
    assert "use_packaged_api=true" in text
    assert "output" in text
    assert "destroy" in text

    demo = tmp_path / "demo.log"
    demo_env = _fake_path(tmp_path, demo)
    demo_env["ENV"] = "demo"
    subprocess.run(["bash", str(AWS / "deploy.sh")], check=True, env=demo_env, cwd=ROOT)
    assert "infra/environments/demo" in demo.read_text(encoding="utf-8")


def test_cost_check_reports_budget_alarms_without_training(tmp_path: Path) -> None:
    log = tmp_path / "aws.log"
    env = _fake_path(tmp_path, log)
    completed = subprocess.run(
        ["bash", str(AWS / "cost-check.sh")],
        check=True,
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    text = log.read_text(encoding="utf-8")
    assert "sts get-caller-identity" in text
    assert "budgets describe-budget" in text
    assert "budgets describe-notifications-for-budget" in text
    assert "forecastops-dev-monthly" in text
    assert "sagemaker" not in text.lower()
    assert "Training was not started." in completed.stdout
    script = (AWS / "cost-check.sh").read_text(encoding="utf-8")
    assert "sagemaker" not in script.lower()
    assert "training-job" not in script.lower()


def test_clean_artifacts_deletes_leftover_versions(tmp_path: Path) -> None:
    log = tmp_path / "aws.log"
    env = _fake_path(tmp_path, log)
    env["NAME_PREFIX"] = "forecastops"
    subprocess.run(["bash", str(AWS / "clean-artifacts.sh")], check=True, env=env, cwd=ROOT)
    text = log.read_text(encoding="utf-8")
    for name in ("data", "artifacts", "forecasts"):
        assert f"forecastops-dev-{name}" in text
    assert "delete-objects" in text
    assert "list-object-versions" in text


def test_version_delete_batches_cover_versions_and_markers() -> None:
    module = _load_delete_versions()
    batches = module.version_delete_batches(
        {
            "Versions": [{"Key": "raw/a.csv", "VersionId": "1"}],
            "DeleteMarkers": [{"Key": "raw/a.csv", "VersionId": "2"}],
        }
    )
    assert batches == [
        {
            "Objects": [
                {"Key": "raw/a.csv", "VersionId": "1"},
                {"Key": "raw/a.csv", "VersionId": "2"},
            ],
            "Quiet": True,
        }
    ]
    assert module.version_delete_batches({}) == []


def test_unknown_environment_is_rejected() -> None:
    completed = subprocess.run(
        ["bash", str(AWS / "status.sh")],
        cwd=ROOT,
        env={**os.environ, "ENV": "prod"},
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "dev or demo" in completed.stderr


def _configuration_files() -> list[Path]:
    files: list[Path] = []
    for path in list(INFRA.rglob("*")) + list(AWS.rglob("*")):
        if path.is_file() and path.suffix in {".tf", ".json", ".py", ".sh"}:
            files.append(path)
    return files


def _fake_path(tmp_path: Path, log: Path) -> dict[str, str]:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    _write_executable(bindir / "terraform", _terraform_stub(log))
    _write_executable(bindir / "aws", _aws_stub(log))
    env = os.environ.copy()
    env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
    env["SKIP_API_PACKAGE"] = "1"
    env.pop("ENV", None)
    return env


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _terraform_stub(log: Path) -> str:
    return f"""#!/bin/sh
printf '%s\\n' "$*" >> "{log}"
exit 0
"""


def _aws_stub(log: Path) -> str:
    return f"""#!/bin/sh
printf '%s\\n' "$*" >> "{log}"
case "$*" in
  *"get-caller-identity"*)
    printf '123456789012\\n'
    ;;
  *"list-object-versions"*)
    printf '%s\\n' '{{"Versions":[{{"Key":"raw/a.csv","VersionId":"v1"}}]}}'
    ;;
  *)
    printf '{{}}\\n'
    ;;
esac
exit 0
"""


def _load_delete_versions():
    spec = importlib.util.spec_from_file_location(
        "delete_versions",
        AWS / "delete_versions.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
