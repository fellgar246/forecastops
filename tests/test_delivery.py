"""Delivery workflows stay manual for training and carry no static cloud keys."""

import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
INFRA = ROOT / "infra"
SMOKE = ROOT / "scripts" / "aws" / "smoke.sh"

REQUIRED_WORKFLOWS = (
    "ci.yml",
    "terraform-plan.yml",
    "deploy-dev.yml",
    "train-model.yml",
    "destroy-demo.yml",
)

FORBIDDEN_KEY_SNIPPETS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "aws_access_key_id",
    "aws_secret_access_key",
    "AKIA",
)

OIDC_WORKFLOWS = (
    "terraform-plan.yml",
    "deploy-dev.yml",
    "destroy-demo.yml",
    "train-model.yml",
)


def test_training_workflow_is_dispatch_only() -> None:
    text = (WORKFLOWS / "train-model.yml").read_text(encoding="utf-8")
    assert _on_keys(text) == {"workflow_dispatch"}


def test_workflow_triggers_match_the_delivery_rules() -> None:
    assert set(REQUIRED_WORKFLOWS) == {path.name for path in WORKFLOWS.glob("*.yml")}
    assert "pull_request" in _on_keys((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))
    assert _on_keys((WORKFLOWS / "deploy-dev.yml").read_text(encoding="utf-8")) == {"push"}
    assert _on_keys((WORKFLOWS / "destroy-demo.yml").read_text(encoding="utf-8")) == {
        "workflow_dispatch"
    }
    plan = _on_keys((WORKFLOWS / "terraform-plan.yml").read_text(encoding="utf-8"))
    assert "pull_request" in plan
    assert "workflow_dispatch" not in plan


def test_workflows_have_no_static_cloud_keys() -> None:
    offenders: list[str] = []
    for name in REQUIRED_WORKFLOWS:
        text = (WORKFLOWS / name).read_text(encoding="utf-8")
        for snippet in FORBIDDEN_KEY_SNIPPETS:
            if snippet in text:
                offenders.append(f"{name}:{snippet}")
    assert offenders == []


def test_cloud_workflows_assume_the_deploy_role_with_oidc() -> None:
    for name in OIDC_WORKFLOWS:
        text = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert "role/GitHubDeployRole" in text
        assert "aws-actions/configure-aws-credentials" in text
        assert "id-token: write" in text


def test_ci_invokes_leakage_tests() -> None:
    text = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    assert "ml/tests/test_features.py" in text
    assert "Leakage tests" in text


def test_main_deploy_does_not_start_training_or_batch_inference() -> None:
    text = (WORKFLOWS / "deploy-dev.yml").read_text(encoding="utf-8")
    assert "make aws-deploy" in text
    assert "make aws-smoke" in text
    for snippet in (
        "start-pipeline",
        "train-deepar",
        "refresh-forecast",
        "CreateTrainingJob",
        "CreateTransformJob",
        "batch-transform",
    ):
        assert snippet not in text
    plan = (WORKFLOWS / "terraform-plan.yml").read_text(encoding="utf-8")
    assert "aws-deploy" not in plan
    assert "aws-destroy" not in plan
    destroy = (WORKFLOWS / "destroy-demo.yml").read_text(encoding="utf-8")
    assert "ENV=demo make aws-destroy" in destroy
    assert "aws-deploy" not in destroy


def test_deploy_role_is_created_once() -> None:
    module = (INFRA / "modules" / "github_oidc" / "main.tf").read_text(encoding="utf-8")
    assert 'name               = "GitHubDeployRole"' in module
    assert "AssumeRoleWithWebIdentity" in module
    dev = (INFRA / "environments" / "dev" / "main.tf").read_text(encoding="utf-8")
    demo = (INFRA / "environments" / "demo" / "main.tf").read_text(encoding="utf-8")
    assert "create_deploy_role   = true" in dev
    assert "create_deploy_role   = false" in demo


def test_smoke_script_lists_only_health_routes() -> None:
    text = SMOKE.read_text(encoding="utf-8")
    assert "/health" in text
    assert "/health/aws" in text
    for snippet in ("/forecasts", "/training", "/metrics", "start-pipeline", "sagemaker"):
        assert snippet not in text


def test_smoke_calls_only_health_routes() -> None:
    seen: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            seen.append(self.path)
            body = b'{"status":"healthy","aws_enabled":false}\n'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        env = os.environ.copy()
        env["SMOKE_BASE_URL"] = f"http://127.0.0.1:{port}"
        completed = subprocess.run(
            ["bash", str(SMOKE)],
            check=True,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
    finally:
        server.shutdown()
    assert seen == ["/health", "/health/aws"]
    assert "Smoke checks finished at the health routes." in completed.stdout


def _on_keys(text: str) -> set[str]:
    keys: set[str] = set()
    in_on = False
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        stripped = line.split("#", 1)[0].rstrip()
        if not in_on:
            if stripped == "on:":
                in_on = True
            continue
        if stripped == "":
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            break
        if indent == 2 and stripped.endswith(":"):
            keys.add(stripped[:-1].strip())
    return keys
