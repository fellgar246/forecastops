"""Cloud auth resources stay scoped to one user pool and one tenant claim."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFRA = ROOT / "infra"


def test_cloud_environments_require_auth_and_a_user_pool() -> None:
    for name in ("dev", "demo"):
        main = (INFRA / "environments" / name / "main.tf").read_text(encoding="utf-8")
        assert 'source = "../../modules/cognito"' in main
        assert re.search(r'AUTH_ENABLED\s+=\s+"true"', main)
        assert "module.cognito.user_pool_id" in main
        assert "module.cognito.app_client_id" in main


def test_user_pool_copies_the_tenant_claim_and_blocks_signup() -> None:
    main = (INFRA / "modules" / "cognito" / "main.tf").read_text(encoding="utf-8")
    handler = (INFRA / "modules" / "cognito" / "src" / "pre_token.py").read_text(encoding="utf-8")
    assert "allow_admin_create_user_only = true" in main
    assert re.search(r'name\s+=\s+"tenant_id"', main)
    assert 'lambda_version = "V2_0"' in main
    assert "generate_secret = false" in main
    assert '"tenant_id": tenant.strip()' in handler
    assert "custom:tenant_id" in handler
