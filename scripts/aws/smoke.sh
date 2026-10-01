#!/usr/bin/env bash
# Call the deployed health routes and stop. This does not start training or inference.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

root="$(aws_root)"
base="${SMOKE_BASE_URL:-}"
if [[ -z "${base}" ]]; then
  environment="$(aws_environment)"
  directory="$(aws_environment_dir "${root}" "${environment}")"
  base="$(terraform -chdir="${directory}" output -raw api_endpoint)"
fi

python3 - "${base}" /health /health/aws <<'PY'
import json
import sys
import urllib.error
import urllib.request

base = sys.argv[1].strip().rstrip("/")
paths = sys.argv[2:]
if paths != ["/health", "/health/aws"]:
    raise SystemExit("Smoke checks only the health routes.")

for path in paths:
    url = f"{base}{path}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{path} returned {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"{path} could not be reached: {exc.reason}.") from exc
    if status != 200:
        raise SystemExit(f"{path} returned {status}.")
    print(raw)
    if path == "/health":
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit("The health route did not return JSON.") from exc
        if body.get("status") != "healthy":
            raise SystemExit("The health route did not report healthy.")

print("Smoke checks finished at the health routes.")
PY
