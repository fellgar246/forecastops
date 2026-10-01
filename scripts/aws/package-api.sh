#!/usr/bin/env bash
# Build the Lambda package for the HTTP API. Wheels target the Lambda runtime.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dest="${root}/infra/modules/lambda/build/package"
requirements="$(mktemp)"
trap 'rm -f "${requirements}"' EXIT

rm -rf "${dest}"
mkdir -p "${dest}"

uv export --project "${root}" --frozen --no-dev --no-emit-workspace --output-file "${requirements}"
uv pip install \
  --python-platform x86_64-manylinux2014 \
  --python-version 3.12 \
  --target "${dest}" \
  --requirement "${requirements}"
uv pip install \
  --no-deps \
  --target "${dest}" \
  "${root}/packages/contracts" \
  "${root}/ml" \
  "${root}/apps/api"

cp "${root}/infra/modules/lambda/src/api_handler.py" "${dest}/handler.py"
find "${dest}" -type d -name '__pycache__' -prune -exec rm -rf {} +
echo "Packaged the API at ${dest}"
