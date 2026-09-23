#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

terraform fmt -check -recursive infra

for environment in dev demo; do
  directory="infra/environments/${environment}"
  terraform -chdir="${directory}" init -backend=false -input=false
  terraform -chdir="${directory}" validate
done
