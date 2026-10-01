#!/usr/bin/env bash
# Apply the selected environment. Dev is the default. Demo requires ENV=demo.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

root="$(aws_root)"
environment="$(aws_environment)"
directory="$(aws_environment_dir "${root}" "${environment}")"

# SKIP_API_PACKAGE=1 reuses an existing package under infra/modules/lambda/build/package.
if [[ "${SKIP_API_PACKAGE:-}" != "1" ]]; then
  "${root}/scripts/aws/package-api.sh"
fi

terraform -chdir="${directory}" init -backend=false -input=false
terraform -chdir="${directory}" apply -input=false -auto-approve -var="use_packaged_api=true"
