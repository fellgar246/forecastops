#!/usr/bin/env bash
# Print Terraform outputs for the selected environment. This does not change resources.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

root="$(aws_root)"
environment="$(aws_environment)"
directory="$(aws_environment_dir "${root}" "${environment}")"

terraform -chdir="${directory}" init -backend=false -input=false
if ! terraform -chdir="${directory}" output; then
  echo "No outputs yet. Run make aws-deploy before make aws-status." >&2
  exit 1
fi
