#!/usr/bin/env bash
# Destroy disposable resources in the selected environment.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

root="$(aws_root)"
environment="$(aws_environment)"
directory="$(aws_environment_dir "${root}" "${environment}")"

terraform -chdir="${directory}" init -backend=false -input=false
if ! terraform -chdir="${directory}" destroy -input=false -auto-approve; then
  echo "Destroy failed. Run make aws-clean-artifacts to remove leftover object versions, then run make aws-destroy again." >&2
  exit 1
fi
echo "Disposable resources for ${environment} were destroyed."
