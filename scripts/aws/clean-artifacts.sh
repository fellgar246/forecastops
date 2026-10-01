#!/usr/bin/env bash
# Remove leftover object versions from the project buckets.
# Destroy does not delete those objects.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

root="$(aws_root)"
environment="$(aws_environment)"
prefix="$(aws_name_prefix)"
aws_prepare_cli

buckets=(
  "${prefix}-${environment}-data"
  "${prefix}-${environment}-artifacts"
  "${prefix}-${environment}-forecasts"
)

for bucket in "${buckets[@]}"; do
  echo "Removing leftover objects from ${bucket}"
  listing="$(aws s3api list-object-versions --bucket "${bucket}" --output json)"
  batches="$(printf '%s' "${listing}" | python3 "${root}/scripts/aws/delete_versions.py")"
  if [[ -z "${batches}" ]]; then
    echo "${bucket} has no object versions."
    continue
  fi
  while IFS= read -r batch; do
    [[ -z "${batch}" ]] && continue
    payload="$(mktemp)"
    printf '%s\n' "${batch}" > "${payload}"
    aws s3api delete-objects --bucket "${bucket}" --delete "file://${payload}"
    rm -f "${payload}"
  done <<< "${batches}"
done
