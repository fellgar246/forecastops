#!/usr/bin/env bash
# Shared selection for the cloud commands. Dev is the default target.

aws_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

aws_environment() {
  local selected="${ENV:-dev}"
  case "${selected}" in
    dev|demo)
      printf '%s\n' "${selected}"
      ;;
    *)
      echo "ENV must be dev or demo." >&2
      return 1
      ;;
  esac
}

aws_environment_dir() {
  local root="$1"
  local environment="$2"
  printf '%s\n' "${root}/infra/environments/${environment}"
}

aws_name_prefix() {
  printf '%s\n' "${NAME_PREFIX:-${TF_VAR_name_prefix:-forecastops}}"
}

aws_prepare_cli() {
  export AWS_DEFAULT_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
}

# Record the time of a finished cleanup so the cost page can show it.
record_cleanup_stamp() {
  local root="$1"
  local stamp
  stamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  mkdir -p "${root}/var/cost"
  printf '%s\n' "${stamp}" > "${root}/var/cost/last_cleanup_at"
  echo "Recorded last cleanup at ${stamp}"
}
