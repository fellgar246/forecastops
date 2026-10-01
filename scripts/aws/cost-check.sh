#!/usr/bin/env bash
# Report the monthly budget and its alarms. This command does not start training.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

environment="$(aws_environment)"
aws_prepare_cli

account="$(aws sts get-caller-identity --query Account --output text)"
budget="forecastops-${environment}-monthly"

echo "Budget ${budget} in account ${account}"
aws budgets describe-budget --account-id "${account}" --budget-name "${budget}"
echo "Notifications for ${budget}"
aws budgets describe-notifications-for-budget --account-id "${account}" --budget-name "${budget}"
echo "Budget alarms reported. Training was not started."
