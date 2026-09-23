#!/usr/bin/env bash
set -euo pipefail

# Prints the project cost ceilings. This command does not call cloud APIs.

cat <<'EOF'
Local development: about $0 cloud spend
Occasional demo: at most $5 USD / month
Temporary showcase: hard internal target at most $10 USD / month
Training: at most 2 jobs per day, 45 minutes, CPU only
Bedrock: at most 30 calls per day
Batch inference: at most 5 jobs per day
EOF
