#!/usr/bin/env bash
# Follow the logs.  ./scripts/logs.sh [sim|eval|gui|rviz]
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose --profile gui --profile rviz logs -f --tail=200 "$@"
