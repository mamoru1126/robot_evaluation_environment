#!/usr/bin/env bash
# Stop and remove all containers of the evaluation environment (results are kept).
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose --profile gui --profile rviz down
