#!/usr/bin/env bash
# Open a shell (ROS 2 sourced) in a running container.  ./scripts/shell.sh [eval|sim]
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose exec "${1:-eval}" /entrypoint.sh bash
