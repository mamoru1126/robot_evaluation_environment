#!/usr/bin/env bash
# Run the unit / integration tests inside the image (no Gazebo needed).
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose run --rm --no-deps eval bash -c \
  "cd /ws/src && python3 -m pytest -q eval_core/test eval_gazebo/test"
