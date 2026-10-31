#!/usr/bin/env bash
# Build the Docker image (ROS 2 Jazzy + Gazebo Harmonic + evaluation environment).
# Extra arguments are passed to "docker compose build" (e.g. --no-cache).
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose build --pull "$@" sim
echo "built image: robot-eval:jazzy"
