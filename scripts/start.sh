#!/usr/bin/env bash
# Start the evaluation environment (Gazebo + evaluation server) in the background.
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
Usage: ./scripts/start.sh [options]

  --gpu             use the NVIDIA GPU for Gazebo rendering (NVIDIA Container Toolkit)
  --gui             also open the Gazebo GUI (X11)
  --rviz            also open RViz2 (X11)
  --virtual-lidar   do not render the LiDAR in Gazebo; ray-cast a virtual scan instead
  --gpu-lidar       always use Gazebo's gpu_lidar (no virtual fallback)
  --2d-only         only the evaluation server with the built-in 2D simulator (no Gazebo)
  --port N          web UI port (default 8080)

Environment: ROS_DOMAIN_ID (default 87), GZ_PARTITION (default robot_eval)
EOF
}

FILES=(-f docker-compose.yml)
PROFILES=()
SERVICES=(sim eval)
UPFLAGS=()
NEED_X=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) FILES+=(-f docker-compose.gpu.yml) ;;
    --gui) PROFILES+=(--profile gui); SERVICES+=(gui); NEED_X=1 ;;
    --rviz) PROFILES+=(--profile rviz); SERVICES+=(rviz); NEED_X=1 ;;
    --virtual-lidar) export EVAL_LIDAR=virtual ;;
    --gpu-lidar) export EVAL_LIDAR=gpu ;;
    --2d-only) SERVICES=(eval); UPFLAGS+=(--no-deps); export EVAL_BACKENDS=sim2d ;;
    --port) export WEB_PORT="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1"; usage; exit 1 ;;
  esac
  shift
done

mkdir -p results
if ! docker image inspect robot-eval:jazzy >/dev/null 2>&1; then
  echo "image robot-eval:jazzy not found -> building it first (this takes a while)"
  ./scripts/build.sh
fi
if [[ -n "$NEED_X" ]]; then
  if command -v xhost >/dev/null 2>&1; then
    xhost +local: >/dev/null || true
  else
    echo "warning: xhost not found; the GUI may not be able to open a window"
  fi
fi

docker compose "${FILES[@]}" "${PROFILES[@]}" up -d "${UPFLAGS[@]}" "${SERVICES[@]}"

echo ""
echo "Web UI : http://localhost:${WEB_PORT:-8080}"
echo "Logs   : ./scripts/logs.sh"
echo "Stop   : ./scripts/stop.sh"
