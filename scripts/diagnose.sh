#!/usr/bin/env bash
# Print the state of the running stack (topics, rates, services) to help debugging.
set -uo pipefail
cd "$(dirname "$0")/.."
docker compose exec eval /entrypoint.sh bash -c '
echo "=== ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}  discovery=${ROS_AUTOMATIC_DISCOVERY_RANGE:-}"
echo "=== nodes";    timeout 10 ros2 node list
echo "=== topics";   timeout 10 ros2 topic list -t
echo "=== services (set_pose)"; timeout 10 ros2 service list | grep -i set_pose
for t in /clock /ground_truth/odom /scan /scan_gz /humans/human_1/odom; do
  echo "=== rate $t"
  timeout 6 ros2 topic hz "$t" --window 20 2>/dev/null | tail -n 2
done
echo "=== local planner status"; timeout 5 ros2 topic echo --once /local_planner/status
echo "=== web";      curl -s "http://localhost:${WEB_PORT:-8080}/api/status" | head -c 600; echo
'
