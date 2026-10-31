#!/bin/bash
# Source ROS 2 and the evaluation workspace, then run the given command.
set -e
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
if [ -f /ws/install/setup.bash ]; then
  source /ws/install/setup.bash
fi
exec "$@"
