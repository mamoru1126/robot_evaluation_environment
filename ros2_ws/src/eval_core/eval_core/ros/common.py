"""Message conversion helpers shared by the ROS nodes."""
from __future__ import annotations

import json
import math
from typing import Any, Tuple

from ..geometry import yaw_from_quaternion
from ..storage import _sanitize

# Topic / service names (keep in sync with eval_gazebo bridge generation)
CMD_VEL = "/cmd_vel"                       # gated command sent to Gazebo (DiffDrive)
CONTROLLER_CMD_VEL = "/controller/cmd_vel"  # raw output of the selected controller
GT_ODOM = "/ground_truth/odom"             # robot ground-truth pose (world frame) + body twist
WHEEL_ODOM = "/odom"
SCAN = "/scan"
PLANNER_COMMAND = "/local_planner/command"  # std_msgs/String (JSON)
PLANNER_STATUS = "/local_planner/status"    # std_msgs/String (JSON)
PLANNER_DEBUG = "/local_planner/debug"      # std_msgs/String (JSON): predicted trajectory etc.
PLANNER_MARKERS = "/local_planner/markers"  # visualization_msgs/MarkerArray (RViz)
EVAL_PATH = "/eval/path"                    # nav_msgs/Path of the current trial (transient local)
EVAL_TRIAL = "/eval/trial"                  # std_msgs/String (JSON): trial info for external controllers


def human_cmd_topic(i: int) -> str:
    return f"/humans/human_{i + 1}/cmd_vel"


def human_odom_topic(i: int) -> str:
    return f"/humans/human_{i + 1}/odom"


def set_pose_service(world: str) -> str:
    return f"/world/{world}/set_pose"


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def odom_to_tuple(msg) -> Tuple[float, float, float, float, float, float, float]:
    """(t, x, y, yaw, v_forward, w, v_lateral) from nav_msgs/Odometry (twist in child frame)."""
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
    tw = msg.twist.twist
    return (stamp_to_sec(msg.header.stamp), p.x, p.y, yaw, tw.linear.x, tw.angular.z, tw.linear.y)


def body_to_world(vx_b: float, vy_b: float, yaw: float) -> Tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return c * vx_b - s * vy_b, s * vx_b + c * vy_b


def world_to_body(vx_w: float, vy_w: float, yaw: float) -> Tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return c * vx_w + s * vy_w, -s * vx_w + c * vy_w


def dumps(data: Any) -> str:
    return json.dumps(_sanitize(data), separators=(",", ":"))
