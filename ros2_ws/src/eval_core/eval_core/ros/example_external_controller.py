"""Minimal *external* controller node (example for custom_controllers/*.yaml).

It only uses standard ROS 2 interfaces, so the same structure works for a C++
node or for Nav2's controller_server wrapped by a small adapter:

  sub: /eval/path (nav_msgs/Path, transient local), /ground_truth/odom (nav_msgs/Odometry)
  pub: /controller/cmd_vel (geometry_msgs/Twist)
"""
from __future__ import annotations

import json
import math
import os

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from ..geometry import yaw_from_quaternion


class ExampleExternalController(Node):
    def __init__(self):
        super().__init__("example_external_controller")
        params = json.loads(os.environ.get("EVAL_CONTROLLER_PARAMS", "{}") or "{}")
        self.target_speed = float(params.get("target_speed", 0.6))
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.path = None
        self.pose = None
        self.create_subscription(Path, "/eval/path", self.on_path, latched)
        self.create_subscription(Odometry, "/ground_truth/odom", self.on_odom, 20)
        self.pub = self.create_publisher(Twist, "/controller/cmd_vel", 10)
        self.create_timer(0.1, self.on_timer)
        self.get_logger().info(f"example external controller (target {self.target_speed} m/s)")

    def on_path(self, msg: Path) -> None:
        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))

    def on_timer(self) -> None:
        if not self.path or self.pose is None:
            return
        x, y, yaw = self.pose
        # pure pursuit on the received path with a fixed 0.8 m lookahead
        best = min(range(len(self.path)), key=lambda i: math.hypot(self.path[i][0] - x, self.path[i][1] - y))
        target = self.path[-1]
        for px, py in self.path[best:]:
            if math.hypot(px - x, py - y) >= 0.8:
                target = (px, py)
                break
        dx, dy = target[0] - x, target[1] - y
        lx = math.cos(yaw) * dx + math.sin(yaw) * dy
        ly = -math.sin(yaw) * dx + math.cos(yaw) * dy
        curvature = 2.0 * ly / max(lx * lx + ly * ly, 1e-6)
        tw = Twist()
        tw.linear.x = self.target_speed
        tw.angular.z = max(-2.0, min(2.0, curvature * self.target_speed))
        self.pub.publish(tw)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExampleExternalController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.pub.publish(Twist())
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
