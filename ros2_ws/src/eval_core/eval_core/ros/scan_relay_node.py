"""/scan provider: relays Gazebo's gpu_lidar or falls back to a virtual LiDAR.

mode:=gpu      relay /scan_gz (Gazebo gpu_lidar via ros_gz_bridge) -> /scan
mode:=virtual  ray-cast the ground-truth humans (two legs) and walls -> /scan
mode:=auto     relay while Gazebo publishes; if no gpu_lidar scan arrives
               (e.g. no OpenGL / EGL in the container) switch to virtual.
"""
from __future__ import annotations

import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from ..sim2d import leg_circles, raycast_scan
from ..specs import load_robot_spec, load_world_spec
from . import common as C


class ScanRelay(Node):
    def __init__(self):
        super().__init__("scan_relay")
        self.robot = load_robot_spec()
        self.world = load_world_spec()
        dyn = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("mode", "auto", dyn)
        self.declare_parameter("grace_period", 20.0, dyn)
        self.mode = str(self.get_parameter("mode").value)
        self.grace = float(self.get_parameter("grace_period").value)
        if self.mode not in ("auto", "gpu", "virtual"):
            self.mode = "auto"
        self.t_start = time.monotonic()
        self.last_gz = 0.0
        self.virtual_active = self.mode == "virtual"
        self.robot_pose = None
        self.humans = [None] * int(self.world.max_humans)
        self.rng = np.random.default_rng(0)
        self.pub = self.create_publisher(LaserScan, C.SCAN, qos_profile_sensor_data)
        self.create_subscription(LaserScan, "/scan_gz", self.on_gz_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, C.GT_ODOM, self.on_robot, 20)
        for i in range(len(self.humans)):
            self.create_subscription(Odometry, C.human_odom_topic(i), lambda m, i=i: self.on_human(i, m), 20)
        self.create_timer(1.0 / max(self.robot.lidar.rate, 1.0), self.on_timer)
        self.get_logger().info(f"scan relay mode={self.mode}")

    def on_gz_scan(self, msg: LaserScan) -> None:
        self.last_gz = time.monotonic()
        if self.virtual_active and self.mode == "auto":
            self.virtual_active = False
            self.get_logger().info("gpu_lidar scans arrived: relaying Gazebo /scan")
        if self.mode in ("auto", "gpu"):
            self.pub.publish(msg)

    def on_robot(self, msg: Odometry) -> None:
        _, x, y, yaw, _, _, _ = C.odom_to_tuple(msg)
        self.robot_pose = (x, y, yaw)

    def on_human(self, i: int, msg: Odometry) -> None:
        _, x, y, yaw, _, _, _ = C.odom_to_tuple(msg)
        self.humans[i] = (x, y, yaw)

    def on_timer(self) -> None:
        now = time.monotonic()
        if self.mode == "auto" and not self.virtual_active:
            if now - self.t_start > self.grace and now - self.last_gz > 2.0 and self.robot_pose is not None:
                self.virtual_active = True
                self.get_logger().warn("no gpu_lidar scan from Gazebo -> publishing a VIRTUAL LiDAR scan "
                                       "(ray casting of ground-truth humans and walls)")
        if not self.virtual_active or self.robot_pose is None:
            return
        circles = leg_circles([h for h in self.humans if h is not None])
        ranges, amin, inc, rmin, rmax = raycast_scan(self.robot_pose, self.robot.lidar, circles,
                                                     self.world.wall_segments(), self.rng)
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "lidar_link"
        msg.angle_min = float(amin)
        msg.angle_max = float(amin + inc * (len(ranges) - 1))
        msg.angle_increment = float(inc)
        msg.time_increment = 0.0
        msg.scan_time = 1.0 / max(self.robot.lidar.rate, 1.0)
        msg.range_min = float(rmin)
        msg.range_max = float(rmax)
        msg.ranges = [float(r) for r in ranges]
        self.pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ScanRelay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
