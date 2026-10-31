"""ROS 2 node that runs the selected Python controller.

Subscribes : /ground_truth/odom (nav_msgs/Odometry), /scan (sensor_msgs/LaserScan),
             /humans/human_N/odom (nav_msgs/Odometry), /local_planner/command (std_msgs/String JSON)
Publishes  : /controller/cmd_vel (geometry_msgs/Twist)  -> gated by the eval server
             /local_planner/status (JSON, transient local), /local_planner/debug (JSON),
             /local_planner/markers (visualization_msgs/MarkerArray for RViz)

The controller is (re)created from the registry on every ``start`` command, so
edits to files in custom_controllers/ are picked up without restarting.
"""
from __future__ import annotations

import json

import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from ..planner_core import LocalPlannerCore
from ..specs import load_robot_spec, load_world_spec
from . import common as C


class LocalPlannerNode(Node):
    def __init__(self):
        super().__init__("local_planner")
        dyn = ParameterDescriptor(dynamic_typing=True)   # accept "10" as well as 10.0 from launch
        self.declare_parameter("control_rate", 10.0, dyn)
        robot = load_robot_spec()
        world = load_world_spec()
        self.declare_parameter("num_humans", int(world.max_humans), dyn)
        self.core = LocalPlannerCore(robot, world)
        self.num_humans = int(self.get_parameter("num_humans").value)
        rate = float(self.get_parameter("control_rate").value)
        self.dt = 1.0 / max(rate, 1.0)
        self.status = {"state": "idle", "trial_id": None, "error": ""}
        self.human_radii = [0.25] * self.num_humans
        self.last_heartbeat = -1e9

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_cmd = self.create_publisher(Twist, C.CONTROLLER_CMD_VEL, 10)
        self.pub_status = self.create_publisher(String, C.PLANNER_STATUS, latched)
        self.pub_debug = self.create_publisher(String, C.PLANNER_DEBUG, 10)
        self.pub_markers = self.create_publisher(MarkerArray, C.PLANNER_MARKERS, 10)
        self.create_subscription(Odometry, C.GT_ODOM, self.on_odom, 50)
        self.create_subscription(LaserScan, C.SCAN, self.on_scan, qos_profile_sensor_data)
        for i in range(self.num_humans):
            self.create_subscription(Odometry, C.human_odom_topic(i),
                                     lambda msg, i=i: self.on_human(i, msg), 20)
        self.create_subscription(String, C.PLANNER_COMMAND, self.on_command, 10)
        self.create_timer(self.dt, self.on_timer)
        self.publish_status()
        self.get_logger().info(f"local_planner ready (control {rate:.1f} Hz, humans={self.num_humans})")

    # ------------------------------------------------------------------
    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def publish_status(self) -> None:
        self.pub_status.publish(String(data=C.dumps(dict(self.status, node_time=self.now(),
                                                         controller=self.core.controller_name))))

    def on_odom(self, msg: Odometry) -> None:
        t, x, y, yaw, v, w, _ = C.odom_to_tuple(msg)
        self.core.update_odom(t, x, y, yaw, v, w)

    def on_scan(self, msg: LaserScan) -> None:
        self.core.update_scan(C.stamp_to_sec(msg.header.stamp), msg.ranges, msg.angle_min, msg.angle_increment,
                              msg.range_min, msg.range_max)

    def on_human(self, i: int, msg: Odometry) -> None:
        t, x, y, yaw, vxb, _, vyb = C.odom_to_tuple(msg)
        vx, vy = C.body_to_world(vxb, vyb, yaw)
        r = self.human_radii[i] if i < len(self.human_radii) else 0.25
        self.core.update_human(f"human_{i + 1}", t, x, y, vx, vy, r)

    def on_command(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except ValueError:
            self.get_logger().warn("invalid command json")
            return
        cmd = data.get("cmd")
        if cmd == "start":
            ctrl = data.get("controller") or {}
            per = data.get("perception") or {}
            self.human_radii = [float(r) for r in (data.get("human_radii") or [0.25] * self.num_humans)]
            self.core.humans.clear()
            ok, text = self.core.start(str(data.get("trial_id")), ctrl.get("name", ""), ctrl.get("params") or {},
                                       per.get("mode", "scan"), data.get("path") or [[0, 0], [1, 0]],
                                       per.get("use_prediction", True), per.get("scan_inflation", 0.15))
            self.status = {"state": "running" if ok else "error", "trial_id": data.get("trial_id"),
                           "error": "" if ok else text}
            if ok:
                self.get_logger().info(f"start trial {data.get('trial_id')} controller={ctrl.get('name')} "
                                       f"perception={per.get('mode')}")
            else:
                self.get_logger().error(text)
            self.publish_status()
        elif cmd == "stop":
            self.core.stop()
            self.pub_cmd.publish(Twist())
            if self.status.get("state") == "running":
                self.status = dict(self.status, state="idle")
            self.publish_status()
        elif cmd == "ping":
            self.publish_status()

    def on_timer(self) -> None:
        now = self.now()
        if now - self.last_heartbeat > 1.0 or now < self.last_heartbeat:
            self.last_heartbeat = now
            self.publish_status()
        if not self.core.active:
            return
        out = self.core.step(now, self.dt)
        if out is None:
            if self.core.error and self.status.get("state") == "running":
                self.status = dict(self.status, state="error", error=self.core.error)
                self.get_logger().error(f"controller error:\n{self.core.error}")
                self.pub_cmd.publish(Twist())
                self.publish_status()
            return
        tw = Twist()
        tw.linear.x = float(out.v)
        tw.angular.z = float(out.w)
        self.pub_cmd.publish(tw)
        traj = out.trajectory or []
        step = max(1, len(traj) // 30)
        dbg = {"trial_id": self.core.trial_id, "t": now, "v": out.v, "w": out.w,
               "trajectory": [[round(p[0], 3), round(p[1], 3)] for p in traj[::step]],
               "candidates": [[[round(p[0], 3), round(p[1], 3)] for p in c[:: max(1, len(c) // 10)]]
                              for c in (out.candidates or [])[:15]]}
        for k, v in (out.info or {}).items():
            if isinstance(v, (int, float, bool, str)):
                dbg[k] = v
        self.pub_debug.publish(String(data=C.dumps(dbg)))
        self.publish_markers(out)

    def publish_markers(self, out) -> None:
        if self.pub_markers.get_subscription_count() == 0:
            return
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "predicted"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.04
        m.color.r, m.color.g, m.color.b, m.color.a = 0.07, 0.64, 0.5, 1.0
        m.pose.orientation.w = 1.0
        for p in out.trajectory or []:
            m.points.append(Point(x=float(p[0]), y=float(p[1]), z=0.05))
        ma.markers.append(m)
        self.pub_markers.publish(ma)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LocalPlannerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.pub_cmd.publish(Twist())
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
