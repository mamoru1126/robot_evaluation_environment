"""Gazebo backend: runs the TrialRunner against the ROS 2 / Gazebo stack.

* reads robot and human ground truth from the bridged OdometryPublisher topics
* moves humans with velocity commands (gz VelocityControl, body frame)
* resets the scene with the bridged ``/world/<world>/set_pose`` service
* starts / stops the controller in the local_planner node (JSON on a topic)
  or spawns an *external* controller process (custom_controllers/*.yaml)
* gates ``/controller/cmd_vel`` -> ``/cmd_vel`` only while a trial is running
"""
from __future__ import annotations

import json
import math
import os
import shlex
import signal
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from .. import controllers as ctrl_registry
from ..geometry import quaternion_from_yaw, wrap_angle
from ..humans import AgentState
from ..planner_core import scan_to_points
from ..runner import BatchManager
from ..specs import RobotSpec, WorldSpec
from ..storage import ResultStore
from . import common as C

try:
    from ros_gz_interfaces.msg import Entity
    from ros_gz_interfaces.srv import SetEntityPose
except ImportError:  # pragma: no cover - only without ros_gz installed
    Entity = None
    SetEntityPose = None

ROBOT_MODEL = "eval_robot"


class GazeboBackend:
    name = "gazebo"
    label = "Gazebo (ROS 2 Jazzy / Gazebo Harmonic)"

    def __init__(self, robot: RobotSpec, world: WorldSpec, store: ResultStore, on_event=None):
        if not rclpy.ok():
            rclpy.init(args=None)
        self.robot_spec = robot
        self.world = world
        self.n_humans = int(world.max_humans)
        self.node = rclpy.create_node("eval_server",
                                      parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.lock = threading.RLock()
        self._robot: Optional[Tuple[float, ...]] = None       # t, x, y, yaw, v, w
        self._humans: List[Optional[Tuple[float, ...]]] = [None] * self.n_humans   # t, x, y, yaw, vx, vy
        self._ctrl_cmd: Optional[Tuple[float, float, float]] = None   # v, w, t
        self._ctrl_status: Dict[str, Any] = {}
        self._ctrl_status_wall = 0.0
        self._debug: Dict[str, Any] = {}
        self._scan_pts = np.zeros((0, 2))
        self._clock_wall = 0.0
        self._odom_wall = 0.0
        self._external: Optional[subprocess.Popen] = None
        self._external_info: Dict[str, Any] = {}

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        n = self.node
        self.pub_cmd = n.create_publisher(Twist, C.CMD_VEL, 10)
        self.pub_human = [n.create_publisher(Twist, C.human_cmd_topic(i), 10) for i in range(self.n_humans)]
        self.pub_planner = n.create_publisher(String, C.PLANNER_COMMAND, 10)
        self.pub_path = n.create_publisher(PathMsg, C.EVAL_PATH, latched)
        self.pub_trial = n.create_publisher(String, C.EVAL_TRIAL, latched)
        n.create_subscription(Odometry, C.GT_ODOM, self._on_robot, 50)
        for i in range(self.n_humans):
            n.create_subscription(Odometry, C.human_odom_topic(i), lambda m, i=i: self._on_human(i, m), 20)
        n.create_subscription(Twist, C.CONTROLLER_CMD_VEL, self._on_ctrl_cmd, 10)
        n.create_subscription(String, C.PLANNER_STATUS, self._on_status, latched)
        n.create_subscription(String, C.PLANNER_DEBUG, self._on_debug, 10)
        n.create_subscription(LaserScan, C.SCAN, self._on_scan, qos_profile_sensor_data)
        from rosgraph_msgs.msg import Clock
        n.create_subscription(Clock, "/clock", self._on_clock, 10)
        self.set_pose = None
        if SetEntityPose is not None:
            self.set_pose = n.create_client(SetEntityPose, C.set_pose_service(world.name))
        self.manager = BatchManager(self, robot, world, store, on_event)
        n.create_timer(0.05, self._tick)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(n)
        self.thread = threading.Thread(target=self._spin, name="ros_spin", daemon=True)
        self.thread.start()

    def _spin(self) -> None:
        try:
            self.executor.spin()
        except Exception:  # noqa: BLE001 - executor shut down
            pass

    # ------------------------------------------------------------------ callbacks
    def now(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _on_clock(self, msg) -> None:
        self._clock_wall = time.monotonic()

    def _on_robot(self, msg: Odometry) -> None:
        t, x, y, yaw, v, w, _ = C.odom_to_tuple(msg)
        self._robot = (t, x, y, yaw, v, w)
        self._odom_wall = time.monotonic()

    def _on_human(self, i: int, msg: Odometry) -> None:
        t, x, y, yaw, vxb, _, vyb = C.odom_to_tuple(msg)
        vx, vy = C.body_to_world(vxb, vyb, yaw)
        self._humans[i] = (t, x, y, yaw, vx, vy)

    def _on_ctrl_cmd(self, msg: Twist) -> None:
        self._ctrl_cmd = (msg.linear.x, msg.angular.z, self.now())
        r = self.manager.runner
        if r is not None and r.gate_open:
            self.pub_cmd.publish(msg)

    def _on_status(self, msg: String) -> None:
        try:
            self._ctrl_status = json.loads(msg.data)
            self._ctrl_status_wall = time.monotonic()
        except ValueError:
            pass

    def _on_debug(self, msg: String) -> None:
        try:
            self._debug = json.loads(msg.data)
        except ValueError:
            pass

    def _on_scan(self, msg: LaserScan) -> None:
        rb = self._robot
        if rb is None:
            return
        lx = self.robot_spec.lidar.x
        pose = (rb[1] + lx * math.cos(rb[3]), rb[2] + lx * math.sin(rb[3]), rb[3])
        pts = scan_to_points(msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max, pose)
        self._scan_pts = pts[::3]

    def _tick(self) -> None:
        now = self.now()
        r = self.manager.runner
        if r is not None and r.gate_open:
            c = self._ctrl_cmd
            if c is None or now - c[2] > 0.5:
                self.pub_cmd.publish(Twist())   # watchdog: controller silent -> stop
        self.manager.tick(now)

    # ------------------------------------------------------------------ TrialIO
    def robot(self):
        rb = self._robot
        if rb is None or time.monotonic() - self._odom_wall > 2.0:
            return None
        return rb[1], rb[2], rb[3], rb[4], rb[5]

    def humans(self) -> List[Optional[AgentState]]:
        out = []
        for h in self._humans:
            out.append(None if h is None else AgentState(h[1], h[2], h[4], h[5], h[3]))
        return out

    def _set_pose(self, name: str, x: float, y: float, z: float, yaw: float) -> None:
        if self.set_pose is None:
            self.node.get_logger().error("ros_gz_interfaces not available: cannot reset the scene")
            return
        req = SetEntityPose.Request()
        req.entity = Entity()
        req.entity.name = name
        req.entity.type = Entity.MODEL
        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(z)
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        req.pose.orientation.x, req.pose.orientation.y = qx, qy
        req.pose.orientation.z, req.pose.orientation.w = qz, qw
        if not self.set_pose.service_is_ready():
            self.node.get_logger().warn(f"set_pose service {C.set_pose_service(self.world.name)} not ready")
        self.set_pose.call_async(req)

    def teleport_robot(self, x: float, y: float, yaw: float) -> None:
        self.pub_cmd.publish(Twist())
        self._set_pose(ROBOT_MODEL, x, y, 0.005, yaw)

    def teleport_human(self, i: int, x: float, y: float, yaw: float) -> None:
        if i < self.n_humans:
            self.pub_human[i].publish(Twist())
            self._set_pose(f"human_{i + 1}", x, y, 0.0, yaw)

    def set_course(self, start_x: float, goal_x: float, y: float, path) -> None:
        self._set_pose("start_line", start_x, 0.0, 0.0, 0.0)
        self._set_pose("goal_line", goal_x, 0.0, 0.0, 0.0)
        msg = PathMsg()
        msg.header.frame_id = "world"
        msg.header.stamp = self.node.get_clock().now().to_msg()
        pts = np.asarray(path, float)
        dense = []
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(2, int(np.hypot(*(b - a)) / 0.05) + 1)
            dense.extend(np.linspace(a, b, n)[:-1].tolist())
        dense.append(pts[-1].tolist())
        for k, p in enumerate(dense):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y = float(p[0]), float(p[1])
            nxt = dense[min(k + 1, len(dense) - 1)]
            prv = dense[max(k - 1, 0)]
            yaw = math.atan2(nxt[1] - prv[1], nxt[0] - prv[0])
            q = quaternion_from_yaw(yaw)
            ps.pose.orientation.z, ps.pose.orientation.w = q[2], q[3]
            msg.poses.append(ps)
        self.pub_path.publish(msg)

    def stop_robot(self) -> None:
        self.pub_cmd.publish(Twist())

    def set_human_cmd(self, i: int, vx: float, vy: float) -> None:
        if i >= self.n_humans:
            return
        h = self._humans[i]
        tw = Twist()
        if h is not None and (abs(vx) > 1e-4 or abs(vy) > 1e-4):
            yaw = h[3]
            bx, by = C.world_to_body(vx, vy, yaw)
            tw.linear.x, tw.linear.y = bx, by
            if math.hypot(vx, vy) > 0.05:
                err = wrap_angle(math.atan2(vy, vx) - yaw)
                tw.angular.z = max(-2.5, min(2.5, 3.0 * err))
        self.pub_human[i].publish(tw)

    def controller_start(self, trial_id: str, controller: Dict[str, Any], perception: Dict[str, Any],
                         path) -> None:
        self._stop_external()
        self._debug = {}
        self._ctrl_cmd = None
        info = {"trial_id": trial_id, "controller": controller, "perception": perception, "path": path}
        self.pub_trial.publish(String(data=C.dumps(info)))
        found, _ = ctrl_registry.discover()
        obj = found.get(controller["name"])
        if isinstance(obj, dict):       # external ROS 2 node
            self._start_external(trial_id, obj, controller)
            return
        runner = self.manager.runner
        radii = [h["radius"] for h in runner.scenario["humans"]] if runner is not None else []
        cmd = {"cmd": "start", "trial_id": trial_id, "controller": controller, "perception": perception,
               "path": path, "human_radii": radii}
        self._ctrl_status = {"state": "starting", "trial_id": trial_id}
        self.pub_planner.publish(String(data=C.dumps(cmd)))

    def _start_external(self, trial_id: str, spec: Dict[str, Any], controller: Dict[str, Any]) -> None:
        env = dict(os.environ)
        env["EVAL_TRIAL_ID"] = trial_id
        env["EVAL_CONTROLLER_PARAMS"] = json.dumps(controller.get("params") or spec.get("params") or {})
        try:
            self._external = subprocess.Popen(["bash", "-lc", str(spec["command"])], env=env,
                                              start_new_session=True)
        except OSError as exc:
            self._ctrl_status = {"state": "error", "trial_id": trial_id, "error": str(exc)}
            return
        self._external_info = {"trial_id": trial_id, "t0": time.monotonic(),
                               "startup": float(spec.get("startup_time", 3.0))}
        self._ctrl_status = {"state": "starting", "trial_id": trial_id}

    def _stop_external(self) -> None:
        p = self._external
        if p is None:
            return
        self._external = None
        try:
            os.killpg(p.pid, signal.SIGINT)
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def controller_stop(self) -> None:
        self.pub_planner.publish(String(data=C.dumps({"cmd": "stop"})))
        threading.Thread(target=self._stop_external, daemon=True).start()
        self.pub_cmd.publish(Twist())

    def controller_state(self) -> Dict[str, Any]:
        if self._external is not None:
            info = self._external_info
            if self._external.poll() is not None:
                return {"state": "error", "trial_id": info.get("trial_id"),
                        "error": f"external controller exited with code {self._external.returncode}"}
            if time.monotonic() - info.get("t0", 0) >= info.get("startup", 3.0):
                return {"state": "running", "trial_id": info.get("trial_id")}
            return {"state": "starting", "trial_id": info.get("trial_id")}
        return self._ctrl_status

    def controller_cmd(self) -> Optional[Tuple[float, float]]:
        c = self._ctrl_cmd
        return None if c is None else (c[0], c[1])

    def controller_debug(self) -> Dict[str, Any]:
        return self._debug

    # ------------------------------------------------------------------ status
    def available(self) -> Dict[str, Any]:
        now = time.monotonic()
        problems = []
        if now - self._clock_wall > 3.0:
            problems.append("Gazebo の /clock が届いていません (sim コンテナ / Gazebo を起動してください)")
        elif now - self._odom_wall > 3.0:
            problems.append("ロボットの /ground_truth/odom が届いていません (ros_gz_bridge を確認)")
        if now - self._ctrl_status_wall > 5.0:
            problems.append("local_planner ノードが応答していません")
        if self.set_pose is not None and not self.set_pose.service_is_ready():
            problems.append(f"{C.set_pose_service(self.world.name)} サービスがありません")
        return {"ok": not problems, "message": " / ".join(problems) if problems else "Gazebo と接続中"}

    def start_batch(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        return self.manager.start(spec)

    def abort(self) -> None:
        self.manager.abort()

    def shutdown(self) -> None:
        try:
            self.manager.abort()
            self._stop_external()
            self.pub_cmd.publish(Twist())
            self.executor.shutdown(timeout_sec=1.0)
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass

    def snapshot(self) -> Dict[str, Any]:
        rb = self._robot
        robot = None if rb is None else {"x": rb[1], "y": rb[2], "yaw": rb[3], "v": rb[4], "w": rb[5],
                                         "radius": self.robot_spec.radius}
        humans = []
        for h in self._humans:
            humans.append({"x": -100.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "yaw": 0.0} if h is None else
                          {"x": h[1], "y": h[2], "vx": h[4], "vy": h[5], "yaw": h[3]})
        c = self._ctrl_cmd
        pts = self._scan_pts
        return {"backend": self.name, "connected": time.monotonic() - self._clock_wall < 3.0,
                "time": self.now(), "robot": robot, "humans": humans,
                "cmd": [c[0], c[1]] if c else [0.0, 0.0],
                "scan": np.round(pts, 3).tolist() if len(pts) else []}
