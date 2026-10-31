"""ROS-independent core of the local planner node.

It turns sensor data (robot odometry, LiDAR scan, ground-truth humans) into
an :class:`ObstacleSet` and calls the selected controller.  The ROS node
(:mod:`eval_core.ros.local_planner_node`) and the 2D simulator both use it,
so the controller sees exactly the same inputs in both environments.
"""
from __future__ import annotations

import bisect
import math
import time
import traceback
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import controllers as ctrl_registry
from .controllers.base import BaseController, ControllerInput, ControlOutput, RobotState
from .geometry import Path, segment_points, wrap_angle
from .obstacles import DynamicAgent, ObstacleSet
from .specs import RobotSpec, WorldSpec
from .tracking import ScanTracker

PERCEPTION_MODES = {
    "scan": "LiDAR 点群のみ (静止物として扱う・予測なし)",
    "scan_tracking": "LiDAR + 人追跡 (クラスタ追跡で速度推定・予測あり)",
    "ground_truth": "真値 (シミュレータの人の位置・速度 + 既知の壁)",
}


def scan_to_points(ranges, angle_min: float, angle_increment: float, range_min: float,
                   range_max: float, sensor_pose: Tuple[float, float, float]) -> np.ndarray:
    r = np.asarray(ranges, float)
    ang = angle_min + angle_increment * np.arange(len(r))
    ok = np.isfinite(r) & (r >= range_min) & (r < range_max)
    r = r[ok]
    ang = ang[ok]
    sx, sy, syaw = sensor_pose
    return np.stack([sx + r * np.cos(syaw + ang), sy + r * np.sin(syaw + ang)], -1)


class LocalPlannerCore:
    def __init__(self, robot: RobotSpec, world: WorldSpec):
        self.robot = robot
        self.world = world
        self.controller: Optional[BaseController] = None
        self.controller_name = ""
        self.path: Optional[Path] = None
        self.perception = "scan"
        self.use_prediction = True
        self.scan_inflation = 0.15
        self.active = False
        self.trial_id: Optional[str] = None
        self.error: Optional[str] = None
        self._pose_t: deque = deque(maxlen=400)
        self._pose: deque = deque(maxlen=400)
        self.state: Optional[RobotState] = None
        self.scan_points = np.zeros((0, 2))
        self.scan_stamp = -1e9
        self.tracker = ScanTracker()
        self.tracked_static = np.zeros((0, 2))
        self.tracked_agents: List[DynamicAgent] = []
        self.humans: Dict[str, Tuple[float, DynamicAgent]] = {}
        self.last_cmd = (0.0, 0.0)
        self.compute_ms: List[float] = []
        self._wall_points = self._make_wall_points()

    # ------------------------------------------------------------------
    def _make_wall_points(self) -> np.ndarray:
        pts = [segment_points(a, b, 0.1) for a, b in self.world.wall_segments()]
        return np.concatenate(pts) if pts else np.zeros((0, 2))

    def start(self, trial_id: str, controller: str, params: Dict[str, Any], perception: str,
              path_points: List[List[float]], use_prediction: bool = True,
              scan_inflation: float = 0.15) -> Tuple[bool, str]:
        try:
            self.controller = ctrl_registry.create(controller, self.robot, params)
            self.controller.reset()
            self.path = Path(path_points)
        except Exception as exc:  # noqa: BLE001
            self.active = False
            self.error = f"{exc}"
            return False, f"failed to start controller '{controller}': {exc}"
        self.controller_name = controller
        self.perception = perception if perception in PERCEPTION_MODES else "scan"
        self.use_prediction = bool(use_prediction)
        self.scan_inflation = max(0.0, float(scan_inflation))
        self.trial_id = trial_id
        self.last_cmd = (0.0, 0.0)
        self.compute_ms = []
        self.error = None
        self.tracker.reset()
        self.tracked_static = np.zeros((0, 2))
        self.tracked_agents = []
        self.active = True
        return True, "started"

    def stop(self) -> None:
        self.active = False
        self.last_cmd = (0.0, 0.0)

    # ------------------------------------------------------------------
    def update_odom(self, t: float, x: float, y: float, yaw: float, v: float, w: float) -> None:
        if self._pose_t and t < self._pose_t[-1] - 1.0:
            # time jumped backwards (simulation reset): drop history
            self._pose_t.clear()
            self._pose.clear()
        self._pose_t.append(t)
        self._pose.append((x, y, yaw))
        self.state = RobotState(t, x, y, yaw, v, w)

    def pose_at(self, t: float) -> Optional[Tuple[float, float, float]]:
        if not self._pose_t:
            return None
        ts = list(self._pose_t)
        i = bisect.bisect_left(ts, t)
        if i <= 0:
            return self._pose[0]
        if i >= len(ts):
            return self._pose[-1]
        t0, t1 = ts[i - 1], ts[i]
        a = (t - t0) / max(t1 - t0, 1e-9)
        x0, y0, h0 = self._pose[i - 1]
        x1, y1, h1 = self._pose[i]
        return x0 + a * (x1 - x0), y0 + a * (y1 - y0), h0 + a * wrap_angle(h1 - h0)

    def update_scan(self, t: float, ranges, angle_min: float, angle_increment: float,
                    range_min: float, range_max: float) -> None:
        pose = self.pose_at(t)
        if pose is None:
            return
        x, y, yaw = pose
        lx = self.robot.lidar.x
        sensor = (x + lx * math.cos(yaw), y + lx * math.sin(yaw), yaw)
        self.scan_points = scan_to_points(ranges, angle_min, angle_increment, range_min, range_max, sensor)
        self.scan_stamp = t
        if self.perception == "scan_tracking":
            self.tracked_static, self.tracked_agents = self.tracker.update(t, self.scan_points)

    def update_human(self, hid: str, t: float, x: float, y: float, vx: float, vy: float,
                     radius: float = 0.25) -> None:
        self.humans[hid] = (t, DynamicAgent(x, y, vx, vy, radius, hid))

    # ------------------------------------------------------------------
    def build_obstacles(self, now: float) -> ObstacleSet:
        st = self.state
        center = (st.x, st.y) if st else (0.0, 0.0)
        if self.perception == "ground_truth":
            agents = [a for (t, a) in self.humans.values() if now - t < 1.0]
            pts = self._wall_points
            if len(pts):
                near = np.hypot(pts[:, 0] - center[0], pts[:, 1] - center[1]) < 8.0
                pts = pts[near]
            return ObstacleSet(pts, agents, self.robot.radius, center, use_prediction=self.use_prediction)
        fresh = now - self.scan_stamp < 1.0
        if self.perception == "scan_tracking":
            pts = self.tracked_static if fresh else np.zeros((0, 2))
            agents = [DynamicAgent(a.x, a.y, a.vx, a.vy, a.radius + self.scan_inflation, a.id)
                      for a in self.tracked_agents] if fresh else []
            return ObstacleSet(pts, agents, self.robot.radius, center, point_radius=self.scan_inflation,
                               use_prediction=self.use_prediction)
        pts = self.scan_points if fresh else np.zeros((0, 2))
        return ObstacleSet(pts, [], self.robot.radius, center, point_radius=self.scan_inflation)

    def step(self, now: float, dt: float) -> Optional[ControlOutput]:
        """Run the controller once. Returns None if inactive or no odometry."""
        if not self.active or self.controller is None or self.path is None or self.state is None:
            return None
        obstacles = self.build_obstacles(now)
        inp = ControllerInput(self.state, self.path, obstacles, dt, self.last_cmd)
        t0 = time.perf_counter()
        try:
            out = self.controller.compute(inp)
        except Exception:  # noqa: BLE001 - user controllers may raise anything
            self.error = traceback.format_exc(limit=4)
            self.active = False
            self.last_cmd = (0.0, 0.0)
            return None
        ms = (time.perf_counter() - t0) * 1e3
        self.compute_ms.append(ms)
        v = float(np.clip(out.v, self.robot.min_v, self.robot.max_v)) if math.isfinite(out.v) else 0.0
        w = float(np.clip(out.w, -self.robot.max_w, self.robot.max_w)) if math.isfinite(out.w) else 0.0
        out.v, out.w = v, w
        out.info = dict(out.info or {})
        out.info["compute_ms"] = ms
        self.last_cmd = (v, w)
        return out
