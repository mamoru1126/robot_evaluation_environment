"""Runtime behaviour of the moving humans.

The same :class:`HumanAgent` drives the humans in Gazebo (through a
velocity command per model) and in the built-in 2D simulator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .scenario import crossing_distance, human_route


@dataclass
class AgentState:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    yaw: float = 0.0
    radius: float = 0.25


class HumanAgent:
    WAITING = "waiting"
    WALKING = "walking"
    DONE = "done"

    def __init__(self, cfg: Dict[str, Any], course: Dict[str, Any], robot_radius: float = 0.25):
        self.cfg = cfg
        self.id = cfg.get("id", "human")
        self.enabled = bool(cfg.get("enabled", True))
        self.route = np.asarray(human_route(cfg, course), float)
        self.speed = float(cfg["speed"])
        self.radius = float(cfg["radius"])
        self.robot_radius = robot_radius
        self.state = self.WAITING
        self.target_index = 1
        self.start_time: Optional[float] = None
        self.course = course

    # ------------------------------------------------------------------
    def initial_pose(self) -> Tuple[float, float, float]:
        (x0, y0), (x1, y1) = self.route[0], self.route[1]
        if math.hypot(x1 - x0, y1 - y0) > 1e-6:
            yaw = math.atan2(y1 - y0, x1 - x0)
        else:
            yaw = math.pi  # face the robot
        return float(x0), float(y0), yaw

    def _triggered(self, t: float, robot_x: float, robot_v: float) -> bool:
        trig = self.cfg["trigger"]
        if trig == "robot_x":
            return robot_x >= self.cfg["trigger_robot_x"]
        if trig == "encounter" and self.cfg["pattern"] == "crossing":
            if robot_x >= self.cfg["cross_x"]:
                return True
            t_h = crossing_distance(self.cfg) / max(self.speed, 1e-3) + self.cfg["encounter_offset"]
            t_r = (self.cfg["cross_x"] - robot_x) / max(robot_v, 0.2)
            return t_r <= t_h
        return t >= self.cfg["trigger_time"]

    def update(self, t: float, pos: Tuple[float, float], robot: Optional[Tuple[float, float, float, float]],
               dt: float) -> Tuple[float, float]:
        """Return the desired world-frame velocity (vx, vy).

        ``robot`` is (x, y, yaw, v) of the robot or None.
        """
        if not self.enabled or self.speed <= 0.0 or self.cfg["pattern"] == "static":
            return 0.0, 0.0
        rx, ry, ryaw, rv = robot if robot is not None else (-1e9, 0.0, 0.0, 0.0)
        if self.state == self.WAITING:
            if self._triggered(t, rx, rv):
                self.state = self.WALKING
                self.start_time = t
            else:
                return 0.0, 0.0
        if self.state == self.DONE:
            return 0.0, 0.0

        p = np.asarray(pos, float)
        # advance waypoint index when close enough
        while True:
            target = self.route[self.target_index]
            dist = float(np.hypot(*(target - p)))
            if dist > max(0.05, 0.5 * self.speed * dt):
                break
            if self.target_index + 1 < len(self.route):
                self.target_index += 1
            elif self.cfg.get("loop") and self.cfg["pattern"] == "waypoints":
                self.target_index = 0
            else:
                self.state = self.DONE
                return 0.0, 0.0
        direction = (target - p) / max(dist, 1e-9)
        speed = min(self.speed, dist / max(dt, 1e-3))
        v = direction * speed

        if self.cfg.get("reactive") and robot is not None:
            rel = p - np.array([rx, ry])
            d = float(np.hypot(*rel))
            if 1e-6 < d < 3.0:
                n = rel / d
                # social-force style repulsion (Helbing): A exp((r_ij - d_ij)/B)
                force = 2.0 * math.exp((self.radius + self.robot_radius - d) / 0.5)
                # mostly push sideways so that the human keeps walking to its goal
                side = np.array([-direction[1], direction[0]])
                lateral = side * (1.0 if float(n @ side) >= 0 else -1.0)
                v = v + force * (0.4 * n + 0.6 * lateral)
                s = float(np.hypot(*v))
                vmax = 1.3 * self.speed
                if s > vmax:
                    v = v / s * vmax
        return float(v[0]), float(v[1])
