"""Lightweight kinematic 2D simulator (fast mode / tests).

It mirrors what the Gazebo world provides:

* differential-drive robot whose commanded (v, w) is passed through the same
  independent linear / angular acceleration limiter as gz-sim's DiffDrive
* humans driven by world-frame velocity commands (like gz VelocityControl)
* a 2D LiDAR that sees the humans' two legs and the corridor walls
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .specs import RobotSpec, WorldSpec


@dataclass
class Body:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0      # robot: forward speed
    w: float = 0.0      # robot: yaw rate
    vx: float = 0.0     # humans: world velocity
    vy: float = 0.0
    radius: float = 0.25


@dataclass
class Sim2D:
    robot_spec: RobotSpec
    world: WorldSpec
    num_humans: int = 2
    seed: int = 0
    t: float = 0.0
    robot: Body = field(default_factory=Body)
    humans: List[Body] = field(default_factory=list)
    cmd: Tuple[float, float] = (0.0, 0.0)
    human_cmd: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        self.robot.radius = self.robot_spec.radius
        self.humans = [Body(x=-4.0, y=2.0 * (i + 1)) for i in range(self.num_humans)]
        self.human_cmd = [(0.0, 0.0)] * self.num_humans
        self.rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------
    def teleport_robot(self, x: float, y: float, yaw: float) -> None:
        self.robot.x, self.robot.y, self.robot.yaw = x, y, yaw
        self.robot.v = self.robot.w = 0.0
        self.cmd = (0.0, 0.0)

    def teleport_human(self, i: int, x: float, y: float, yaw: float) -> None:
        h = self.humans[i]
        h.x, h.y, h.yaw, h.vx, h.vy = x, y, yaw, 0.0, 0.0
        self.human_cmd[i] = (0.0, 0.0)

    def set_cmd(self, v: float, w: float) -> None:
        self.cmd = (float(v), float(w))

    def set_human_cmd(self, i: int, vx: float, vy: float) -> None:
        self.human_cmd[i] = (float(vx), float(vy))

    # ------------------------------------------------------------------
    def step(self, dt: float) -> None:
        r = self.robot_spec
        rb = self.robot
        v_t = min(max(self.cmd[0], -r.max_v), r.max_v)
        w_t = min(max(self.cmd[1], -r.max_w), r.max_w)
        dv = v_t - rb.v
        dv = min(dv, r.max_acc * dt) if dv > 0 else max(dv, -r.max_dec * dt)
        dw = w_t - rb.w
        dw = min(max(dw, -r.max_alpha * dt), r.max_alpha * dt)
        v = rb.v + dv
        w = rb.w + dw
        if abs(w) < 1e-9:
            rb.x += v * math.cos(rb.yaw) * dt
            rb.y += v * math.sin(rb.yaw) * dt
        else:
            rb.x += v / w * (math.sin(rb.yaw + w * dt) - math.sin(rb.yaw))
            rb.y -= v / w * (math.cos(rb.yaw + w * dt) - math.cos(rb.yaw))
        rb.yaw = (rb.yaw + w * dt + math.pi) % (2 * math.pi) - math.pi
        rb.v, rb.w = v, w
        for h, (vx, vy) in zip(self.humans, self.human_cmd):
            h.vx, h.vy = vx, vy
            h.x += vx * dt
            h.y += vy * dt
            if math.hypot(vx, vy) > 0.05:
                h.yaw = math.atan2(vy, vx)
        self.t += dt

    # ------------------------------------------------------------------
    def leg_circles(self) -> np.ndarray:
        """(M, 3) circles (x, y, r) of the humans' legs as seen by the LiDAR."""
        return leg_circles([(h.x, h.y, h.yaw) for h in self.humans])

    def scan(self) -> Tuple[np.ndarray, float, float, float, float]:
        """Simulated LaserScan: (ranges, angle_min, angle_increment, range_min, range_max)."""
        rb = self.robot
        return raycast_scan((rb.x, rb.y, rb.yaw), self.robot_spec.lidar, self.leg_circles(),
                            self.world.wall_segments(), self.rng)


def leg_circles(humans) -> np.ndarray:
    """Two legs (r = 7 cm, 20 cm apart) per human given as (x, y, yaw)."""
    out = []
    for x, y, yaw in humans:
        nx, ny = -math.sin(yaw), math.cos(yaw)
        for s in (-0.1, 0.1):
            out.append((x + s * nx, y + s * ny, 0.07))
    return np.asarray(out, float).reshape(-1, 3)


def raycast_scan(robot_pose, L, circles: np.ndarray, walls, rng=None):
    """2D ray casting against circles and horizontal wall segments.

    Returns (ranges, angle_min, angle_increment, range_min, range_max) like
    sensor_msgs/LaserScan (no hit = +inf).
    """
    rx, ry, ryaw = robot_pose
    n = int(L.samples)
    inc = (L.max_angle - L.min_angle) / max(n - 1, 1)
    ang = L.min_angle + inc * np.arange(n)
    ox = rx + L.x * math.cos(ryaw)
    oy = ry + L.x * math.sin(ryaw)
    dx = np.cos(ryaw + ang)
    dy = np.sin(ryaw + ang)
    best = np.full(n, np.inf)
    for cx, cy, cr in circles:
        ex, ey = cx - ox, cy - oy
        b = dx * ex + dy * ey
        c = ex * ex + ey * ey - cr * cr
        disc = b * b - c
        hit = disc >= 0
        tt = b - np.sqrt(np.where(hit, disc, 0.0))
        ok = hit & (tt > 0)
        best = np.where(ok, np.minimum(best, tt), best)
    for (x0, y0), (x1, _) in walls:
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (y0 - oy) / dy
        xs = ox + tt * dx
        ok = np.isfinite(tt) & (tt > 0) & (xs >= x0) & (xs <= x1)
        best = np.where(ok, np.minimum(best, tt), best)
    if L.noise_std > 0 and rng is not None:
        fin = np.isfinite(best)
        best[fin] += rng.normal(0.0, L.noise_std, int(fin.sum()))
    best[best >= L.range_max] = np.inf
    return best, L.min_angle, inc, L.range_min, L.range_max
