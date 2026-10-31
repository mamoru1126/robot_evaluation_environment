"""Obstacle representation used by every controller.

* ``points``  : static obstacle points in the world frame (LiDAR returns or
                known walls).  They are rasterised into a local grid and turned
                into a Euclidean distance field (scipy EDT) for fast queries.
* ``agents``  : dynamic circular obstacles with velocity (ground-truth humans).
                They can be predicted with a constant-velocity model.

``clearance`` returns the *surface* distance between the robot footprint
(circle) and the closest obstacle; a value <= 0 means collision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:  # pragma: no cover
    distance_transform_edt = None


@dataclass
class DynamicAgent:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 0.25
    id: str = ""


class ObstacleSet:
    def __init__(self, points: Optional[np.ndarray], agents: Sequence[DynamicAgent], robot_radius: float,
                 center: Sequence[float], grid_size: float = 12.0, resolution: float = 0.05,
                 use_prediction: bool = True, far: float = 10.0, point_radius: float = 0.0):
        self.points = np.zeros((0, 2)) if points is None else np.asarray(points, float).reshape(-1, 2)
        self.point_radius = float(point_radius)   # inflation of static points (e.g. legs -> body)
        self.agents: List[DynamicAgent] = list(agents)
        self.robot_radius = float(robot_radius)
        self.use_prediction = use_prediction
        self.far = far
        self.res = resolution
        self.grid = None
        cx, cy = float(center[0]), float(center[1])
        half = grid_size / 2.0
        self.origin = np.array([cx - half, cy - half])
        self.n = int(round(grid_size / resolution))
        if len(self.points):
            local = self.points - self.origin
            idx = np.floor(local / resolution).astype(int)
            ok = (idx[:, 0] >= 0) & (idx[:, 0] < self.n) & (idx[:, 1] >= 0) & (idx[:, 1] < self.n)
            idx = idx[ok]
            if len(idx):
                occ = np.ones((self.n, self.n), dtype=bool)
                occ[idx[:, 0], idx[:, 1]] = False
                if distance_transform_edt is not None:
                    self.grid = distance_transform_edt(occ) * resolution
                else:  # very slow fallback, only for environments without scipy
                    gx, gy = np.meshgrid(np.arange(self.n), np.arange(self.n), indexing="ij")
                    cells = np.stack([gx, gy], -1).reshape(-1, 2)
                    d = np.min(np.hypot(*(cells[:, None, :] - idx[None, :, :]).transpose(2, 0, 1)), axis=1)
                    self.grid = d.reshape(self.n, self.n) * resolution
        self._agent_pos = np.array([[a.x, a.y] for a in self.agents], float).reshape(-1, 2)
        self._agent_vel = np.array([[a.vx, a.vy] for a in self.agents], float).reshape(-1, 2)
        self._agent_rad = np.array([a.radius for a in self.agents], float)

    # ------------------------------------------------------------------
    @property
    def empty(self) -> bool:
        return self.grid is None and not self.agents

    def static_distance(self, xy: np.ndarray) -> np.ndarray:
        """Distance from point(s) to the closest static obstacle point."""
        xy = np.asarray(xy, float)
        if self.grid is None:
            return np.full(xy.shape[:-1], self.far)
        local = (xy - self.origin) / self.res
        i = np.floor(local[..., 0]).astype(int)
        j = np.floor(local[..., 1]).astype(int)
        inside = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        out = np.full(xy.shape[:-1], self.far)
        out[inside] = self.grid[i[inside], j[inside]] - self.point_radius
        return out

    def dynamic_distance(self, xy: np.ndarray, tau=0.0) -> np.ndarray:
        """Surface distance (minus agent radius) to the predicted agents."""
        xy = np.asarray(xy, float)
        if not self.agents:
            return np.full(xy.shape[:-1], self.far)
        tau = np.asarray(tau, float) if self.use_prediction else np.zeros(())
        best = np.full(xy.shape[:-1], self.far)
        for p, v, r in zip(self._agent_pos, self._agent_vel, self._agent_rad):
            px = p[0] + v[0] * tau
            py = p[1] + v[1] * tau
            d = np.hypot(xy[..., 0] - px, xy[..., 1] - py) - r
            best = np.minimum(best, d)
        return best

    def clearance(self, xy: np.ndarray, tau=0.0) -> np.ndarray:
        """Robot-surface clearance at position(s) ``xy`` at future time ``tau``."""
        d = np.minimum(self.static_distance(xy), self.dynamic_distance(xy, tau))
        return d - self.robot_radius

    def clearance_now(self, x: float, y: float) -> float:
        return float(self.clearance(np.array([x, y])))

    # ------------------------------------------------------------------
    def nearby_static_points(self, x: float, y: float, radius: float, max_points: int = 8,
                             min_separation: float = 0.3) -> np.ndarray:
        """A handful of representative static points around (x, y).

        Used by optimisation based controllers (MPC) that need explicit
        obstacle positions: greedy selection of the closest points that are at
        least ``min_separation`` apart.
        """
        if not len(self.points):
            return np.zeros((0, 2))
        d = np.hypot(self.points[:, 0] - x, self.points[:, 1] - y)
        order = np.argsort(d)
        chosen: List[np.ndarray] = []
        for k in order:
            if d[k] > radius or len(chosen) >= max_points:
                break
            p = self.points[k]
            if all(np.hypot(*(p - c)) >= min_separation for c in chosen):
                chosen.append(p)
        return np.asarray(chosen).reshape(-1, 2)
