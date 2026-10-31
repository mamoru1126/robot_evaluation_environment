"""Small geometry helpers shared by controllers, simulators and metrics."""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def wrap_angle(a):
    """Wrap angle(s) to [-pi, pi). Works for floats and numpy arrays."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi if isinstance(a, np.ndarray) \
        else (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    """Return (x, y, z, w)."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def to_robot_frame(px, py, rx: float, ry: float, ryaw: float):
    """Transform world point(s) into the robot frame."""
    dx = np.asarray(px) - rx
    dy = np.asarray(py) - ry
    c, s = math.cos(ryaw), math.sin(ryaw)
    return c * dx + s * dy, -s * dx + c * dy


def crossing_time(t0: float, x0: float, t1: float, x1: float, line_x: float) -> Optional[float]:
    """Linearly interpolated time at which a coordinate crosses ``line_x``
    while going in +x direction between two samples."""
    if x0 < line_x <= x1 and x1 > x0:
        a = (line_x - x0) / (x1 - x0)
        return t0 + a * (t1 - t0)
    return None


class Path:
    """2D polyline with arc-length parametrisation.

    The evaluation path is a straight START->GOAL line, but every controller
    goes through this class so that arbitrary polylines also work.
    """

    def __init__(self, points: Sequence[Sequence[float]]):
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) < 2:
            raise ValueError("A path needs at least two points")
        # drop consecutive duplicates
        keep = [0]
        for i in range(1, len(pts)):
            if np.hypot(*(pts[i] - pts[keep[-1]])) > 1e-6:
                keep.append(i)
        pts = pts[keep]
        if len(pts) < 2:
            raise ValueError("Degenerate path")
        self.points = pts
        seg = np.diff(pts, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.seg_dir = seg / self.seg_len[:, None]
        self.seg_heading = np.arctan2(seg[:, 1], seg[:, 0])
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.s[-1])

    # ------------------------------------------------------------------
    @classmethod
    def straight(cls, start: Sequence[float], goal: Sequence[float], spacing: float = 0.0) -> "Path":
        start = np.asarray(start, float)
        goal = np.asarray(goal, float)
        if spacing > 0:
            n = max(2, int(math.ceil(np.hypot(*(goal - start)) / spacing)) + 1)
            return cls(np.linspace(start, goal, n))
        return cls([start, goal])

    def to_list(self):
        return self.points.tolist()

    @property
    def goal(self) -> np.ndarray:
        return self.points[-1]

    # ------------------------------------------------------------------
    def project(self, x: float, y: float) -> Tuple[float, float, int]:
        """Project a point onto the path.

        Returns (s, signed_lateral_error, segment_index). Lateral error is
        positive when the point is on the left of the path direction.
        """
        p = np.array([x, y])
        rel = p - self.points[:-1]
        t = np.einsum("ij,ij->i", rel, self.seg_dir)
        t = np.clip(t, 0.0, self.seg_len)
        proj = self.points[:-1] + self.seg_dir * t[:, None]
        d = np.hypot(*(p - proj).T)
        i = int(np.argmin(d))
        lat = float(self.seg_dir[i, 0] * rel[i, 1] - self.seg_dir[i, 1] * rel[i, 0])
        return float(self.s[i] + t[i]), lat, i

    def point_at(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, self.length))
        i = int(np.searchsorted(self.s, s, side="right") - 1)
        i = min(max(i, 0), len(self.seg_len) - 1)
        return self.points[i] + self.seg_dir[i] * (s - self.s[i])

    def points_at(self, s: np.ndarray) -> np.ndarray:
        s = np.clip(np.asarray(s, float), 0.0, self.length)
        i = np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.seg_len) - 1)
        return self.points[i] + self.seg_dir[i] * (s - self.s[i])[..., None]

    def heading_at(self, s) -> np.ndarray:
        s = np.clip(np.asarray(s, float), 0.0, self.length)
        i = np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.seg_len) - 1)
        return self.seg_heading[i]

    def lateral_errors(self, xy: np.ndarray) -> np.ndarray:
        """Unsigned distance from many points (..., 2) to the polyline."""
        xy = np.asarray(xy, float)
        flat = xy.reshape(-1, 2)
        best = np.full(len(flat), np.inf)
        for a, d, L in zip(self.points[:-1], self.seg_dir, self.seg_len):
            rel = flat - a
            t = np.clip(rel @ d, 0.0, L)
            proj = a + t[:, None] * d
            best = np.minimum(best, np.hypot(*(flat - proj).T))
        return best.reshape(xy.shape[:-1])

    def signed_lateral(self, xy: np.ndarray) -> np.ndarray:
        """Signed distance (left of the path direction = positive) for many points (..., 2)."""
        xy = np.asarray(xy, float)
        flat = xy.reshape(-1, 2)
        best_d = np.full(len(flat), np.inf)
        best_sd = np.zeros(len(flat))
        for a, d, L in zip(self.points[:-1], self.seg_dir, self.seg_len):
            rel = flat - a
            t = np.clip(rel @ d, 0.0, L)
            proj = a + t[:, None] * d
            dist = np.hypot(*(flat - proj).T)
            sd = d[0] * rel[:, 1] - d[1] * rel[:, 0]
            better = dist < best_d
            best_d = np.where(better, dist, best_d)
            best_sd = np.where(better, sd, best_sd)
        return best_sd.reshape(xy.shape[:-1])

    def progress(self, xy: np.ndarray) -> np.ndarray:
        """Arc length of the projection of many points (..., 2) on the path."""
        xy = np.asarray(xy, float)
        flat = xy.reshape(-1, 2)
        best_d = np.full(len(flat), np.inf)
        best_s = np.zeros(len(flat))
        for a, d, L, s0 in zip(self.points[:-1], self.seg_dir, self.seg_len, self.s[:-1]):
            rel = flat - a
            t = np.clip(rel @ d, 0.0, L)
            proj = a + t[:, None] * d
            dist = np.hypot(*(flat - proj).T)
            better = dist < best_d
            best_d = np.where(better, dist, best_d)
            best_s = np.where(better, s0 + t, best_s)
        return best_s.reshape(xy.shape[:-1])

    def lookahead_point(self, x: float, y: float, distance: float) -> Tuple[np.ndarray, float]:
        """Point on the path ``distance`` metres (euclidean) ahead of the robot.

        Mirrors nav2_util::getLookAheadPoint: the first path point beyond the
        lookahead circle, interpolated on the circle. Returns (point, s).
        """
        s0, _, _ = self.project(x, y)
        p = np.array([x, y])
        # sample the remaining path densely and find first point outside circle
        if distance <= 0:
            return self.point_at(s0), s0
        ss = np.linspace(s0, self.length, max(2, int((self.length - s0) / 0.02) + 2))
        pts = self.points_at(ss)
        d = np.hypot(*(pts - p).T)
        idx = np.nonzero(d >= distance)[0]
        if len(idx) == 0:
            return self.points[-1].copy(), self.length
        k = int(idx[0])
        if k == 0:
            return pts[0], float(ss[0])
        # interpolate between k-1 and k on the circle
        a, b = pts[k - 1], pts[k]
        da, db = d[k - 1], d[k]
        if db - da < 1e-9:
            return b, float(ss[k])
        r = (distance - da) / (db - da)
        return a + r * (b - a), float(ss[k - 1] + r * (ss[k] - ss[k - 1]))

    def remaining(self, x: float, y: float) -> float:
        s, _, _ = self.project(x, y)
        return self.length - s


def segment_points(a: Iterable[float], b: Iterable[float], spacing: float) -> np.ndarray:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = max(2, int(math.ceil(np.hypot(*(b - a)) / spacing)) + 1)
    return np.linspace(a, b, n)
