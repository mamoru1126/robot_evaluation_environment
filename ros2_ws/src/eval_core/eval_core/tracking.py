"""Simple LiDAR people tracker (perception mode ``scan_tracking``).

1. Euclidean clustering of the scan points (in the world frame).
2. Clusters smaller than ``max_person_size`` are person candidates
   (two legs merge into one cluster); bigger ones (walls) stay static points.
3. Nearest-neighbour association + alpha-beta filter -> position & velocity.
4. Tracks that move faster than ``min_speed`` become dynamic agents that the
   controllers can predict; everything else is handed over as static points.

This is intentionally simple (no occlusion handling, no multi-hypothesis),
i.e. roughly what a leg tracker on a real robot provides.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .obstacles import DynamicAgent


@dataclass
class Track:
    id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 0.2
    t_last: float = 0.0
    t_first: float = 0.0
    hits: int = 1


@dataclass
class ScanTracker:
    cluster_dist: float = 0.35       # points closer than this belong to the same cluster [m]
    max_person_size: float = 0.8     # max cluster diameter treated as a person [m]
    gate: float = 0.8                # association gate [m]
    alpha: float = 0.5
    beta: float = 0.3
    min_speed: float = 0.25          # tracks faster than this are dynamic [m/s]
    min_hits: int = 3
    timeout: float = 0.6             # drop tracks not seen for this long [s]
    tracks: List[Track] = field(default_factory=list)
    _next_id: int = 1

    def reset(self) -> None:
        self.tracks = []

    @staticmethod
    def cluster(points: np.ndarray, dist: float) -> List[np.ndarray]:
        """Connected components with a distance threshold (grid hashing, O(N))."""
        n = len(points)
        if n == 0:
            return []
        cell = {}
        keys = np.floor(points / dist).astype(int)
        for i, (a, b) in enumerate(keys):
            cell.setdefault((a, b), []).append(i)
        labels = -np.ones(n, int)
        lab = 0
        d2 = dist * dist
        for i in range(n):
            if labels[i] >= 0:
                continue
            labels[i] = lab
            stack = [i]
            while stack:
                j = stack.pop()
                a, b = keys[j]
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        for k in cell.get((a + da, b + db), ()):
                            if labels[k] < 0:
                                dx = points[k, 0] - points[j, 0]
                                dy = points[k, 1] - points[j, 1]
                                if dx * dx + dy * dy <= d2:
                                    labels[k] = lab
                                    stack.append(k)
            lab += 1
        return [points[labels == L] for L in range(lab)]

    def update(self, t: float, points: np.ndarray) -> Tuple[np.ndarray, List[DynamicAgent]]:
        """Returns (static_points, dynamic_agents)."""
        clusters = self.cluster(points, self.cluster_dist)
        static = []
        dets = []        # (cx, cy, radius, points)
        for c in clusters:
            ext = float(np.max(np.hypot(*(c - c.mean(axis=0)).T))) if len(c) > 1 else 0.0
            if 2 * ext <= self.max_person_size and len(c) >= 2:
                cx, cy = c.mean(axis=0)
                dets.append((float(cx), float(cy), ext, c))
            else:
                static.append(c)
        # --- association (greedy nearest neighbour on predicted positions) ---
        used = set()
        for tr in sorted(self.tracks, key=lambda q: -q.hits):
            dt = max(t - tr.t_last, 1e-3)
            px, py = tr.x + tr.vx * dt, tr.y + tr.vy * dt
            best, best_d = None, self.gate
            for k, (cx, cy, _, _) in enumerate(dets):
                if k in used:
                    continue
                d = math.hypot(cx - px, cy - py)
                if d < best_d:
                    best, best_d = k, d
            if best is None:
                continue
            used.add(best)
            cx, cy, ext, _ = dets[best]
            rx, ry = cx - px, cy - py
            tr.x = px + self.alpha * rx
            tr.y = py + self.alpha * ry
            tr.vx += self.beta * rx / dt
            tr.vy += self.beta * ry / dt
            sp = math.hypot(tr.vx, tr.vy)
            if sp > 3.0:
                tr.vx, tr.vy = tr.vx / sp * 3.0, tr.vy / sp * 3.0
            tr.radius = 0.7 * tr.radius + 0.3 * ext
            tr.t_last = t
            tr.hits += 1
        for k, (cx, cy, ext, _) in enumerate(dets):
            if k not in used:
                self.tracks.append(Track(self._next_id, cx, cy, radius=ext, t_last=t, t_first=t))
                self._next_id += 1
        self.tracks = [tr for tr in self.tracks if t - tr.t_last <= self.timeout]
        # --- output -----------------------------------------------------------
        agents: List[DynamicAgent] = []
        moving_ids = set()
        for tr in self.tracks:
            if tr.hits >= self.min_hits and math.hypot(tr.vx, tr.vy) >= self.min_speed and tr.t_last == t:
                agents.append(DynamicAgent(tr.x, tr.y, tr.vx, tr.vy, tr.radius, f"track_{tr.id}"))
                moving_ids.add((round(tr.x, 3), round(tr.y, 3)))
        for cx, cy, ext, c in dets:
            # detections that belong to a moving track are represented by the agent
            if any(math.hypot(cx - a.x, cy - a.y) < 0.4 for a in agents):
                continue
            static.append(c)
        static_pts = np.concatenate(static) if static else np.zeros((0, 2))
        return static_pts, agents
