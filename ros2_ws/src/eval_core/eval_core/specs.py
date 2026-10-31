"""Robot / world specifications shared by Gazebo world generation,
the 2D simulator and the controllers.

Values are read from ``<config_dir>/robot.yaml`` and ``<config_dir>/world.yaml``
when they exist; otherwise the defaults below are used.  ``config_dir`` is
``$EVAL_CONFIG_DIR`` (``/workspace/config`` inside Docker).
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path as FsPath
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is available in ROS images
    yaml = None


def config_dir() -> FsPath:
    env = os.environ.get("EVAL_CONFIG_DIR")
    if env:
        return FsPath(env)
    # source checkout: <repo>/ros2_ws/src/eval_core/eval_core/specs.py -> <repo>/config
    here = FsPath(__file__).resolve()
    for parent in here.parents:
        cand = parent / "config"
        if (cand / "scenarios").is_dir():
            return cand
    return FsPath("/workspace/config")


def _load_yaml(path: FsPath) -> Dict[str, Any]:
    if yaml is None or not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _from_dict(cls, data: Dict[str, Any]):
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in (data or {}).items() if k in names})


@dataclass
class LidarSpec:
    z: float = 0.30            # mounting height above ground [m]
    x: float = 0.0             # forward offset from robot centre [m]
    samples: int = 360
    min_angle: float = -3.14159
    max_angle: float = 3.14159
    range_min: float = 0.10
    range_max: float = 12.0
    rate: float = 10.0
    noise_std: float = 0.01


@dataclass
class RobotSpec:
    """Cylindrical differential-drive serving robot."""
    radius: float = 0.25        # footprint radius [m]
    height: float = 1.10
    mass: float = 30.0
    max_v: float = 1.0          # [m/s]
    min_v: float = 0.0          # no reversing by default
    max_w: float = 2.0          # [rad/s]
    max_acc: float = 1.0        # linear acceleration [m/s^2]
    max_dec: float = 1.5        # linear deceleration (positive number) [m/s^2]
    max_alpha: float = 3.0      # angular acceleration [rad/s^2]
    wheel_separation: float = 0.36
    wheel_radius: float = 0.075
    lidar: LidarSpec = field(default_factory=LidarSpec)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RobotSpec":
        data = dict(data or {})
        lidar = _from_dict(LidarSpec, data.pop("lidar", {}) or {})
        spec = _from_dict(cls, data)
        spec.lidar = lidar
        return spec

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class WorldSpec:
    """Straight corridor / hall used for the evaluation."""
    name: str = "eval_world"
    walls: bool = True
    corridor_width: float = 8.0   # distance between the two side walls [m]
    x_min: float = -6.0           # wall extent along x [m]
    x_max: float = 30.0
    wall_thickness: float = 0.2
    wall_height: float = 1.0
    max_humans: int = 2           # number of human models spawned in Gazebo

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "WorldSpec":
        return _from_dict(cls, data or {})

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def wall_segments(self) -> List[List[List[float]]]:
        """Inner faces of the side walls as line segments [[x0,y0],[x1,y1]]."""
        if not self.walls:
            return []
        h = self.corridor_width / 2.0
        return [[[self.x_min, h], [self.x_max, h]], [[self.x_min, -h], [self.x_max, -h]]]


def load_robot_spec(cfg: Optional[FsPath] = None) -> RobotSpec:
    cfg = cfg or config_dir()
    return RobotSpec.from_dict(_load_yaml(FsPath(cfg) / "robot.yaml"))


def load_world_spec(cfg: Optional[FsPath] = None) -> WorldSpec:
    cfg = cfg or config_dir()
    return WorldSpec.from_dict(_load_yaml(FsPath(cfg) / "world.yaml"))
