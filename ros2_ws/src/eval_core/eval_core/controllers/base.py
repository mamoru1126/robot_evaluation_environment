"""Common interface for all local planners / path-tracking controllers.

Writing your own controller
---------------------------
Create a ``.py`` file in ``custom_controllers/`` (mounted into the container)
that defines a subclass of :class:`BaseController`::

    from eval_core.controllers.base import BaseController, ControlOutput, Param

    class MyController(BaseController):
        name = "my_controller"            # unique key
        label = "My controller"           # shown in the web UI
        PARAMS = {"target_speed": Param(0.8, 0.1, 1.5, 0.05, "目標速度 [m/s]")}

        def compute(self, inp):
            st = inp.state
            ...
            return ControlOutput(v, w, trajectory=[[x, y], ...])

The class is discovered automatically (no rebuild required) and its
``PARAMS`` are rendered as an editable form in the web UI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geometry import Path
from ..obstacles import ObstacleSet
from ..specs import RobotSpec


@dataclass
class RobotState:
    t: float
    x: float
    y: float
    yaw: float
    v: float = 0.0   # measured forward velocity [m/s]
    w: float = 0.0   # measured yaw rate [rad/s]


@dataclass
class ControllerInput:
    state: RobotState
    path: Path
    obstacles: ObstacleSet
    dt: float                                   # control period [s]
    last_cmd: Tuple[float, float] = (0.0, 0.0)  # previously published (v, w)


@dataclass
class ControlOutput:
    v: float
    w: float
    trajectory: Optional[List[List[float]]] = None   # predicted / selected trajectory (world frame)
    candidates: Optional[List[List[List[float]]]] = None  # optional extra trajectories for display
    info: Dict[str, Any] = field(default_factory=dict)


class Param:
    """Parameter description used to build the web UI form."""

    def __init__(self, default, min=None, max=None, step=None, label: str = "", help: str = "",
                 type: Optional[str] = None, choices: Optional[Sequence[str]] = None):
        self.default = default
        self.min = min
        self.max = max
        self.step = step
        self.label = label
        self.help = help
        if type is None:
            if isinstance(default, bool):
                type = "bool"
            elif isinstance(default, int):
                type = "int"
            elif isinstance(default, str):
                type = "choice" if choices else "str"
            else:
                type = "float"
        self.type = type
        self.choices = list(choices) if choices else None

    def coerce(self, value):
        if value is None:
            return self.default
        try:
            if self.type == "bool":
                if isinstance(value, str):
                    return value.strip().lower() in ("1", "true", "yes", "on")
                return bool(value)
            if self.type == "int":
                v = int(round(float(value)))
            elif self.type == "float":
                v = float(value)
                if not math.isfinite(v):
                    return self.default
            else:
                v = str(value)
                if self.choices and v not in self.choices:
                    return self.default
                return v
        except (TypeError, ValueError):
            return self.default
        if self.min is not None:
            v = max(v, self.min)
        if self.max is not None:
            v = min(v, self.max)
        return v

    def to_dict(self, key: str) -> Dict[str, Any]:
        return {"key": key, "default": self.default, "min": self.min, "max": self.max,
                "step": self.step, "label": self.label or key, "help": self.help,
                "type": self.type, "choices": self.choices}


class BaseController:
    name: str = "base"
    label: str = "Base"
    description: str = ""
    PARAMS: Dict[str, Param] = {}

    def __init__(self, robot: RobotSpec, params: Optional[Dict[str, Any]] = None):
        self.robot = robot
        self.p: Dict[str, Any] = {}
        params = params or {}
        for key, spec in self.PARAMS.items():
            self.p[key] = spec.coerce(params.get(key, spec.default))
        self.reset()

    # ------------------------------------------------------------------
    @classmethod
    def param_schema(cls) -> List[Dict[str, Any]]:
        return [spec.to_dict(k) for k, spec in cls.PARAMS.items()]

    @classmethod
    def defaults(cls) -> Dict[str, Any]:
        return {k: s.default for k, s in cls.PARAMS.items()}

    def reset(self) -> None:
        """Called before every trial."""

    def compute(self, inp: ControllerInput) -> ControlOutput:  # pragma: no cover - interface
        raise NotImplementedError


# ----------------------------------------------------------------------------
# helpers shared by the built-in controllers
# ----------------------------------------------------------------------------
def rollout_constant(x: float, y: float, yaw: float, v: float, w: float, horizon: float,
                     dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Positions and times of a constant (v, w) arc (exact integration)."""
    n = max(1, int(round(horizon / dt)))
    t = np.arange(1, n + 1) * dt
    if abs(w) < 1e-6:
        xs = x + v * t * math.cos(yaw)
        ys = y + v * t * math.sin(yaw)
    else:
        r = v / w
        th = yaw + w * t
        xs = x + r * (np.sin(th) - math.sin(yaw))
        ys = y - r * (np.cos(th) - math.cos(yaw))
    return np.stack([xs, ys], -1), t


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v
