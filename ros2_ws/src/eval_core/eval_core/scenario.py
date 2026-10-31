"""Scenario definition: course (START/GOAL lines), human behaviours,
termination and evaluation settings.

A scenario is a plain JSON-compatible dict so it can be edited in the web UI,
stored as a preset in ``config/scenarios/*.json`` and saved with results.
"""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path as FsPath
from typing import Any, Dict, List, Optional, Tuple

from .specs import config_dir

PATTERNS = ["crossing", "head_on", "same_direction", "static", "waypoints"]
PATTERN_LABELS = {
    "crossing": "横切る (Crossing)",
    "head_on": "対向 (Head-on)",
    "same_direction": "同方向・追い越し (Same direction)",
    "static": "静止 (Static)",
    "waypoints": "ウェイポイント (Custom)",
}
TRIGGERS = ["time", "robot_x", "encounter"]
TRIGGER_LABELS = {
    "time": "時刻 (試行開始からの秒数)",
    "robot_x": "ロボットが x 座標を通過",
    "encounter": "出会い頭 (横切り時に到達時刻を合わせる)",
}

DEFAULT_HUMAN: Dict[str, Any] = {
    "enabled": True,
    "pattern": "crossing",
    "speed": 1.0,             # walking speed [m/s]
    "radius": 0.25,           # body radius used for distance metrics [m]
    # crossing
    "cross_x": 5.0,           # x coordinate where the human crosses the robot path
    "from_side": "left",      # "left" (+y, robot's left) or "right" (-y)
    "lateral_start": 3.0,     # lateral distance from the path at the start [m]
    "lateral_end": 3.0,       # lateral distance past the path at the end [m]
    "angle_deg": 90.0,        # angle between robot heading and walking direction
    # head_on / same_direction
    "start_x": 12.0,
    "lateral_offset": 0.0,    # +y offset from the robot path [m]
    "travel_distance": 14.0,
    # static
    "x": 5.0,
    "y": 0.0,
    # waypoints
    "waypoints": [[8.0, 2.0], [4.0, -1.0]],
    "loop": False,
    # trigger
    "trigger": "encounter",
    "trigger_time": 0.0,
    "trigger_robot_x": 0.0,
    "encounter_offset": 0.0,  # [s] >0: human reaches the crossing point later than the robot
    # reaction
    "reactive": False,        # simple social-force style avoidance of the robot
}

DEFAULT_SCENARIO: Dict[str, Any] = {
    "name": "crossing_and_head_on",
    "description": "1人目が左から横切り、2人目が正面から対向してくる標準シナリオ",
    "course": {
        "start_x": 0.0,             # START line x [m]
        "goal_x": 10.0,             # GOAL line x [m]
        "y": 0.0,                   # y of the straight path [m]
        "robot_start_offset": 1.0,  # robot starts this far behind the START line [m]
        "goal_extension": 2.0,      # path continues this far beyond GOAL [m]
    },
    "humans": [
        dict(DEFAULT_HUMAN, pattern="crossing", cross_x=4.0, from_side="left", speed=1.0,
             trigger="encounter"),
        dict(DEFAULT_HUMAN, pattern="head_on", start_x=13.0, lateral_offset=0.3, speed=1.0,
             trigger="time", trigger_time=0.0),
    ],
    "termination": {"timeout": 60.0, "stop_on_collision": True},
    "evaluation": {"personal_space": 1.2, "intimate_space": 0.45},
}


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (update or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _f(v, default: float) -> float:
    try:
        x = float(v)
        if math.isfinite(x):
            return x
    except (TypeError, ValueError):
        pass
    return float(default)


def normalize_human(h: Optional[Dict[str, Any]], index: int) -> Dict[str, Any]:
    out = _deep_merge(DEFAULT_HUMAN, h or {})
    out["id"] = f"human_{index + 1}"
    out["enabled"] = bool(out.get("enabled", True))
    if out["pattern"] not in PATTERNS:
        out["pattern"] = "crossing"
    if out["trigger"] not in TRIGGERS:
        out["trigger"] = "time"
    if out["from_side"] not in ("left", "right"):
        out["from_side"] = "left"
    for key in ("speed", "radius", "cross_x", "lateral_start", "lateral_end", "angle_deg",
                "start_x", "lateral_offset", "travel_distance", "x", "y", "trigger_time",
                "trigger_robot_x", "encounter_offset"):
        out[key] = _f(out.get(key), DEFAULT_HUMAN[key])
    out["speed"] = min(max(out["speed"], 0.0), 3.0)
    out["radius"] = min(max(out["radius"], 0.05), 1.0)
    out["angle_deg"] = min(max(out["angle_deg"], 15.0), 165.0)
    out["lateral_start"] = max(out["lateral_start"], 0.1)
    out["lateral_end"] = max(out["lateral_end"], 0.0)
    out["travel_distance"] = max(out["travel_distance"], 0.0)
    out["trigger_time"] = max(out["trigger_time"], 0.0)
    wps = []
    for p in out.get("waypoints") or []:
        try:
            wps.append([float(p[0]), float(p[1])])
        except (TypeError, ValueError, IndexError):
            continue
    out["waypoints"] = wps or copy.deepcopy(DEFAULT_HUMAN["waypoints"])
    out["loop"] = bool(out.get("loop", False))
    out["reactive"] = bool(out.get("reactive", False))
    return out


def normalize_scenario(sc: Optional[Dict[str, Any]], num_humans: Optional[int] = None) -> Dict[str, Any]:
    """Fill defaults, coerce types and clamp values."""
    sc = sc or {}
    out = _deep_merge({k: v for k, v in DEFAULT_SCENARIO.items() if k != "humans"},
                      {k: v for k, v in sc.items() if k != "humans"})
    humans_in = sc.get("humans", DEFAULT_SCENARIO["humans"]) or []
    n = len(humans_in) if num_humans is None else num_humans
    humans = []
    for i in range(n):
        h = humans_in[i] if i < len(humans_in) else {"enabled": False}
        humans.append(normalize_human(h, i))
    out["humans"] = humans
    c = out["course"]
    for key in ("start_x", "goal_x", "y", "robot_start_offset", "goal_extension"):
        c[key] = _f(c.get(key), DEFAULT_SCENARIO["course"][key])
    if c["goal_x"] <= c["start_x"] + 0.5:
        c["goal_x"] = c["start_x"] + 0.5
    c["robot_start_offset"] = max(c["robot_start_offset"], 0.0)
    c["goal_extension"] = max(c["goal_extension"], 0.3)
    t = out["termination"]
    t["timeout"] = max(_f(t.get("timeout"), 60.0), 1.0)
    t["stop_on_collision"] = bool(t.get("stop_on_collision", True))
    e = out["evaluation"]
    e["personal_space"] = max(_f(e.get("personal_space"), 1.2), 0.0)
    e["intimate_space"] = max(_f(e.get("intimate_space"), 0.45), 0.0)
    out["name"] = str(out.get("name") or "scenario")
    out["description"] = str(out.get("description") or "")
    return out


# ----------------------------------------------------------------------------
# Course helpers
# ----------------------------------------------------------------------------
def robot_start_pose(sc: Dict[str, Any]) -> Tuple[float, float, float]:
    c = sc["course"]
    return c["start_x"] - c["robot_start_offset"], c["y"], 0.0


def evaluation_path(sc: Dict[str, Any]) -> List[List[float]]:
    """Straight path from the robot start pose to beyond the GOAL line."""
    c = sc["course"]
    x0, y0, _ = robot_start_pose(sc)
    return [[x0, y0], [c["goal_x"] + c["goal_extension"], c["y"]]]


# ----------------------------------------------------------------------------
# Human route geometry
# ----------------------------------------------------------------------------
def human_route(h: Dict[str, Any], course: Dict[str, Any]) -> List[List[float]]:
    """Polyline the human walks along (first point = initial position)."""
    py = course["y"]
    pat = h["pattern"]
    if pat == "crossing":
        a = math.radians(h["angle_deg"])
        side = 1.0 if h["from_side"] == "left" else -1.0
        d = (math.cos(a), -side * math.sin(a))
        cx, cy = h["cross_x"], py
        k0 = h["lateral_start"] / math.sin(a)
        k1 = h["lateral_end"] / math.sin(a)
        return [[cx - d[0] * k0, cy - d[1] * k0], [cx + d[0] * k1, cy + d[1] * k1]]
    if pat == "head_on":
        y = py + h["lateral_offset"]
        return [[h["start_x"], y], [h["start_x"] - h["travel_distance"], y]]
    if pat == "same_direction":
        y = py + h["lateral_offset"]
        return [[h["start_x"], y], [h["start_x"] + h["travel_distance"], y]]
    if pat == "static":
        return [[h["x"], h["y"]], [h["x"], h["y"]]]
    wps = h.get("waypoints") or [[h["x"], h["y"]]]
    if len(wps) == 1:
        wps = [wps[0], wps[0]]
    return [list(map(float, p)) for p in wps]


def human_initial_pose(h: Dict[str, Any], course: Dict[str, Any]) -> Tuple[float, float, float]:
    route = human_route(h, course)
    (x0, y0), (x1, y1) = route[0], route[1]
    yaw = math.atan2(y1 - y0, x1 - x0) if math.hypot(x1 - x0, y1 - y0) > 1e-6 else math.pi
    return x0, y0, yaw


def crossing_distance(h: Dict[str, Any]) -> float:
    """Walking distance from the start to the crossing point (crossing pattern)."""
    return h["lateral_start"] / math.sin(math.radians(h["angle_deg"]))


# ----------------------------------------------------------------------------
# Preset storage
# ----------------------------------------------------------------------------
_SAFE = re.compile(r"[^0-9A-Za-z_\-]+")


def safe_name(name: str) -> str:
    s = _SAFE.sub("_", (name or "").strip()).strip("_")
    return s[:64] or "scenario"


def scenarios_dir() -> FsPath:
    return config_dir() / "scenarios"


def list_presets() -> List[Dict[str, Any]]:
    d = scenarios_dir()
    out = []
    if d.is_dir():
        for p in sorted(d.glob("*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data = normalize_scenario(data)
                data["name"] = p.stem
                out.append(data)
            except (OSError, ValueError):
                continue
    return out


def save_preset(sc: Dict[str, Any]) -> str:
    sc = normalize_scenario(sc)
    name = safe_name(sc.get("name", "scenario"))
    sc["name"] = name
    d = scenarios_dir()
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(sc, f, ensure_ascii=False, indent=2)
    return name


def delete_preset(name: str) -> bool:
    p = scenarios_dir() / f"{safe_name(name)}.json"
    if p.is_file():
        p.unlink()
        return True
    return False
