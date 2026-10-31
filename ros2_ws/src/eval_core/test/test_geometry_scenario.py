import math

import numpy as np
import pytest

from eval_core.geometry import Path, crossing_time, to_robot_frame, wrap_angle
from eval_core.scenario import (DEFAULT_SCENARIO, crossing_distance, evaluation_path, human_route,
                                list_presets, normalize_scenario, robot_start_pose)


def test_wrap_angle():
    assert wrap_angle(3 * math.pi) == pytest.approx(-math.pi)
    a = wrap_angle(np.array([0.0, 2 * math.pi + 0.1, -math.pi - 0.1]))
    assert a == pytest.approx([0.0, 0.1, math.pi - 0.1])


def test_path_projection_and_lookahead():
    p = Path([[0, 0], [10, 0]])
    s, lat, _ = p.project(3.0, 0.5)
    assert s == pytest.approx(3.0)
    assert lat == pytest.approx(0.5)        # left of the path is positive
    pt, s2 = p.lookahead_point(3.0, 0.5, 1.0)
    assert math.hypot(pt[0] - 3.0, pt[1] - 0.5) == pytest.approx(1.0, abs=0.02)
    assert pt[1] == pytest.approx(0.0)
    assert p.remaining(4.0, 0.0) == pytest.approx(6.0)
    assert p.signed_lateral(np.array([[1.0, -0.3]]))[0] == pytest.approx(-0.3)


def test_polyline_progress():
    p = Path([[0, 0], [5, 0], [5, 5]])
    assert p.length == pytest.approx(10.0)
    assert p.progress(np.array([[5.2, 2.0]]))[0] == pytest.approx(7.0)
    assert p.heading_at(7.0) == pytest.approx(math.pi / 2)


def test_crossing_time():
    assert crossing_time(1.0, -0.5, 2.0, 0.5, 0.0) == pytest.approx(1.5)
    assert crossing_time(1.0, 0.5, 2.0, 0.6, 0.0) is None


def test_robot_frame():
    x, y = to_robot_frame(1.0, 1.0, 0.0, 0.0, math.pi / 2)
    assert float(x) == pytest.approx(1.0)
    assert float(y) == pytest.approx(-1.0)


def test_normalize_defaults_and_clamp():
    sc = normalize_scenario({"course": {"goal_x": -5}, "humans": [{"speed": 99, "pattern": "nope"}]}, 2)
    assert sc["course"]["goal_x"] > sc["course"]["start_x"]
    assert sc["humans"][0]["speed"] == 3.0
    assert sc["humans"][0]["pattern"] == "crossing"
    assert sc["humans"][1]["enabled"] is False
    assert [h["id"] for h in sc["humans"]] == ["human_1", "human_2"]


def test_crossing_route_geometry():
    sc = normalize_scenario(DEFAULT_SCENARIO, 2)
    h = dict(sc["humans"][0], pattern="crossing", cross_x=5.0, from_side="left", lateral_start=3.0,
             lateral_end=2.0, angle_deg=90.0)
    (x0, y0), (x1, y1) = human_route(h, sc["course"])
    assert (x0, y0) == pytest.approx((5.0, 3.0))
    assert (x1, y1) == pytest.approx((5.0, -2.0))
    h = dict(h, from_side="right", angle_deg=60.0)
    (x0, y0), (x1, y1) = human_route(h, sc["course"])
    assert y0 == pytest.approx(-3.0)
    assert x0 < 5.0 < x1                          # moves forward (same direction as the robot)
    assert crossing_distance(h) == pytest.approx(3.0 / math.sin(math.radians(60)))


def test_head_on_route_and_course():
    sc = normalize_scenario(DEFAULT_SCENARIO, 2)
    h = dict(sc["humans"][1], pattern="head_on", start_x=12.0, lateral_offset=0.5, travel_distance=10.0)
    assert human_route(h, sc["course"]) == [[12.0, 0.5], [2.0, 0.5]]
    x, y, yaw = robot_start_pose(sc)
    assert (x, y, yaw) == (sc["course"]["start_x"] - sc["course"]["robot_start_offset"], 0.0, 0.0)
    path = evaluation_path(sc)
    assert path[-1][0] == pytest.approx(sc["course"]["goal_x"] + sc["course"]["goal_extension"])


def test_presets_load():
    presets = {p["name"]: p for p in list_presets()}
    for name in ("crossing_and_head_on", "head_on_two", "no_humans"):
        assert name in presets
        assert len(presets[name]["humans"]) == 2
