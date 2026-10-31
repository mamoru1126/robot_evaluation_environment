import math

import numpy as np
import pytest

from eval_core.metrics import aggregate, compute, ttc_series
from eval_core.scenario import DEFAULT_SCENARIO, normalize_scenario
from eval_core.sim2d import Sim2D
from eval_core.specs import RobotSpec, WorldSpec
from eval_core.tracking import ScanTracker
from eval_core.planner_core import scan_to_points

COLS = ["t", "x", "y", "yaw", "v", "w", "cmd_v", "cmd_w", "compute_ms", "blocked",
        "h1_x", "h1_y", "h1_vx", "h1_vy", "h2_x", "h2_y", "h2_vx", "h2_vy"]


def _synthetic():
    """Robot drives from x=-1 to 11 at 1 m/s; human 1 stands at (5, 1); human 2 disabled."""
    t = np.arange(0, 12.01, 0.05)
    x = -1.0 + t
    n = len(t)
    data = np.zeros((n, len(COLS)))
    data[:, 0] = t
    data[:, 1] = x
    data[:, 4] = 1.0
    data[:, 6] = 1.0
    data[:, 8] = 2.0
    data[:, 10] = 5.0
    data[:, 11] = 1.0
    data[:, 14:18] = np.nan
    sc = normalize_scenario(DEFAULT_SCENARIO, 2)
    sc["humans"][0]["enabled"] = True
    sc["humans"][1]["enabled"] = False
    return data, sc


def test_travel_time_and_distances():
    data, sc = _synthetic()
    m = compute(data, COLS, sc, 0.25, "goal", 0)
    assert m["travel_time"] == pytest.approx(10.0, abs=1e-6)       # START x=0 at t=1, GOAL x=10 at t=11
    assert m["mean_speed"] == pytest.approx(1.0)
    assert m["min_center_distance"] == pytest.approx(1.0)
    assert m["min_clearance"] == pytest.approx(0.5)
    # personal space 1.2 m: |x-5| < sqrt(1.2^2-1) = 0.663 -> 1.327 s
    assert m["personal_space_time"] == pytest.approx(2 * math.sqrt(1.2 ** 2 - 1.0), abs=0.06)
    assert m["intimate_space_time"] == pytest.approx(0.0)
    assert m["success"] is True
    assert m["humans"][0]["t_min_distance"] == pytest.approx(6.0)
    assert m["compute_ms_mean"] == pytest.approx(2.0)


def test_ttc():
    p = np.array([[5.0, 0.0], [5.0, 3.0]])
    u = np.array([[-1.0, 0.0], [-1.0, 0.0]])
    t = ttc_series(p, u, 0.5)
    assert t[0] == pytest.approx(4.5)
    assert t[1] == pytest.approx(10.0)      # never collides -> capped


def test_aggregate():
    rows = [{"controller": "a", "travel_time": 10.0, "success": True},
            {"controller": "a", "travel_time": 12.0, "success": False},
            {"controller": "b", "travel_time": 8.0, "success": True}]
    g = {x["controller"]: x for x in aggregate(rows)}
    assert g["a"]["travel_time"]["mean"] == pytest.approx(11.0)
    assert g["a"]["success_rate"] == pytest.approx(0.5)
    assert g["b"]["n"] == 1


def test_tracker_estimates_velocity_of_walking_human():
    robot, world = RobotSpec(), WorldSpec()
    sim = Sim2D(robot, world, num_humans=1, seed=1)
    sim.teleport_robot(0, 0, 0)
    sim.teleport_human(0, 4.0, 3.0, -math.pi / 2)
    sim.set_human_cmd(0, 0.0, -1.0)
    tr = ScanTracker()
    agents = []
    for k in range(200):
        sim.step(0.01)
        if k % 10 == 0:
            ranges, amin, inc, rmin, rmax = sim.scan()
            pts = scan_to_points(ranges, amin, inc, rmin, rmax, (0.0, 0.0, 0.0))
            _, agents = tr.update(sim.t, pts)
    assert len(agents) == 1
    a = agents[0]
    assert a.vy == pytest.approx(-1.0, abs=0.25)
    assert abs(a.vx) < 0.3
    assert math.hypot(a.x - sim.humans[0].x, a.y - sim.humans[0].y) < 0.25
