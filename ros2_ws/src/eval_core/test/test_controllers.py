import math
import os
import tempfile

import numpy as np
import pytest

from eval_core.backends.sim2d_backend import Sim2DBackend
from eval_core.controllers import BUILTIN, create, discover
from eval_core.controllers.base import ControllerInput, RobotState
from eval_core.controllers.mpc import MPCController
from eval_core.controllers.pure_pursuit import (compute_dynamic_window, compute_dynamic_window_velocities,
                                                compute_optimal_velocity_within_dynamic_window)
from eval_core.geometry import Path
from eval_core.obstacles import DynamicAgent, ObstacleSet
from eval_core.scenario import list_presets
from eval_core.specs import RobotSpec, load_robot_spec, load_world_spec
from eval_core.storage import ResultStore


# ----------------------------------------------------------------------------- DWPP (port of nav2 functions)
def test_dynamic_window_bounds():
    # moving forward at 0.5 m/s, dt = 0.1, acc 1.0, dec -2.0
    dw = compute_dynamic_window(0.5, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0, -2.0, 2.0, -2.0, 0.1)
    assert dw == pytest.approx([0.6, 0.3, 0.2, -0.2])
    # standing still: symmetric window, clipped by min velocity
    dw = compute_dynamic_window(0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0, -2.0, 2.0, -2.0, 0.1)
    assert dw == pytest.approx([0.1, 0.0, 0.2, -0.2])


def test_dwpp_line_intersects_window():
    # curvature 0.5: w = 0.5 v ; window v in [0.3, 0.6], w in [-0.2, 0.2] -> intersection at v=0.4 (w=0.2)
    v, w = compute_optimal_velocity_within_dynamic_window([0.6, 0.3, 0.2, -0.2], 0.5, 1.0)
    assert (v, w) == pytest.approx((0.4, 0.2))
    # straight path keeps w = 0 and the highest v
    v, w = compute_optimal_velocity_within_dynamic_window([0.6, 0.3, 0.2, -0.2], 0.0, 1.0)
    assert (v, w) == pytest.approx((0.6, 0.0))


def test_dwpp_no_intersection_uses_closest_corner():
    # curvature 5: line w = 5 v leaves the window [0.3,0.6]x[-0.2,0.2]; closest corner is (0.3, 0.2)
    v, w = compute_optimal_velocity_within_dynamic_window([0.6, 0.3, 0.2, -0.2], 5.0, 1.0)
    assert (v, w) == pytest.approx((0.3, 0.2))


def test_dwpp_regulation_collapses_window():
    # regulated speed 0.1 is below the reachable window -> v is the lowest reachable speed
    v, _ = compute_dynamic_window_velocities(0.5, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0, -2.0, 2.0, -2.0, 0.1, 0.0, 1.0, 0.1)
    assert v == pytest.approx(0.3)


# ----------------------------------------------------------------------------- MPC gradient
def test_mpc_gradient_matches_finite_differences():
    robot = RobotSpec()
    c = MPCController(robot, {"horizon_steps": 8, "dt": 0.15})
    path = Path([[0, 0], [10, 0]])
    agents = [DynamicAgent(2.0, 0.2, -0.8, 0.0, 0.25)]
    obs = ObstacleSet(np.array([[1.5, -0.6]]), agents, robot.radius, (0, 0), point_radius=0.1)
    inp = ControllerInput(RobotState(0, 0.0, 0.05, 0.1, 0.4, 0.0), path, obs, 0.1, (0.4, 0.0))
    N, dt, ref, psi, opos, orad = c._setup(inp)
    rng = np.random.default_rng(3)
    z = np.concatenate([rng.uniform(0.2, 0.8, N), rng.uniform(-0.5, 0.5, N)])
    args = (0.0, 0.05, 0.1, 0.4, 0.0, N, dt, ref, psi, opos, orad)
    J, g, _ = c._cost_grad(z, *args)
    eps = 1e-6
    num = np.zeros_like(z)
    for k in range(len(z)):
        zp = z.copy()
        zm = z.copy()
        zp[k] += eps
        zm[k] -= eps
        num[k] = (c._cost_grad(zp, *args)[0] - c._cost_grad(zm, *args)[0]) / (2 * eps)
    assert np.allclose(g, num, rtol=1e-4, atol=1e-5)


# ----------------------------------------------------------------------------- registry
def test_registry_contains_required_controllers():
    for name in ("dwa", "pure_pursuit", "mpc", "mppi", "dwpp"):
        assert name in BUILTIN
    found, errors = discover()
    assert not errors, errors
    assert "example_stop_and_go" in found


def test_controller_param_coercion():
    c = create("dwa", RobotSpec(), {"v_samples": "7", "w_obstacle": "nan", "use_measured_velocity": "false"})
    assert c.p["v_samples"] == 7
    assert c.p["w_obstacle"] == c.PARAMS["w_obstacle"].default
    assert c.p["use_measured_velocity"] is False


# ----------------------------------------------------------------------------- closed loop in the 2D simulator
def _run(controllers, scenario, perception, tmp):
    robot = load_robot_spec()
    world = load_world_spec()
    presets = {p["name"]: p for p in list_presets()}
    be = Sim2DBackend(robot, world, ResultStore(tmp))
    be.start_batch({"name": "t", "scenarios": [presets[scenario]],
                    "controllers": [{"name": n, "params": {"seed": 7} if n == "mppi" else {}} for n in controllers],
                    "repeats": 1, "perception": perception})
    be.thread.join(600)
    return {r["summary"]["controller"]: r["summary"] for r in be.manager.finished_runs}


@pytest.mark.parametrize("perception", ["scan", "ground_truth"])
def test_all_controllers_reach_goal_without_humans(tmp_path, perception):
    res = _run(["pure_pursuit", "dwpp", "dwa", "mppi", "mpc"], "no_humans", perception, tmp_path)
    for name, s in res.items():
        assert s["outcome"] == "goal", (name, s.get("message"))
        assert s["travel_time"] == pytest.approx(10.0 / 0.8, rel=0.15), name
        assert s["max_lateral_deviation"] < 0.1, name


def test_predictive_controllers_avoid_crossing_and_head_on(tmp_path):
    res = _run(["dwa", "mppi", "mpc"], "crossing_and_head_on", "ground_truth", tmp_path)
    for name, s in res.items():
        assert s["outcome"] == "goal", (name, s.get("message"))
        assert s["collisions"] == 0
        assert s["min_clearance"] > 0.0


def test_path_trackers_stop_instead_of_avoiding(tmp_path):
    # pure pursuit / DWPP do not avoid: they stop (blocked) when a collision is imminent
    res = _run(["pure_pursuit", "dwpp"], "overtake_slow_walker", "ground_truth", tmp_path)
    for name, s in res.items():
        assert s["outcome"] == "goal", (name, s.get("message"))
        assert s["max_lateral_deviation"] < 0.05
        assert s["min_clearance"] > 0.0
