"""Runs the real ROS nodes (local planner, eval server Gazebo backend, scan relay)
against a fake rclpy and a fake Gazebo that mimics the gz-sim systems used in
the world (DiffDrive limiter, VelocityControl in the body frame,
OdometryPublisher world pose + body twist, UserCommands set_pose).

This checks topic names, message fields, frame conversions, the set_pose reset,
the command gating and the trial state machine end-to-end without ROS/Gazebo.
"""
import importlib
import math
import sys

import pytest

import fake_ros as F


@pytest.fixture()
def ros():
    saved = F.install()
    F.reset_bus()
    for n in [k for k in sys.modules if k.startswith("eval_core.ros")]:
        sys.modules.pop(n, None)
    yield F
    F.uninstall(saved)


class FakeGazebo(F.Node):
    def __init__(self, robot, world):
        super().__init__("fake_gazebo")
        from eval_core.sim2d import Sim2D
        self.sim = Sim2D(robot, world, num_humans=world.max_humans, seed=0)
        self.sim.teleport_robot(-1.0, 0.0, 0.0)
        self.h_cmd = [(0.0, 0.0, 0.0)] * world.max_humans
        self.lines = {}
        self.set_pose_calls = []
        self.pub_clock = self.create_publisher(F.Clock, "/clock", 10)
        self.pub_gt = self.create_publisher(F.Odometry, "/ground_truth/odom", 10)
        self.pub_h = [self.create_publisher(F.Odometry, f"/humans/human_{i + 1}/odom", 10)
                      for i in range(world.max_humans)]
        self.pub_scan = self.create_publisher(F.LaserScan, "/scan_gz", 10)
        self.create_subscription(F.Twist, "/cmd_vel", self.on_cmd, 10)
        for i in range(world.max_humans):
            self.create_subscription(F.Twist, f"/humans/human_{i + 1}/cmd_vel",
                                     lambda m, i=i: self.on_hcmd(i, m), 10)
        self.create_service(F.SetEntityPose, f"/world/{world.name}/set_pose", self.on_set_pose)
        self.k = 0

    def on_cmd(self, m):
        self.sim.set_cmd(m.linear.x, m.angular.z)

    def on_hcmd(self, i, m):
        self.h_cmd[i] = (m.linear.x, m.linear.y, m.angular.z)

    def on_set_pose(self, req, resp):
        name = req.entity.name
        assert req.entity.type == F.Entity.MODEL
        q = req.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        x, y = req.pose.position.x, req.pose.position.y
        self.set_pose_calls.append(name)
        if name == "eval_robot":
            self.sim.robot.x, self.sim.robot.y, self.sim.robot.yaw = x, y, yaw
        elif name.startswith("human_"):
            i = int(name.split("_")[1]) - 1
            h = self.sim.humans[i]
            h.x, h.y, h.yaw = x, y, yaw
        elif name in ("start_line", "goal_line"):
            self.lines[name] = x
        else:
            resp.success = False
            return resp
        resp.success = True
        return resp

    @staticmethod
    def _odom(t, x, y, yaw, vxb, vyb, wz):
        m = F.Odometry()
        st = F._Stamp(t).to_msg()
        m.header.stamp = st
        m.pose.pose.position.x = x
        m.pose.pose.position.y = y
        m.pose.pose.orientation.z = math.sin(yaw / 2)
        m.pose.pose.orientation.w = math.cos(yaw / 2)
        m.twist.twist.linear.x = vxb
        m.twist.twist.linear.y = vyb
        m.twist.twist.angular.z = wz
        return m

    def step(self, dt):
        # humans: VelocityControl -> body-frame linear velocity + yaw rate
        for h, (bx, by, wz) in zip(self.sim.humans, self.h_cmd):
            c, s = math.cos(h.yaw), math.sin(h.yaw)
            h.vx, h.vy = c * bx - s * by, s * bx + c * by
            h.x += h.vx * dt
            h.y += h.vy * dt
            h.yaw += wz * dt
        self.sim.human_cmd = [(0.0, 0.0)] * len(self.sim.humans)
        hs = [(h.x, h.y, h.yaw, h.vx, h.vy) for h in self.sim.humans]
        self.sim.step(dt)
        for h, (x, y, yaw, vx, vy) in zip(self.sim.humans, hs):   # undo sim2d's own human update
            h.x, h.y, h.yaw, h.vx, h.vy = x, y, yaw, vx, vy
        self.k += 1
        t = self.sim.t
        F.BUS.sim_time = t
        ck = F.Clock()
        ck.clock = F._Stamp(t).to_msg()
        self.pub_clock.publish(ck)
        r = self.sim.robot
        if self.k % 2 == 0:
            self.pub_gt.publish(self._odom(t, r.x, r.y, r.yaw, r.v, 0.0, r.w))
        if self.k % 3 == 0:
            for i, h in enumerate(self.sim.humans):
                c, s = math.cos(h.yaw), math.sin(h.yaw)
                self.pub_h[i].publish(self._odom(t, h.x, h.y, h.yaw, c * h.vx + s * h.vy,
                                                 -s * h.vx + c * h.vy, 0.0))
        if self.k % 10 == 0:
            ranges, amin, inc, rmin, rmax = self.sim.scan()
            m = F.LaserScan()
            m.header.stamp = F._Stamp(t).to_msg()
            m.header.frame_id = "lidar_link"
            m.angle_min, m.angle_increment = amin, inc
            m.angle_max = amin + inc * (len(ranges) - 1)
            m.range_min, m.range_max = rmin, rmax
            m.ranges = [float(v) for v in ranges]
            self.pub_scan.publish(m)


def _run_batch(ros, tmp_path, controllers, perception, scenario="crossing_and_head_on", max_t=120.0):
    from eval_core.scenario import list_presets
    from eval_core.specs import load_robot_spec, load_world_spec
    from eval_core.storage import ResultStore
    lp_mod = importlib.import_module("eval_core.ros.local_planner_node")
    gb_mod = importlib.import_module("eval_core.ros.gazebo_backend")
    sr_mod = importlib.import_module("eval_core.ros.scan_relay_node")
    robot, world = load_robot_spec(), load_world_spec()
    gz = FakeGazebo(robot, world)
    planner = lp_mod.LocalPlannerNode()
    relay = sr_mod.ScanRelay()
    relay.mode = "gpu"
    backend = gb_mod.GazeboBackend(robot, world, ResultStore(tmp_path))
    for _ in range(100):                     # let /clock, odometry and heartbeats flow
        gz.step(0.01)
        F.BUS.spin_once()
    av = backend.available()
    assert av["ok"], av
    presets = {p["name"]: p for p in list_presets()}
    backend.start_batch({"name": "fake", "scenarios": [presets[scenario]],
                         "controllers": [{"name": c} for c in controllers], "perception": perception})
    steps = 0
    while backend.manager.busy and steps < int(max_t * len(controllers) / 0.01):
        gz.step(0.01)
        F.BUS.spin_once()
        steps += 1
    assert not backend.manager.busy, backend.manager.status()
    return gz, planner, backend


def test_ros_glue_full_trial(ros, tmp_path):
    gz, planner, backend = _run_batch(ros, tmp_path, ["mppi", "dwpp"], "scan_tracking")
    runs = {r["summary"]["controller"]: r["summary"] for r in backend.manager.finished_runs}
    assert set(runs) == {"mppi", "dwpp"}
    assert runs["mppi"]["outcome"] == "goal", runs["mppi"].get("message")
    assert runs["mppi"]["min_clearance"] > 0.1
    assert runs["dwpp"]["outcome"] in ("goal", "collision")
    # the scene was reset through the set_pose service
    assert {"eval_robot", "human_1", "human_2", "start_line", "goal_line"} <= set(gz.set_pose_calls)
    assert gz.lines == {"start_line": 0.0, "goal_line": 10.0}
    # the gated command reached Gazebo and the robot really drove the course
    assert F.BUS.published["/cmd_vel"] > 50
    assert F.BUS.published["/controller/cmd_vel"] > 50
    assert F.BUS.published["/eval/path"] >= 2
    # debug output for the live view
    assert backend.controller_debug().get("trajectory")
    snap = backend.snapshot()
    assert snap["robot"] is not None and len(snap["humans"]) == 2


def test_ros_glue_ground_truth_humans_move_correctly(ros, tmp_path):
    """Humans are commanded in the body frame: check they follow their routes."""
    gz, planner, backend = _run_batch(ros, tmp_path, ["mpc"], "ground_truth", "crossing_both_sides", max_t=60)
    run = backend.manager.finished_runs[0]["summary"]
    assert run["outcome"] == "goal", run.get("message")
    # human 1 crossed from the left (+y) to the right side, human 2 from the right to the left
    h1, h2 = gz.sim.humans[0], gz.sim.humans[1]
    assert h1.y < -1.0
    assert h2.y > 1.0
    assert abs(h1.x - 3.5) < 0.3
    assert abs(h2.x - 7.0) < 0.3
