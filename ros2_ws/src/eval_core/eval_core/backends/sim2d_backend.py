"""Built-in kinematic 2D simulator backend (no ROS / Gazebo needed).

Useful for quickly debugging a controller and for automated tests.  It runs
the very same TrialRunner, LocalPlannerCore and controllers as the Gazebo
backend; only the physics and the LiDAR are replaced by :mod:`eval_core.sim2d`.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from ..humans import AgentState
from ..planner_core import LocalPlannerCore
from ..runner import BatchManager
from ..sim2d import Sim2D
from ..specs import RobotSpec, WorldSpec
from ..storage import ResultStore


class Sim2DBackend:
    name = "sim2d"
    label = "簡易2Dシミュレータ (高速・ROS不要)"

    PHYS_DT = 0.01
    TICK_DT = 0.05
    CONTROL_DT = 0.1

    def __init__(self, robot: RobotSpec, world: WorldSpec, store: ResultStore, on_event=None):
        self.robot_spec = robot
        self.world = world
        self.sim = Sim2D(robot, world, num_humans=world.max_humans)
        self.core = LocalPlannerCore(robot, world)
        self.manager = BatchManager(self, robot, world, store, on_event)
        self.thread: Optional[threading.Thread] = None
        self.speed = 0.0                # real-time factor, 0 = as fast as possible
        self.ctrl_state: Dict[str, Any] = {"state": "idle", "trial_id": None}
        self.debug: Dict[str, Any] = {}
        self.course: Dict[str, Any] = {}
        self._stop_flag = False

    # ------------------------------------------------------------------
    # TrialIO
    def robot(self) -> Optional[Tuple[float, float, float, float, float]]:
        r = self.sim.robot
        return r.x, r.y, r.yaw, r.v, r.w

    def humans(self) -> List[Optional[AgentState]]:
        return [AgentState(h.x, h.y, h.vx, h.vy, h.yaw) for h in self.sim.humans]

    def teleport_robot(self, x: float, y: float, yaw: float) -> None:
        self.sim.teleport_robot(x, y, yaw)
        self.core.update_odom(self.sim.t, x, y, yaw, 0.0, 0.0)

    def teleport_human(self, i: int, x: float, y: float, yaw: float) -> None:
        if i < len(self.sim.humans):
            self.sim.teleport_human(i, x, y, yaw)

    def set_course(self, start_x: float, goal_x: float, y: float, path) -> None:
        self.course = {"start_x": start_x, "goal_x": goal_x, "y": y, "path": path}

    def stop_robot(self) -> None:
        self.sim.set_cmd(0.0, 0.0)

    def set_human_cmd(self, i: int, vx: float, vy: float) -> None:
        if i < len(self.sim.humans):
            self.sim.set_human_cmd(i, vx, vy)

    def controller_start(self, trial_id: str, controller: Dict[str, Any], perception: Dict[str, Any],
                         path) -> None:
        ok, msg = self.core.start(trial_id, controller["name"], controller.get("params") or {},
                                  perception.get("mode", "scan"), path, perception.get("use_prediction", True),
                                  perception.get("scan_inflation", 0.15))
        self.ctrl_state = {"state": "running" if ok else "error", "trial_id": trial_id,
                           "error": "" if ok else msg}
        self.debug = {}

    def controller_stop(self) -> None:
        self.core.stop()
        if self.ctrl_state.get("state") == "running":
            self.ctrl_state = dict(self.ctrl_state, state="idle")

    def controller_state(self) -> Dict[str, Any]:
        return self.ctrl_state

    def controller_cmd(self) -> Optional[Tuple[float, float]]:
        return self.core.last_cmd

    def controller_debug(self) -> Dict[str, Any]:
        return self.debug

    # ------------------------------------------------------------------
    def available(self) -> Dict[str, Any]:
        return {"ok": True, "message": "ready"}

    def start_batch(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        speed = float(spec.get("sim_speed", 0.0) or 0.0)
        batch = self.manager.start(spec)
        self.speed = max(speed, 0.0)
        if self.thread is None or not self.thread.is_alive():
            self._stop_flag = False
            self.thread = threading.Thread(target=self._loop, name="sim2d", daemon=True)
            self.thread.start()
        return batch

    def abort(self) -> None:
        self.manager.abort()

    def shutdown(self) -> None:
        self._stop_flag = True
        self.manager.abort()

    def _loop(self) -> None:
        sim = self.sim
        next_tick = sim.t
        next_ctrl = sim.t
        next_scan = sim.t
        wall0 = time.monotonic()
        sim0 = sim.t
        lidar_dt = 1.0 / max(self.robot_spec.lidar.rate, 1.0)
        while not self._stop_flag:
            if self.manager.runner is None:
                break
            sim.step(self.PHYS_DT)
            r = sim.robot
            self.core.update_odom(sim.t, r.x, r.y, r.yaw, r.v, r.w)
            for i, h in enumerate(sim.humans):
                rad = 0.25
                runner = self.manager.runner
                if runner is not None and i < len(runner.scenario["humans"]):
                    rad = runner.scenario["humans"][i]["radius"]
                self.core.update_human(f"human_{i + 1}", sim.t, h.x, h.y, h.vx, h.vy, rad)
            if sim.t + 1e-9 >= next_scan:
                ranges, amin, inc, rmin, rmax = sim.scan()
                self.core.update_scan(sim.t, ranges, amin, inc, rmin, rmax)
                next_scan += lidar_dt
            if sim.t + 1e-9 >= next_ctrl:
                next_ctrl += self.CONTROL_DT
                runner = self.manager.runner
                out = self.core.step(sim.t, self.CONTROL_DT)
                if out is not None:
                    self.debug = {"trial_id": self.core.trial_id, "trajectory": out.trajectory,
                                  "candidates": out.candidates, **{k: v for k, v in out.info.items()
                                                                    if isinstance(v, (int, float, bool, str))}}
                    if runner is not None and runner.gate_open:
                        sim.set_cmd(out.v, out.w)
                elif self.core.error and self.ctrl_state.get("state") == "running":
                    self.ctrl_state = dict(self.ctrl_state, state="error", error=self.core.error)
            if sim.t + 1e-9 >= next_tick:
                next_tick += self.TICK_DT
                self.manager.tick(sim.t)
            if self.speed > 0:
                lag = (sim.t - sim0) / self.speed - (time.monotonic() - wall0)
                if lag > 0.002:
                    time.sleep(lag)
            elif int(sim.t / self.PHYS_DT) % 200 == 0:
                time.sleep(0)  # let other threads (web server) run
        # batch finished: the simulator is no longer stepped, so leave everything at rest
        sim.set_cmd(0.0, 0.0)
        sim.robot.v = sim.robot.w = 0.0
        for i, h in enumerate(sim.humans):
            sim.set_human_cmd(i, 0.0, 0.0)
            h.vx = h.vy = 0.0

    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        sim = self.sim
        r = sim.robot
        return {
            "backend": self.name, "connected": True, "time": sim.t,
            "robot": {"x": r.x, "y": r.y, "yaw": r.yaw, "v": r.v, "w": r.w, "radius": self.robot_spec.radius},
            "humans": [{"x": h.x, "y": h.y, "vx": h.vx, "vy": h.vy, "yaw": h.yaw} for h in sim.humans],
            "cmd": list(self.core.last_cmd),
            "scan": self.core.scan_points[::4].round(3).tolist() if len(self.core.scan_points) else [],
        }
