"""Backend-independent trial / batch orchestration.

A *batch* is a list of *trials* (controller x scenario x repeat).  For each
trial the :class:`TrialRunner` state machine

    STOP_ROBOT -> TELEPORT -> SETTLE -> START_CONTROLLER -> RUNNING -> FINISHED

is advanced by calling :meth:`BatchManager.tick` periodically (20 Hz) with
the current *simulation* time.  The backend supplies a :class:`TrialIO`
implementation (Gazebo through ROS 2, or the built-in 2D simulator).
"""
from __future__ import annotations

import copy
import math
import threading
import time
import traceback
import uuid
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

from . import metrics as metrics_mod
from .humans import AgentState, HumanAgent
from .scenario import evaluation_path, human_initial_pose, normalize_scenario, robot_start_pose
from .specs import RobotSpec, WorldSpec
from .storage import ResultStore


class TrialIO(Protocol):  # pragma: no cover - interface
    name: str

    def robot(self) -> Optional[Tuple[float, float, float, float, float]]: ...          # x, y, yaw, v, w
    def humans(self) -> List[Optional[AgentState]]: ...
    def teleport_robot(self, x: float, y: float, yaw: float) -> None: ...
    def teleport_human(self, i: int, x: float, y: float, yaw: float) -> None: ...
    def set_course(self, start_x: float, goal_x: float, y: float, path: List[List[float]]) -> None: ...
    def stop_robot(self) -> None: ...
    def set_human_cmd(self, i: int, vx: float, vy: float) -> None: ...
    def controller_start(self, trial_id: str, controller: Dict[str, Any], perception: Dict[str, Any],
                         path: List[List[float]]) -> None: ...
    def controller_stop(self) -> None: ...
    def controller_state(self) -> Dict[str, Any]: ...
    def controller_cmd(self) -> Optional[Tuple[float, float]]: ...
    def controller_debug(self) -> Dict[str, Any]: ...


# ----------------------------------------------------------------------------
class TrialRunner:
    STOP_ROBOT = "stopping"
    TELEPORT = "resetting"
    SETTLE = "settling"
    START_CONTROLLER = "starting_controller"
    RUNNING = "running"
    FINISHED = "finished"

    def __init__(self, io: TrialIO, trial: Dict[str, Any], robot: RobotSpec, world: WorldSpec,
                 record_period: float = 0.05):
        self.io = io
        self.trial = trial
        self.trial_id = trial["id"]
        self.scenario = trial["scenario"]
        self.controller = trial["controller"]
        self.robot = robot
        self.world = world
        self.record_period = record_period
        self.state = self.STOP_ROBOT
        self.state_since: Optional[float] = None
        self.t0: Optional[float] = None
        self.outcome: Optional[str] = None
        self.message = ""
        self.path = evaluation_path(self.scenario)
        course = self.scenario["course"]
        self.agents = [HumanAgent(h, course, robot.radius) for h in self.scenario["humans"]]
        self.rows: List[List[float]] = []
        self.last_record_t = -1e9
        self.last_tick_t: Optional[float] = None
        self.min_clearance = math.inf
        self.collisions = 0
        self._in_collision = False
        self.result: Optional[Dict[str, Any]] = None
        self.debug: Dict[str, Any] = {}
        self.compute_ms: List[float] = []
        self.abort_requested = False
        self.start_retry = 0

    # ------------------------------------------------------------------
    @property
    def gate_open(self) -> bool:
        """True while the controller's commands may reach the robot."""
        return self.state == self.RUNNING

    def elapsed(self, now: float) -> float:
        return 0.0 if self.t0 is None else now - self.t0

    def _goto(self, state: str, now: float) -> None:
        self.state = state
        self.state_since = now
        self._entered = False

    def _enter(self) -> bool:
        """True exactly once per state (entry actions)."""
        if getattr(self, "_entered", False):
            return False
        self._entered = True
        return True

    def abort(self) -> None:
        self.abort_requested = True

    # ------------------------------------------------------------------
    def tick(self, now: float) -> None:
        if self.state_since is None:
            self.state_since = now
        if self.last_tick_t is not None and now < self.last_tick_t - 1.0:
            # simulation time went backwards (e.g. Gazebo restarted)
            self._finish(now, "error", "simulation time jumped backwards (simulator restarted?)")
            return
        self.last_tick_t = now
        if self.abort_requested and self.state != self.FINISHED:
            self._finish(now, "aborted", "aborted by user")
            return
        dt_state = now - self.state_since
        if self.state == self.STOP_ROBOT:
            if self._enter():
                self.io.controller_stop()
            self.io.stop_robot()
            for i in range(len(self.agents)):
                self.io.set_human_cmd(i, 0.0, 0.0)
            rb = self.io.robot()
            slow = rb is not None and abs(rb[3]) < 0.03 and abs(rb[4]) < 0.05
            if (slow and dt_state > 0.3) or dt_state > 3.0:
                self._goto(self.TELEPORT, now)
        elif self.state == self.TELEPORT:
            x, y, yaw = robot_start_pose(self.scenario)
            self.io.teleport_robot(x, y, yaw)
            c = self.scenario["course"]
            for i, (h, ag) in enumerate(zip(self.scenario["humans"], self.agents)):
                if h["enabled"]:
                    hx, hy, hyaw = ag.initial_pose()
                else:  # park disabled humans outside the corridor
                    hx, hy, hyaw = self.world.x_min - 2.0 - 1.5 * i, 0.0, 0.0
                self.io.teleport_human(i, hx, hy, hyaw)
            self.io.set_course(c["start_x"], c["goal_x"], c["y"], self.path)
            self._goto(self.SETTLE, now)
        elif self.state == self.SETTLE:
            self.io.stop_robot()
            if dt_state < 0.6:
                return
            rb = self.io.robot()
            x, y, yaw = robot_start_pose(self.scenario)
            far = rb is None or math.hypot(rb[0] - x, rb[1] - y) > 0.3
            if far and dt_state < 4.0:
                if dt_state > 2.0 and not getattr(self, "_teleport_retried", False):
                    self._teleport_retried = True
                    self.io.teleport_robot(x, y, yaw)
                return
            if far:
                self._finish(now, "error", "no robot odometry" if rb is None else
                             "robot could not be moved to the start pose (set_pose failed?)")
                return
            self.io.controller_start(self.trial_id, self.controller, self.trial["perception_cfg"], self.path)
            self._goto(self.START_CONTROLLER, now)
        elif self.state == self.START_CONTROLLER:
            st = self.io.controller_state()
            if st.get("trial_id") == self.trial_id and st.get("state") == "running":
                self.t0 = now
                self._goto(self.RUNNING, now)
            elif st.get("trial_id") == self.trial_id and st.get("state") == "error":
                self._finish(now, "error", "controller failed to start: " + str(st.get("error", "")))
            elif dt_state > 3.0 and self.start_retry < 2:
                self.start_retry += 1
                self.io.controller_start(self.trial_id, self.controller, self.trial["perception_cfg"], self.path)
                self.state_since = now
            elif dt_state > 3.0:
                self._finish(now, "error", "local planner did not respond (is the eval stack running?)")
        elif self.state == self.RUNNING:
            self._running(now)

    def _running(self, now: float) -> None:
        t = self.elapsed(now)
        rb = self.io.robot()
        hs = self.io.humans()
        dt = 0.05
        robot_tuple = None if rb is None else (rb[0], rb[1], rb[2], rb[3])
        for i, ag in enumerate(self.agents):
            h = hs[i] if i < len(hs) else None
            if h is None or not ag.enabled:
                continue
            vx, vy = ag.update(t, (h.x, h.y), robot_tuple, dt)
            self.io.set_human_cmd(i, vx, vy)

        st = self.io.controller_state()
        if st.get("state") == "error" and st.get("trial_id") == self.trial_id:
            self._record(now, t, rb, hs, force=True)
            self._finish(now, "error", "controller error: " + str(st.get("error", "")))
            return
        dbg = self.io.controller_debug() or {}
        if dbg.get("trial_id") == self.trial_id:
            self.debug = dbg
        if rb is None:
            if t > 5.0:
                self._finish(now, "error", "no robot odometry")
            return
        self._record(now, t, rb, hs)

        # collision / clearance check (ground truth)
        clear = math.inf
        for i, (h, cfg) in enumerate(zip(hs, self.scenario["humans"])):
            if h is None or not cfg["enabled"]:
                continue
            clear = min(clear, math.hypot(h.x - rb[0], h.y - rb[1]) - cfg["radius"] - self.robot.radius)
        self.min_clearance = min(self.min_clearance, clear)
        if clear <= 0.0:
            if not self._in_collision:
                self.collisions += 1
            self._in_collision = True
            if self.scenario["termination"]["stop_on_collision"]:
                self._record(now, t, rb, hs, force=True)
                self._finish(now, "collision", "robot collided with a human")
                return
        else:
            self._in_collision = False

        goal_x = self.scenario["course"]["goal_x"]
        if rb[0] >= goal_x:
            self._record(now, t, rb, hs, force=True)
            self._finish(now, "goal", "reached the GOAL line")
            return
        if t >= self.scenario["termination"]["timeout"]:
            self._finish(now, "timeout", "timeout")

    def _record(self, now: float, t: float, rb, hs, force: bool = False) -> None:
        if not force and t - self.last_record_t < self.record_period - 1e-6:
            return
        self.last_record_t = t
        cmd = self.io.controller_cmd() or (0.0, 0.0)
        cms = float(self.debug.get("compute_ms", float("nan"))) if self.debug else float("nan")
        blocked = 1.0 if self.debug.get("blocked") else 0.0
        row = [t, rb[0], rb[1], rb[2], rb[3], rb[4], cmd[0], cmd[1], cms, blocked]
        for i in range(len(self.scenario["humans"])):
            h = hs[i] if i < len(hs) else None
            if h is None or not self.scenario["humans"][i]["enabled"]:
                row += [float("nan")] * 4
            else:
                row += [h.x, h.y, h.vx, h.vy]
        self.rows.append(row)

    def _finish(self, now: float, outcome: str, message: str) -> None:
        if self.state == self.FINISHED:
            return
        self.outcome = outcome
        self.message = message
        try:
            self.io.controller_stop()
            self.io.stop_robot()
            for i in range(len(self.agents)):
                self.io.set_human_cmd(i, 0.0, 0.0)
        except Exception:  # noqa: BLE001
            pass
        self._goto(self.FINISHED, now)

    def columns(self) -> List[str]:
        cols = ["t", "x", "y", "yaw", "v", "w", "cmd_v", "cmd_w", "compute_ms", "blocked"]
        for i in range(len(self.scenario["humans"])):
            p = f"h{i + 1}_"
            cols += [p + "x", p + "y", p + "vx", p + "vy"]
        return cols


# ----------------------------------------------------------------------------
def build_trials(spec: Dict[str, Any], num_humans: int) -> List[Dict[str, Any]]:
    """Expand a batch request into a list of trials."""
    scenarios = spec.get("scenarios") or [spec.get("scenario") or {}]
    scenarios = [normalize_scenario(s, num_humans) for s in scenarios]
    controllers = spec.get("controllers") or [{"name": "dwa", "params": {}}]
    repeats = max(1, min(int(spec.get("repeats", 1) or 1), 100))
    perception = spec.get("perception", "scan")
    if perception not in ("scan", "scan_tracking", "ground_truth"):
        perception = "scan"
    try:
        inflation = float(spec.get("scan_inflation", 0.15))
    except (TypeError, ValueError):
        inflation = 0.15
    perception_cfg = {"mode": perception, "use_prediction": bool(spec.get("use_prediction", True)),
                      "scan_inflation": min(max(inflation, 0.0), 1.0)}
    trials = []
    n = 0
    for sc in scenarios:
        for ctrl in controllers:
            for r in range(repeats):
                n += 1
                trials.append({
                    "index": n,
                    "id": f"{n:03d}_{ctrl['name']}_{sc['name']}_r{r + 1}",
                    "controller": {"name": ctrl["name"], "params": dict(ctrl.get("params") or {}),
                                   "label": ctrl.get("label", ctrl["name"])},
                    "scenario": sc,
                    "repeat": r + 1,
                    "perception": perception,
                    "perception_cfg": dict(perception_cfg),
                })
    return trials


class BatchManager:
    """Runs a batch of trials one after the other.  Thread-safe."""

    def __init__(self, io: TrialIO, robot: RobotSpec, world: WorldSpec, store: ResultStore,
                 on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        self.io = io
        self.robot = robot
        self.world = world
        self.store = store
        self.on_event = on_event
        self.lock = threading.RLock()
        self.batch: Optional[Dict[str, Any]] = None
        self.trials: List[Dict[str, Any]] = []
        self.index = -1
        self.runner: Optional[TrialRunner] = None
        self.cancelled = False
        self.finished_runs: List[Dict[str, Any]] = []
        self.last_error = ""

    # ------------------------------------------------------------------
    @property
    def busy(self) -> bool:
        with self.lock:
            return self.batch is not None and self.batch.get("status") == "running"

    def start(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            if self.busy:
                raise RuntimeError("a batch is already running")
            trials = build_trials(spec, self.world.max_humans)
            batch = self.store.create_batch(spec.get("name") or "batch", spec, trials, backend=self.io.name)
            self.batch = batch
            self.trials = trials
            self.index = -1
            self.runner = None
            self.cancelled = False
            self.finished_runs = []
            self._next_trial()
            return batch

    def abort(self) -> None:
        with self.lock:
            self.cancelled = True
            if self.runner is not None:
                self.runner.abort()

    def _next_trial(self) -> None:
        self.index += 1
        if self.cancelled or self.index >= len(self.trials):
            self.runner = None
            if self.batch is not None:
                self.batch["status"] = "aborted" if self.cancelled else "finished"
                self.batch["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self.store.update_batch(self.batch)
                self._emit("batch_finished", {"batch_id": self.batch["id"]})
            return
        trial = self.trials[self.index]
        self.runner = TrialRunner(self.io, trial, self.robot, self.world)
        self._emit("trial_started", {"trial_id": trial["id"], "index": self.index})

    def tick(self, now: float) -> None:
        with self.lock:
            r = self.runner
            if r is None:
                return
            try:
                r.tick(now)
            except Exception:  # noqa: BLE001
                self.last_error = traceback.format_exc(limit=4)
                r._finish(now, "error", self.last_error)
            if r.state == TrialRunner.FINISHED:
                self._save(r)
                self._next_trial()

    def _save(self, r: TrialRunner) -> None:
        try:
            summary = metrics_mod.compute(np.asarray(r.rows, float).reshape(-1, len(r.columns())),
                                          r.columns(), r.scenario, self.robot.radius, r.outcome or "error",
                                          r.collisions)
        except Exception:  # noqa: BLE001
            summary = {"outcome": r.outcome, "success": False, "error": traceback.format_exc(limit=4)}
        summary["message"] = r.message
        summary["controller"] = r.controller["name"]
        summary["controller_label"] = r.controller.get("label", r.controller["name"])
        summary["scenario"] = r.scenario["name"]
        summary["repeat"] = r.trial["repeat"]
        summary["perception"] = r.trial["perception"]
        summary["backend"] = self.io.name
        trial_cfg = dict(r.trial, robot_radius=self.robot.radius, robot=self.robot.to_dict(),
                         world=self.world.to_dict(), backend=self.io.name)
        run = self.store.save_run(self.batch["id"], trial_cfg, r.columns(), r.rows, summary)
        self.finished_runs.append(run)
        self.batch.setdefault("completed", []).append(r.trial["id"])
        self.store.update_batch(self.batch)
        self._emit("trial_finished", {"trial_id": r.trial_id, "summary": summary})

    def _emit(self, kind: str, data: Dict[str, Any]) -> None:
        if self.on_event:
            try:
                self.on_event(kind, data)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    def status(self, now: Optional[float] = None) -> Dict[str, Any]:
        with self.lock:
            r = self.runner
            out: Dict[str, Any] = {
                "busy": self.busy,
                "batch": None if self.batch is None else {
                    "id": self.batch["id"], "name": self.batch.get("name"), "status": self.batch.get("status"),
                    "total": len(self.trials), "done": len(self.finished_runs),
                },
                "trial": None,
                "last_runs": [dict(run["summary"], run_id=run["run_id"], batch_id=run["batch_id"])
                              for run in self.finished_runs[-5:]],
            }
            if r is not None:
                out["trial"] = {
                    "id": r.trial_id, "index": r.trial["index"], "state": r.state,
                    "controller": r.controller["name"], "scenario": r.scenario["name"],
                    "repeat": r.trial["repeat"],
                    "elapsed": r.elapsed(now) if (now is not None and r.t0 is not None) else 0.0,
                    "min_clearance": None if not math.isfinite(r.min_clearance) else r.min_clearance,
                    "collisions": r.collisions,
                }
            return out

    def live_scene(self) -> Dict[str, Any]:
        """Data for the live view (scenario geometry + controller debug)."""
        with self.lock:
            r = self.runner
            if r is None:
                return {}
            dbg = r.debug or {}
            return {
                "course": r.scenario["course"],
                "path": r.path,
                "humans_cfg": [{"enabled": h["enabled"], "radius": h["radius"], "pattern": h["pattern"]}
                               for h in r.scenario["humans"]],
                "trajectory": dbg.get("trajectory"),
                "candidates": dbg.get("candidates"),
                "debug_info": {k: v for k, v in dbg.items() if k not in ("trajectory", "candidates")},
            }
