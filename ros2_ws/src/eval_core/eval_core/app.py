"""Entry point of the evaluation server (web UI + trial orchestration).

    ros2 run eval_core eval_server                 # Gazebo backend (+ 2D simulator)
    python3 -m eval_core.app --backends sim2d      # without ROS (2D simulator only)
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

from .backends.sim2d_backend import Sim2DBackend
from .specs import load_robot_spec, load_world_spec
from .storage import ResultStore
from .web.server import EvalService, serve


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--ros-args" in argv:  # strip ROS arguments when started with `ros2 run`
        argv = argv[:argv.index("--ros-args")]
    ap = argparse.ArgumentParser(description="Robot evaluation web server")
    ap.add_argument("--host", default=os.environ.get("EVAL_WEB_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("EVAL_WEB_PORT", "8080")))
    ap.add_argument("--backends", default=os.environ.get("EVAL_BACKENDS", "gazebo,sim2d"),
                    help="comma separated: gazebo,sim2d")
    ap.add_argument("--default-backend", default=os.environ.get("EVAL_DEFAULT_BACKEND", ""))
    args = ap.parse_args(argv)

    robot = load_robot_spec()
    world = load_world_spec()
    store = ResultStore()
    wanted = [b.strip() for b in args.backends.split(",") if b.strip()]
    backends = {}
    if "gazebo" in wanted:
        try:
            from .ros.gazebo_backend import GazeboBackend
            backends["gazebo"] = GazeboBackend(robot, world, store)
        except Exception:  # noqa: BLE001
            print("[eval_server] Gazebo backend unavailable:\n" + traceback.format_exc(), flush=True)
    if "sim2d" in wanted or not backends:
        backends["sim2d"] = Sim2DBackend(robot, world, store)
    default = args.default_backend if args.default_backend in backends else next(iter(backends))
    service = EvalService(backends, default, store, robot, world)
    print(f"[eval_server] backends={list(backends)} default={default} results={store.root} "
          f"-> http://localhost:{args.port}/", flush=True)
    serve(service, args.host, args.port)


if __name__ == "__main__":
    main()
