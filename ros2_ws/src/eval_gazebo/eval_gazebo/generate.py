"""Write the generated world SDF and bridge YAML to a directory.

    ros2 run eval_gazebo generate_world --out /tmp/eval_sim [--no-lidar]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

from eval_core.scenario import DEFAULT_SCENARIO, normalize_scenario, robot_start_pose
from eval_core.specs import load_robot_spec, load_world_spec

from .world_gen import bridge_yaml, world_sdf


def generate(out_dir: str, lidar: bool = True) -> Tuple[str, str]:
    robot = load_robot_spec()
    world = load_world_spec()
    sc = normalize_scenario(DEFAULT_SCENARIO, world.max_humans)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    world_path = out / f"{world.name}.sdf"
    bridge_path = out / "bridge.yaml"
    world_path.write_text(world_sdf(robot, world, lidar, robot_start_pose(sc),
                                    (sc["course"]["start_x"], sc["course"]["goal_x"])), encoding="utf-8")
    bridge_path.write_text(bridge_yaml(world, lidar), encoding="utf-8")
    return str(world_path), str(bridge_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("EVAL_SIM_DIR", "/tmp/eval_sim"))
    ap.add_argument("--no-lidar", action="store_true", help="do not add the gpu_lidar sensor")
    args, _ = ap.parse_known_args()
    w, b = generate(args.out, not args.no_lidar)
    print(w)
    print(b)


if __name__ == "__main__":
    main()
