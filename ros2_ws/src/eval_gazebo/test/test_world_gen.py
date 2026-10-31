import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]
for p in (SRC / "eval_gazebo", SRC / "eval_core"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
REPO = SRC.parents[1]
if "EVAL_CONFIG_DIR" not in os.environ and (REPO / "config" / "scenarios").is_dir():
    os.environ["EVAL_CONFIG_DIR"] = str(REPO / "config")

import yaml  # noqa: E402

from eval_core.specs import RobotSpec, WorldSpec  # noqa: E402
from eval_gazebo.generate import generate  # noqa: E402
from eval_gazebo.world_gen import bridge_config, world_sdf  # noqa: E402


def test_world_structure(tmp_path):
    world_path, bridge_path = generate(str(tmp_path), lidar=True)
    root = ET.parse(world_path).getroot()
    w = root.find("world")
    assert w.get("name") == "eval_world"
    models = {m.get("name"): m for m in w.findall("model")}
    for name in ("ground_plane", "walls", "start_line", "goal_line", "eval_robot", "human_1", "human_2"):
        assert name in models, name
    robot = models["eval_robot"]
    plugins = {p.get("name") for p in robot.findall("plugin")}
    assert "gz::sim::systems::DiffDrive" in plugins
    assert "gz::sim::systems::OdometryPublisher" in plugins
    sensor = robot.find("link/sensor")
    assert sensor.get("type") == "gpu_lidar"
    assert sensor.find("topic").text == "scan"
    human_plugins = {p.get("name") for p in models["human_1"].findall("plugin")}
    assert "gz::sim::systems::VelocityControl" in human_plugins
    world_plugins = {p.get("name") for p in w.findall("plugin")}
    assert {"gz::sim::systems::Physics", "gz::sim::systems::UserCommands",
            "gz::sim::systems::SceneBroadcaster", "gz::sim::systems::Sensors"} <= world_plugins
    cfg = yaml.safe_load(Path(bridge_path).read_text())
    assert any(c.get("service_name") == "/world/eval_world/set_pose" for c in cfg)


def test_no_lidar_and_more_humans():
    world = WorldSpec(max_humans=4, walls=False)
    root = ET.fromstring(world_sdf(RobotSpec(), world, lidar=False).encode())
    w = root.find("world")
    names = [m.get("name") for m in w.findall("model")]
    assert [n for n in names if n.startswith("human_")] == ["human_1", "human_2", "human_3", "human_4"]
    assert "walls" not in names
    assert w.find("model[@name='eval_robot']/link/sensor") is None
    assert "gz::sim::systems::Sensors" not in {p.get("name") for p in w.findall("plugin")}
    topics = [c.get("ros_topic_name") for c in bridge_config(world, lidar=False)]
    assert "/scan_gz" not in topics
    assert "/humans/human_4/odom" in topics


def test_robot_dimensions_follow_spec():
    spec = RobotSpec(radius=0.3, height=1.3, wheel_separation=0.4, max_v=1.2)
    root = ET.fromstring(world_sdf(spec, WorldSpec()).encode())
    robot = root.find("world/model[@name='eval_robot']")
    cyl = robot.find("link[@name='base_link']/collision/geometry/cylinder")
    assert float(cyl.find("radius").text) == 0.3
    dd = robot.find("plugin[@name='gz::sim::systems::DiffDrive']")
    assert float(dd.find("wheel_separation").text) == 0.4
    assert float(dd.find("max_linear_velocity").text) == 1.2
