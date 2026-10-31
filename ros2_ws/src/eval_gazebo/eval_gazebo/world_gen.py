"""Generate the Gazebo (Harmonic, SDFormat) world and the ros_gz_bridge config.

The robot, the corridor and the number of humans come from
``config/robot.yaml`` and ``config/world.yaml`` (see eval_core.specs), so the
simulation and the controllers always share the same dimensions and limits.

Entities created (names are used by the evaluation server):

* ``eval_robot``  cylindrical differential-drive serving robot
                  DiffDrive (cmd: /cmd_vel), OdometryPublisher (ground truth),
                  gpu_lidar (gz topic /scan, frame lidar_link)
* ``human_N``     moving person (N = 1..max_humans), VelocityControl
                  (/model/human_N/cmd_vel) + OdometryPublisher (/model/human_N/odometry)
* ``start_line`` / ``goal_line``  floor markers, moved per scenario with set_pose
* ``walls``, ``ground_plane``
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from eval_core.specs import RobotSpec, WorldSpec

ROBOT = "eval_robot"
HUMAN_COLORS = [(0.90, 0.33, 0.05), (0.19, 0.51, 0.74), (0.19, 0.64, 0.33), (0.46, 0.42, 0.69),
                (0.80, 0.30, 0.55), (0.55, 0.34, 0.29)]


def _f(v: float) -> str:
    return f"{v:.6g}"


def _material(r: float, g: float, b: float, a: float = 1.0) -> str:
    return (f"<material><ambient>{_f(r)} {_f(g)} {_f(b)} {_f(a)}</ambient>"
            f"<diffuse>{_f(r)} {_f(g)} {_f(b)} {_f(a)}</diffuse>"
            f"<specular>0.1 0.1 0.1 1</specular></material>")


def _cyl_inertia(m: float, r: float, h: float) -> Tuple[float, float, float]:
    ixx = m * (3 * r * r + h * h) / 12.0
    return ixx, ixx, m * r * r / 2.0


def _inertial(m: float, ixx: float, iyy: float, izz: float, z: float = 0.0) -> str:
    return (f"<inertial><pose>0 0 {_f(z)} 0 0 0</pose><mass>{_f(m)}</mass><inertia>"
            f"<ixx>{_f(ixx)}</ixx><ixy>0</ixy><ixz>0</ixz><iyy>{_f(iyy)}</iyy><iyz>0</iyz><izz>{_f(izz)}</izz>"
            f"</inertia></inertial>")


def _cyl_visual(name: str, r: float, z0: float, z1: float, color, x: float = 0.0, y: float = 0.0) -> str:
    return (f"<visual name='{name}'><pose>{_f(x)} {_f(y)} {_f((z0 + z1) / 2)} 0 0 0</pose>"
            f"<geometry><cylinder><radius>{_f(r)}</radius><length>{_f(z1 - z0)}</length></cylinder></geometry>"
            f"{_material(*color)}</visual>")


# ----------------------------------------------------------------------------
def robot_sdf(robot: RobotSpec, pose: Tuple[float, float, float], lidar: bool) -> str:
    R = robot.radius
    H = robot.height
    wr = robot.wheel_radius
    sep = robot.wheel_separation
    caster_r = 0.04
    body_z0 = 0.04
    ixx, iyy, izz = _cyl_inertia(robot.mass, R, H - body_z0)
    L = robot.lidar
    lz = L.z
    # visuals: a serving robot (dark base, lidar slot, white body with trays, blue head ring)
    vis = [
        _cyl_visual("base", R, body_z0, max(lz - 0.06, body_z0 + 0.05), (0.18, 0.2, 0.24)),
        _cyl_visual("lidar_window", 0.06, lz - 0.03, lz + 0.03, (0.05, 0.05, 0.05)),
        _cyl_visual("body", R * 0.96, lz + 0.06, H - 0.12, (0.93, 0.94, 0.96)),
        _cyl_visual("tray_low", R + 0.02, 0.62, 0.635, (0.80, 0.65, 0.45)),
        _cyl_visual("tray_high", R + 0.02, 0.88, 0.895, (0.80, 0.65, 0.45)),
        _cyl_visual("head", R * 0.9, H - 0.12, H, (0.12, 0.44, 0.92)),
        # small marker that shows the front (+x)
        f"<visual name='front_marker'><pose>{_f(R * 0.9)} 0 {_f(H - 0.06)} 0 0 0</pose>"
        f"<geometry><box><size>0.06 0.12 0.06</size></box></geometry>{_material(1.0, 1.0, 1.0)}</visual>",
    ]
    sensor = ""
    if lidar:
        sensor = f"""
        <sensor name='lidar' type='gpu_lidar'>
          <pose>{_f(L.x)} 0 {_f(lz)} 0 0 0</pose>
          <topic>scan</topic>
          <gz_frame_id>lidar_link</gz_frame_id>
          <update_rate>{_f(L.rate)}</update_rate>
          <always_on>1</always_on>
          <visualize>false</visualize>
          <lidar>
            <scan>
              <horizontal>
                <samples>{int(L.samples)}</samples>
                <resolution>1</resolution>
                <min_angle>{_f(L.min_angle)}</min_angle>
                <max_angle>{_f(L.max_angle)}</max_angle>
              </horizontal>
              <vertical>
                <samples>1</samples>
                <resolution>1</resolution>
                <min_angle>0</min_angle>
                <max_angle>0</max_angle>
              </vertical>
            </scan>
            <range>
              <min>{_f(L.range_min)}</min>
              <max>{_f(L.range_max)}</max>
              <resolution>0.01</resolution>
            </range>
            <noise>
              <type>gaussian</type>
              <mean>0.0</mean>
              <stddev>{_f(L.noise_std)}</stddev>
            </noise>
          </lidar>
        </sensor>"""
    wm = 1.0
    wixx, wiyy, wizz = _cyl_inertia(wm, wr, 0.04)
    wheels = []
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        wheels.append(f"""
      <link name='{side}_wheel'>
        <pose>0 {_f(sgn * sep / 2)} {_f(wr)} -1.5707963 0 0</pose>
        {_inertial(wm, wixx, wiyy, wizz)}
        <collision name='collision'>
          <geometry><cylinder><radius>{_f(wr)}</radius><length>0.04</length></cylinder></geometry>
          <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface>
        </collision>
        <visual name='visual'>
          <geometry><cylinder><radius>{_f(wr)}</radius><length>0.04</length></cylinder></geometry>
          {_material(0.08, 0.08, 0.08)}
        </visual>
      </link>
      <joint name='{side}_wheel_joint' type='revolute'>
        <parent>base_link</parent>
        <child>{side}_wheel</child>
        <axis>
          <xyz>0 0 1</xyz>
          <limit><lower>-1.79769e+308</lower><upper>1.79769e+308</upper></limit>
        </axis>
      </joint>""")
    casters = []
    cx = min(R - caster_r - 0.04, 0.17)
    for name, x in (("front", cx), ("rear", -cx)):
        casters.append(f"""
      <link name='caster_{name}'>
        <pose>{_f(x)} 0 {_f(caster_r)} 0 0 0</pose>
        {_inertial(0.3, 2e-4, 2e-4, 2e-4)}
        <collision name='collision'>
          <geometry><sphere><radius>{_f(caster_r)}</radius></sphere></geometry>
          <surface><friction><ode><mu>0.0</mu><mu2>0.0</mu2></ode></friction></surface>
        </collision>
        <visual name='visual'>
          <geometry><sphere><radius>{_f(caster_r)}</radius></sphere></geometry>
          {_material(0.3, 0.3, 0.3)}
        </visual>
      </link>
      <joint name='caster_{name}_joint' type='fixed'>
        <parent>base_link</parent>
        <child>caster_{name}</child>
      </joint>""")
    x, y, yaw = pose
    return f"""
    <model name='{ROBOT}'>
      <pose>{_f(x)} {_f(y)} 0.005 0 0 {_f(yaw)}</pose>
      <link name='base_link'>
        {_inertial(robot.mass, ixx, iyy, izz, z=0.35)}
        <collision name='body_collision'>
          <pose>0 0 {_f((body_z0 + H) / 2)} 0 0 0</pose>
          <geometry><cylinder><radius>{_f(R)}</radius><length>{_f(H - body_z0)}</length></cylinder></geometry>
          <surface><friction><ode><mu>0.2</mu><mu2>0.2</mu2></ode></friction></surface>
        </collision>
        {''.join(vis)}{sensor}
      </link>{''.join(wheels)}{''.join(casters)}
      <plugin filename='gz-sim-diff-drive-system' name='gz::sim::systems::DiffDrive'>
        <left_joint>left_wheel_joint</left_joint>
        <right_joint>right_wheel_joint</right_joint>
        <wheel_separation>{_f(sep)}</wheel_separation>
        <wheel_radius>{_f(wr)}</wheel_radius>
        <topic>/cmd_vel</topic>
        <odom_topic>/model/{ROBOT}/odometry</odom_topic>
        <tf_topic>/model/{ROBOT}/wheel_tf</tf_topic>
        <frame_id>odom</frame_id>
        <child_frame_id>base_link</child_frame_id>
        <odom_publish_frequency>50</odom_publish_frequency>
        <max_linear_velocity>{_f(robot.max_v)}</max_linear_velocity>
        <min_linear_velocity>{_f(-robot.max_v)}</min_linear_velocity>
        <max_angular_velocity>{_f(robot.max_w)}</max_angular_velocity>
        <min_angular_velocity>{_f(-robot.max_w)}</min_angular_velocity>
        <max_linear_acceleration>{_f(robot.max_acc)}</max_linear_acceleration>
        <min_linear_acceleration>{_f(-robot.max_dec)}</min_linear_acceleration>
        <max_angular_acceleration>{_f(robot.max_alpha)}</max_angular_acceleration>
        <min_angular_acceleration>{_f(-robot.max_alpha)}</min_angular_acceleration>
      </plugin>
      <plugin filename='gz-sim-odometry-publisher-system' name='gz::sim::systems::OdometryPublisher'>
        <odom_frame>world</odom_frame>
        <robot_base_frame>base_link</robot_base_frame>
        <odom_topic>/model/{ROBOT}/ground_truth</odom_topic>
        <tf_topic>/model/{ROBOT}/ground_truth_tf</tf_topic>
        <odom_publish_frequency>50</odom_publish_frequency>
        <dimensions>2</dimensions>
      </plugin>
    </model>"""


def human_sdf(i: int, pose: Tuple[float, float, float], radius: float = 0.25) -> str:
    """A person: two legs (seen by the 0.3 m LiDAR), torso, arms, head.

    Moved kinematically by VelocityControl (body-frame twist). The collision
    cylinder is frictionless so that ground contact does not resist the motion.
    """
    name = f"human_{i + 1}"
    col = HUMAN_COLORS[i % len(HUMAN_COLORS)]
    skin = (0.94, 0.80, 0.69)
    pants = (0.2, 0.22, 0.28)
    ixx, iyy, izz = _cyl_inertia(70.0, radius, 1.75)
    x, y, yaw = pose
    vis = [
        _cyl_visual("leg_l", 0.07, 0.0, 0.85, pants, y=0.1),
        _cyl_visual("leg_r", 0.07, 0.0, 0.85, pants, y=-0.1),
        f"<visual name='torso'><pose>0 0 1.15 0 0 0</pose><geometry><box><size>0.24 0.42 0.6</size></box></geometry>"
        f"{_material(*col)}</visual>",
        _cyl_visual("arm_l", 0.05, 0.85, 1.42, col, y=0.26),
        _cyl_visual("arm_r", 0.05, 0.85, 1.42, col, y=-0.26),
        f"<visual name='head'><pose>0 0 1.6 0 0 0</pose><geometry><sphere><radius>0.11</radius></sphere></geometry>"
        f"{_material(*skin)}</visual>",
        f"<visual name='nose'><pose>0.1 0 1.6 0 0 0</pose><geometry><sphere><radius>0.03</radius></sphere></geometry>"
        f"{_material(*skin)}</visual>",
    ]
    return f"""
    <model name='{name}'>
      <pose>{_f(x)} {_f(y)} 0 0 0 {_f(yaw)}</pose>
      <link name='body'>
        {_inertial(70.0, ixx, iyy, izz, z=0.9)}
        <collision name='collision'>
          <pose>0 0 0.875 0 0 0</pose>
          <geometry><cylinder><radius>{_f(radius)}</radius><length>1.75</length></cylinder></geometry>
          <surface><friction><ode><mu>0.0</mu><mu2>0.0</mu2></ode></friction></surface>
        </collision>
        {''.join(vis)}
      </link>
      <plugin filename='gz-sim-velocity-control-system' name='gz::sim::systems::VelocityControl'>
        <topic>/model/{name}/cmd_vel</topic>
      </plugin>
      <plugin filename='gz-sim-odometry-publisher-system' name='gz::sim::systems::OdometryPublisher'>
        <odom_frame>world</odom_frame>
        <robot_base_frame>{name}</robot_base_frame>
        <odom_topic>/model/{name}/odometry</odom_topic>
        <odom_publish_frequency>30</odom_publish_frequency>
        <dimensions>2</dimensions>
      </plugin>
    </model>"""


def line_sdf(name: str, x: float, width: float, color) -> str:
    return f"""
    <model name='{name}'>
      <static>true</static>
      <pose>{_f(x)} 0 0 0 0 0</pose>
      <link name='link'>
        <visual name='visual'>
          <pose>0 0 0.003 0 0 0</pose>
          <geometry><box><size>0.10 {_f(width)} 0.004</size></box></geometry>
          {_material(*color)}
        </visual>
      </link>
    </model>"""


def walls_sdf(world: WorldSpec) -> str:
    if not world.walls:
        return ""
    length = world.x_max - world.x_min
    cx = (world.x_max + world.x_min) / 2
    t = world.wall_thickness
    h = world.wall_height
    parts = []
    for name, y in (("left", world.corridor_width / 2 + t / 2), ("right", -world.corridor_width / 2 - t / 2)):
        geo = f"<geometry><box><size>{_f(length)} {_f(t)} {_f(h)}</size></box></geometry>"
        parts.append(f"<collision name='{name}_collision'><pose>{_f(cx)} {_f(y)} {_f(h / 2)} 0 0 0</pose>{geo}"
                     f"</collision><visual name='{name}_visual'><pose>{_f(cx)} {_f(y)} {_f(h / 2)} 0 0 0</pose>{geo}"
                     f"{_material(0.72, 0.74, 0.78)}</visual>")
    return f"""
    <model name='walls'>
      <static>true</static>
      <link name='link'>{''.join(parts)}</link>
    </model>"""


def floor_grid_sdf(world: WorldSpec) -> str:
    """Thin 1 m grid lines inside the corridor (visual only)."""
    w = world.corridor_width if world.walls else 8.0
    vis = []
    for k, x in enumerate(range(int(math.ceil(world.x_min)), int(math.floor(world.x_max)) + 1)):
        vis.append(f"<visual name='gx{k}'><pose>{x} 0 0.001 0 0 0</pose><geometry><box><size>0.015 {_f(w)} 0.001</size>"
                   f"</box></geometry>{_material(0.62, 0.64, 0.68)}</visual>")
    return f"""
    <model name='floor_grid'>
      <static>true</static>
      <link name='link'>{''.join(vis)}</link>
    </model>"""


def world_sdf(robot: RobotSpec, world: WorldSpec, lidar: bool = True,
              robot_pose: Tuple[float, float, float] = (-1.0, 0.0, 0.0),
              course: Tuple[float, float] = (0.0, 10.0)) -> str:
    n = int(world.max_humans)
    humans = "".join(human_sdf(i, (world.x_min - 2.0 - 1.5 * i, 0.0, 0.0)) for i in range(n))
    line_w = world.corridor_width if world.walls else 8.0
    sensors = ("<plugin filename='gz-sim-sensors-system' name='gz::sim::systems::Sensors'>"
               "<render_engine>ogre2</render_engine></plugin>") if lidar else ""
    return f"""<?xml version='1.0'?>
<sdf version='1.9'>
  <world name='{world.name}'>
    <physics name='1ms' type='ignored'>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename='gz-sim-physics-system' name='gz::sim::systems::Physics'/>
    <plugin filename='gz-sim-user-commands-system' name='gz::sim::systems::UserCommands'/>
    <plugin filename='gz-sim-scene-broadcaster-system' name='gz::sim::systems::SceneBroadcaster'/>
    {sensors}
    <gravity>0 0 -9.8</gravity>
    <scene>
      <ambient>0.6 0.6 0.6 1</ambient>
      <background>0.85 0.88 0.92 1</background>
      <shadows>false</shadows>
      <grid>false</grid>
    </scene>
    <light type='directional' name='sun'>
      <cast_shadows>false</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.4 0.2 -0.9</direction>
    </light>
    <model name='ground_plane'>
      <static>true</static>
      <link name='link'>
        <collision name='collision'>
          <geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
          <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface>
        </collision>
        <visual name='visual'>
          <geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
          {_material(0.86, 0.86, 0.84)}
        </visual>
      </link>
    </model>{floor_grid_sdf(world)}{walls_sdf(world)}{line_sdf('start_line', course[0], line_w, (0.10, 0.62, 0.36))}{line_sdf('goal_line', course[1], line_w, (0.82, 0.27, 0.23))}{robot_sdf(robot, robot_pose, lidar)}{humans}
  </world>
</sdf>
"""


# ----------------------------------------------------------------------------
def bridge_config(world: WorldSpec, lidar: bool = True) -> List[Dict[str, str]]:
    """ros_gz_bridge parameter_bridge YAML (list of dicts)."""
    cfg: List[Dict[str, str]] = [
        {"ros_topic_name": "/clock", "gz_topic_name": "/clock", "ros_type_name": "rosgraph_msgs/msg/Clock",
         "gz_type_name": "gz.msgs.Clock", "direction": "GZ_TO_ROS"},
        {"ros_topic_name": "/cmd_vel", "gz_topic_name": "/cmd_vel", "ros_type_name": "geometry_msgs/msg/Twist",
         "gz_type_name": "gz.msgs.Twist", "direction": "ROS_TO_GZ"},
        {"ros_topic_name": "/odom", "gz_topic_name": f"/model/{ROBOT}/odometry",
         "ros_type_name": "nav_msgs/msg/Odometry", "gz_type_name": "gz.msgs.Odometry", "direction": "GZ_TO_ROS"},
        {"ros_topic_name": "/ground_truth/odom", "gz_topic_name": f"/model/{ROBOT}/ground_truth",
         "ros_type_name": "nav_msgs/msg/Odometry", "gz_type_name": "gz.msgs.Odometry", "direction": "GZ_TO_ROS"},
        {"ros_topic_name": "/tf", "gz_topic_name": f"/model/{ROBOT}/ground_truth_tf",
         "ros_type_name": "tf2_msgs/msg/TFMessage", "gz_type_name": "gz.msgs.Pose_V", "direction": "GZ_TO_ROS"},
    ]
    if lidar:
        cfg.append({"ros_topic_name": "/scan_gz", "gz_topic_name": "/scan",
                    "ros_type_name": "sensor_msgs/msg/LaserScan", "gz_type_name": "gz.msgs.LaserScan",
                    "direction": "GZ_TO_ROS"})
    for i in range(int(world.max_humans)):
        name = f"human_{i + 1}"
        cfg.append({"ros_topic_name": f"/humans/{name}/cmd_vel", "gz_topic_name": f"/model/{name}/cmd_vel",
                    "ros_type_name": "geometry_msgs/msg/Twist", "gz_type_name": "gz.msgs.Twist",
                    "direction": "ROS_TO_GZ"})
        cfg.append({"ros_topic_name": f"/humans/{name}/odom", "gz_topic_name": f"/model/{name}/odometry",
                    "ros_type_name": "nav_msgs/msg/Odometry", "gz_type_name": "gz.msgs.Odometry",
                    "direction": "GZ_TO_ROS"})
    cfg.append({"service_name": f"/world/{world.name}/set_pose",
                "ros_type_name": "ros_gz_interfaces/srv/SetEntityPose",
                "gz_req_type_name": "gz.msgs.Pose", "gz_rep_type_name": "gz.msgs.Boolean"})
    return cfg


def bridge_yaml(world: WorldSpec, lidar: bool = True) -> str:
    lines = []
    for entry in bridge_config(world, lidar):
        first = True
        for k, v in entry.items():
            lines.append(("- " if first else "  ") + f'{k}: "{v}"')
            first = False
    return "\n".join(lines) + "\n"
