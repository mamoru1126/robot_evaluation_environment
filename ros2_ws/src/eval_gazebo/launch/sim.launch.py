"""Gazebo Harmonic simulation of the evaluation corridor.

    ros2 launch eval_gazebo sim.launch.py [gui:=true] [lidar:=auto|gpu|virtual] [rviz:=true]

* generates the world SDF / bridge YAML from config/robot.yaml + config/world.yaml
* starts the Gazebo server (headless, EGL rendering for the gpu_lidar)
* ros_gz_bridge (clock, cmd_vel, odometry, scan, humans, set_pose service)
* scan_relay: /scan from the gpu_lidar, or a virtual LiDAR fallback
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _setup(context, *args, **kwargs):
    from eval_core.specs import load_robot_spec
    from eval_gazebo.generate import generate

    gui = _truthy(LaunchConfiguration("gui").perform(context))
    rviz = _truthy(LaunchConfiguration("rviz").perform(context))
    lidar = LaunchConfiguration("lidar").perform(context).strip().lower()
    if lidar not in ("auto", "gpu", "virtual"):
        lidar = "auto"
    headless = _truthy(LaunchConfiguration("headless_rendering").perform(context))
    verbosity = LaunchConfiguration("verbosity").perform(context)
    out_dir = LaunchConfiguration("out_dir").perform(context)

    use_gpu_lidar = lidar in ("auto", "gpu")
    world_path, bridge_path = generate(out_dir, use_gpu_lidar)
    robot = load_robot_spec()
    gz_args = f"-r -s -v {verbosity} "
    if headless and use_gpu_lidar:
        gz_args += "--headless-rendering "
    gz_args += world_path
    gz_launch = os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")

    actions = [
        LogInfo(msg=f"[eval_gazebo] world={world_path} lidar={lidar} gz_args='{gz_args}'"),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(gz_launch),
                                 launch_arguments={"gz_args": gz_args, "on_exit_shutdown": "true"}.items()),
        Node(package="ros_gz_bridge", executable="parameter_bridge", name="gz_bridge", output="screen",
             parameters=[{"config_file": bridge_path}]),
        Node(package="tf2_ros", executable="static_transform_publisher", name="lidar_tf", output="log",
             arguments=["--x", str(robot.lidar.x), "--y", "0", "--z", str(robot.lidar.z),
                        "--frame-id", "base_link", "--child-frame-id", "lidar_link"],
             parameters=[{"use_sim_time": True}]),
        Node(package="eval_core", executable="scan_relay", name="scan_relay", output="screen",
             parameters=[{"mode": lidar, "use_sim_time": True}]),
    ]
    if gui:
        actions.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(gz_launch),
                                                launch_arguments={"gz_args": f"-g -v {verbosity}"}.items()))
    if rviz:
        cfg = os.path.join(get_package_share_directory("eval_gazebo"), "rviz", "eval.rviz")
        actions.append(Node(package="rviz2", executable="rviz2", name="rviz2", arguments=["-d", cfg],
                            parameters=[{"use_sim_time": True}], output="log"))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="false", description="also start the Gazebo GUI client"),
        DeclareLaunchArgument("rviz", default_value="false", description="start RViz2"),
        DeclareLaunchArgument("lidar", default_value="auto",
                              description="auto: gpu_lidar with virtual fallback / gpu / virtual"),
        DeclareLaunchArgument("headless_rendering", default_value="true",
                              description="use EGL headless rendering for the gpu_lidar"),
        DeclareLaunchArgument("verbosity", default_value="2", description="gz verbosity 0-4"),
        DeclareLaunchArgument("out_dir", default_value="/tmp/eval_sim",
                              description="where the generated world / bridge files are written"),
        OpaqueFunction(function=_setup),
    ])
