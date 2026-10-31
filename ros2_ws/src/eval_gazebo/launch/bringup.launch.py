"""Everything in one launch (single container / native install):
Gazebo simulation + local planner + evaluation web server.

    ros2 launch eval_gazebo bringup.launch.py [gui:=true] [lidar:=auto] [port:=8080]
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    sim = os.path.join(get_package_share_directory("eval_gazebo"), "launch", "sim.launch.py")
    ev = os.path.join(get_package_share_directory("eval_core"), "launch", "eval.launch.py")
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="false"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("lidar", default_value="auto"),
        DeclareLaunchArgument("port", default_value="8080"),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(sim), launch_arguments={
            "gui": LaunchConfiguration("gui"), "rviz": LaunchConfiguration("rviz"),
            "lidar": LaunchConfiguration("lidar")}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(ev), launch_arguments={
            "port": LaunchConfiguration("port")}.items()),
    ])
