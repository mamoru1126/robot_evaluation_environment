"""Evaluation side: local planner (controllers) + web server (trial orchestration).

    ros2 launch eval_core eval.launch.py [port:=8080] [backends:=gazebo,sim2d]
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="8080", description="web UI port"),
        DeclareLaunchArgument("backends", default_value="gazebo,sim2d",
                              description="gazebo and/or sim2d (built-in 2D simulator)"),
        DeclareLaunchArgument("control_rate", default_value="10.0", description="controller rate [Hz]"),
        Node(package="eval_core", executable="local_planner", name="local_planner", output="screen",
             parameters=[{"use_sim_time": True,
                          "control_rate": LaunchConfiguration("control_rate")}]),
        Node(package="eval_core", executable="eval_server", name="eval_server", output="screen",
             arguments=["--port", LaunchConfiguration("port"), "--backends", LaunchConfiguration("backends")]),
    ])
