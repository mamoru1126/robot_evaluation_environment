# Robot evaluation environment: ROS 2 Jazzy + Gazebo Harmonic + evaluation web UI
#
#   docker compose build            (or ./scripts/build.sh)
#
ARG ROS_DISTRO=jazzy
FROM ros:${ROS_DISTRO}-ros-base

ARG ROS_DISTRO=jazzy
ENV ROS_DISTRO=${ROS_DISTRO} \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1

# ROS 2 <-> Gazebo Harmonic (ros_gz pulls the Gazebo Harmonic vendor packages),
# RViz2, scientific Python, Mesa (software OpenGL/EGL for the gpu_lidar without GPU)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-ros-gz \
      ros-${ROS_DISTRO}-tf2-ros \
      ros-${ROS_DISTRO}-rviz2 \
      python3-colcon-common-extensions \
      python3-numpy \
      python3-scipy \
      python3-matplotlib \
      python3-yaml \
      python3-pytest \
      python3-pip \
      libegl1 \
      libgl1 \
      libgl1-mesa-dri \
      libglx-mesa0 \
      mesa-utils \
      curl \
 && rm -rf /var/lib/apt/lists/*

# Web server stack (versions tested with this repository).  --ignore-installed
# avoids pip trying to uninstall Debian-packaged dependencies.
RUN pip3 install --no-cache-dir --break-system-packages --ignore-installed \
      "starlette==1.7.0" \
      "uvicorn==0.54.0" \
      "plotly>=5.20"

# Default config / custom controllers (docker-compose mounts the repository versions over these)
COPY config /workspace/config
COPY custom_controllers /workspace/custom_controllers
RUN mkdir -p /workspace/results

# Build the ROS 2 workspace
WORKDIR /ws
COPY ros2_ws/src ./src
RUN . /opt/ros/${ROS_DISTRO}/setup.sh \
 && colcon build --event-handlers console_direct+

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV EVAL_CONFIG_DIR=/workspace/config \
    EVAL_RESULTS_DIR=/workspace/results \
    EVAL_CUSTOM_CONTROLLER_DIR=/workspace/custom_controllers \
    QT_X11_NO_MITSHM=1

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
