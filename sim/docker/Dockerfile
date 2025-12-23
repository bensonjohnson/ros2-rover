# ROS 2 Jazzy with Webots support for simulation
# Webots GUI runs on host (Fedora/Wayland), controller runs in this container
# Communication via extern controller mechanism over TCP

FROM ros:jazzy-ros-base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    ros-jazzy-tf2-ros \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-xacro \
    wget \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install Webots R2023b (for controller library only, not GUI)
# This provides the Python controller API needed for extern controller
ENV WEBOTS_HOME=/usr/local/webots
RUN wget -qO- https://github.com/cyberbotics/webots/releases/download/R2023b/webots-R2023b-x86-64.tar.bz2 | tar xjf - -C /usr/local \
    && ln -s /usr/local/webots-R2023b /usr/local/webots

# IMPORTANT: Remove the ROS 2 controller package that conflicts with Webots
# ROS 2 Jazzy has a 'controller' Python package that shadows Webots controller
RUN rm -rf /opt/ros/jazzy/lib/python3.12/site-packages/controller* || true

# Set up Webots Python path (BEFORE ROS 2 paths)
ENV PYTHONPATH="${WEBOTS_HOME}/lib/controller/python:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="${WEBOTS_HOME}/lib/controller:${LD_LIBRARY_PATH}"

# Install Python dependencies
RUN pip3 install --break-system-packages --ignore-installed \
    opencv-python-headless \
    nats-py \
    tqdm

# Create workspace  
WORKDIR /ros2_ws

# Copy package files first (for layer caching)
COPY sim/package.xml /ros2_ws/src/webots_sim/package.xml
COPY sim/CMakeLists.txt /ros2_ws/src/webots_sim/CMakeLists.txt

# Install ROS dependencies
RUN . /opt/ros/jazzy/setup.sh && \
    apt-get update && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y || true && \
    rm -rf /var/lib/apt/lists/*

# Copy full source
COPY sim/ /ros2_ws/src/webots_sim/
COPY src/tractor_bringup/ /ros2_ws/src/tractor_bringup/

# Build workspace
RUN . /opt/ros/jazzy/setup.sh && \
    colcon build --packages-select webots_sim tractor_bringup --cmake-args -DCMAKE_BUILD_TYPE=Release || true

# Setup entrypoint
COPY sim/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
