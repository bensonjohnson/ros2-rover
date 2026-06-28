"""Deep Explorer Network — full autonomous mapping launch.

Brings up the complete system with NO camera dependency (LiDAR-only by default):

  1. LiDAR driver (STL19P)
  2. IMU (LSM9DS1 or BNO085)
  3. HiWonder motor driver
  4. slam_toolbox LiDAR SLAM (publishes /map from LiDAR alone)    ← NEW: no camera needed
  5. Lidar safety monitor (downstream gate)
  6. Deep Explorer Network runner (NN on NPU or CPU)
  7. Explore Manager (frontier-based high-level coordination)
  8. Map Integrator (occupancy crop publisher)
  9. Data Collector (experience logging for remote training)

Optional extras (with --depth):
  - RealSense D435i RGB-D camera
  - RTAB-Map RGB-D SLAM (replaces slam_toolbox)
  - Depth image stream to the NN

Sensor fusion for the NN with LiDAR-only:
  - /scan (72 bins) → lidar encoder
  - /map (64×64 crop) → occupancy grid encoder — from slam_toolbox
  - /imu/data (yaw rate) + /joint_states (wheel velocities) → proprio encoder
  - Place novelty + safety hold → interoception
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_bringup = get_package_share_directory("tractor_bringup")
    pkg_sensors = get_package_share_directory("tractor_sensors")
    pkg_explorer = get_package_share_directory("tractor_explorer")

    # ---- Arguments ----
    args = [
        DeclareLaunchArgument("control_rate_hz", default_value="15.0"),
        DeclareLaunchArgument("action_scale", default_value="0.6"),
        DeclareLaunchArgument("exploration_noise", default_value="0.05"),
        DeclareLaunchArgument("lidar_port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("use_depth", default_value="false",
                              description="Enable RealSense D435i + RTAB-Map RGB-D SLAM"),
        DeclareLaunchArgument("mode", default_value="auto",
                              description="auto|explore|collect"),
        DeclareLaunchArgument("imu_type", default_value="bno085",
                              description="bno085 or lsm9ds1"),
        DeclareLaunchArgument("learn", default_value="false",
                              description="Online learning (requires PyTorch)"),
        DeclareLaunchArgument("dashboard_port", default_value="8083"),
        DeclareLaunchArgument("model_path",
                              default_value=os.path.expanduser("~/.ros/explorer_brain.pt")),
        DeclareLaunchArgument("rknn_model_path",
                              default_value=os.path.expanduser("~/.ros/explorer_brain.rknn")),
    ]

    # ---- 1. Robot description (URDF) ----
    robot_desc = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, "launch", "robot_description.launch.py")))

    # ---- 2. Motor driver ----
    motor = Node(
        package="tractor_control",
        executable="hiwonder_motor_driver",
        name="hiwonder_motor_driver",
        output="screen",
        parameters=[os.path.join(pkg_bringup, "config", "hiwonder_motor_params.yaml"),
                    {"publish_tf": False}],
    )

    # ---- 3. LiDAR ----
    lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_sensors, "launch", "stl19p_lidar.launch.py")),
        launch_arguments={"port_name": LaunchConfiguration("lidar_port"),
                          "frame_id": "laser_link"}.items())

    # ---- 4. LiDAR odometry (RF2O — scan matching for odometry) ----
    rf2o = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_sensors, "launch", "lidar_odometry.launch.py")),
        launch_arguments={"publish_tf": "false"}.items())

    # ---- 5. IMU ----
    imu_lsm = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_sensors, "launch", "lsm9ds1_imu.launch.py")),
        condition=IfCondition(PythonExpression(
            ["'", LaunchConfiguration("imu_type"), "' == 'lsm9ds1'"])))
    imu_bno = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_sensors, "launch", "bno085_imu.launch.py")),
        condition=IfCondition(PythonExpression(
            ["'", LaunchConfiguration("imu_type"), "' == 'bno085'"])))

    # ---- 6. EKF (fuses RF2O + IMU + wheels into /odometry/filtered) ----
    ekf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, "launch", "robot_localization.launch.py")),
        launch_arguments={"use_sim_time": "false", "use_gps": "false"}.items())

    # ---- 7. SLAM — TWO OPTIONS ----
    # Option A (default, no camera): slam_toolbox — pure LiDAR SLAM
    #   Publishes /map from LiDAR scans alone. No camera needed.
    slam_toolbox_node = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[os.path.join(pkg_bringup, "config", "slam_toolbox_params.yaml")],
        condition=UnlessCondition(LaunchConfiguration("use_depth")))

    # Option B (--depth): RTAB-Map RGB-D SLAM with RealSense D435i
    rtab_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, "launch", "rtabmap_nav2.launch.py")),
        condition=IfCondition(LaunchConfiguration("use_depth")))

    # ---- 8. Safety monitor (gates /track_cmd_ai → /track_cmd) ----
    safety = Node(
        package="tractor_bringup",
        executable="lidar_safety_monitor.py",
        name="lidar_safety_monitor",
        output="screen",
        parameters=[{
            "scan_topic": "/scan",
            "input_cmd_topic": "/cmd_vel_unused",
            "output_cmd_topic": "/cmd_vel",
            "input_track_topic": "/track_cmd_ai",
            "output_track_topic": "/track_cmd",
            "stop_distance": 0.15,
            "slow_distance": 0.15,
            "robot_front_offset": 0.06,
            "robot_half_width": 0.12,
            "min_block_points": 3,
            "block_scans": 2,
        }])

    # ---- 9. Deep Explorer Network runner (NN inference) ----
    explorer_runner = Node(
        package="tractor_explorer",
        executable="explorer_runner",
        name="explorer_runner",
        output="screen",
        parameters=[{
            "control_rate_hz": LaunchConfiguration("control_rate_hz"),
            "action_scale": LaunchConfiguration("action_scale"),
            "exploration_noise": LaunchConfiguration("exploration_noise"),
            "use_depth": LaunchConfiguration("use_depth"),
            "mode": LaunchConfiguration("mode"),
            "learn": LaunchConfiguration("learn"),
            "model_path": LaunchConfiguration("model_path"),
            "rknn_model_path": LaunchConfiguration("rknn_model_path"),
            "dashboard_port": LaunchConfiguration("dashboard_port"),
        }])

    # ---- 10. Explore Manager (frontier-driven coordination) ----
    explore_manager = Node(
        package="tractor_explorer",
        executable="explore_manager",
        name="explore_manager",
        output="screen",
        parameters=[{
            "update_rate_hz": 5.0,
            "coverage_threshold": 0.95,
            "stuck_timeout_s": 30.0,
        }])

    # ---- 11. Map Integrator (occupancy crop → NN) ----
    map_integrator = Node(
        package="tractor_explorer",
        executable="map_integrator",
        name="map_integrator",
        output="screen",
        parameters=[{
            "crop_size": 64,
            "crop_half_meters": 2.0,
        }])

    # ---- 12. Data Collector (experience logging) ----
    data_collector = Node(
        package="tractor_explorer",
        executable="data_collector",
        name="data_collector",
        output="screen",
        parameters=[{
            "chunk_len": 64,
            "server_addr": "tcp://192.168.1.100:5557",
            "use_zmq": False,
            "save_local": True,
        }])

    # ---- Assembly with staggered startup ----
    return LaunchDescription(args + [
        robot_desc, motor,
        TimerAction(period=2.0, actions=[lidar]),
        TimerAction(period=3.0, actions=[rf2o]),
        TimerAction(period=4.0, actions=[imu_lsm, imu_bno]),
        TimerAction(period=5.0, actions=[ekf]),
        TimerAction(period=5.5, actions=[slam_toolbox_node,
                                          rtab_launch]),
        TimerAction(period=5.5, actions=[safety]),
        TimerAction(period=7.0, actions=[
            explorer_runner, explore_manager, map_integrator, data_collector,
        ]),
    ])
