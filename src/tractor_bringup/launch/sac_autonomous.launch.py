import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get package directories
    tractor_bringup_dir = get_package_share_directory('tractor_bringup')
    realsense_dir = get_package_share_directory('realsense2_camera')
    tractor_sensors_dir = get_package_share_directory('tractor_sensors')
    
    # Launch arguments
    server_addr_arg = DeclareLaunchArgument(
        'server_addr',
        default_value='tcp://10.0.0.200:5556',
        description='Address of V620 training server'
    )
    
    max_speed_arg = DeclareLaunchArgument(
        'max_speed',
        default_value='0.18',
        description='Maximum linear speed (m/s)'
    )

    collision_dist_arg = DeclareLaunchArgument(
        'collision_distance',
        default_value='0.12',
        description='Collision detection distance (m)'
    )
    
    # 1. Robot Description
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_bringup_dir, 'launch', 'robot_description.launch.py')
        )
    )

    # 2. Motor Driver
    hiwonder_motor_node = Node(
        package="tractor_control",
        executable="hiwonder_motor_driver",
        name="hiwonder_motor_driver",
        output="screen",
        parameters=[os.path.join(tractor_bringup_dir, "config", "hiwonder_motor_params.yaml")]
    )

    # 3. RealSense Camera (Depth + RGB, no PointCloud for speed)
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(realsense_dir, "launch", "rs_launch.py")
        ),
        launch_arguments={
            "pointcloud.enable": "false",
            "align_depth.enable": "true",
            "enable_color": "true",
            "enable_depth": "true",
            "enable_sync": "true",
            "device_type": "435i",
            "depth_module.depth_profile": "424x240x30",
            "rgb_camera.color_profile": "424x240x30",
            "enable_gyro": "false",
            "enable_accel": "false",
            "enable_imu": "false",
        }.items()
    )

    # 4. IMU (LSM9DS1)
    lsm9ds1_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_sensors_dir, "launch", "lsm9ds1_imu.launch.py")
        )
    )

    # 5. Velocity Feedback Controller
    vfc_node = Node(
        package="tractor_control",
        executable="velocity_feedback_controller",
        name="velocity_feedback_controller",
        output="screen",
        parameters=[{"control_frequency": 50.0}]
    )

    # 6. Safety Monitor (Depth-based)
    safety_monitor_node = Node(
        package="tractor_bringup",
        executable="simple_depth_safety_monitor.py",
        name="simple_depth_safety_monitor",
        output="screen",
        parameters=[{
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "input_cmd_topic": "cmd_vel_ai",
            "output_cmd_topic": "cmd_vel_raw",
            "emergency_stop_distance": 0.25,
            "hard_stop_distance": LaunchConfiguration("collision_distance"),
            "depth_scale": 0.001,
            "forward_roi_width_ratio": 0.6,
            "forward_roi_height_ratio": 0.5,
            "max_eval_distance": 5.0,
        }],
    )
    
    # 7. SAC Episode Runner Node
    sac_runner_node = Node(
        package='tractor_bringup',
        executable='sac_episode_runner',
        name='sac_episode_runner',
        output='screen',
        parameters=[{
            'server_addr': LaunchConfiguration('server_addr'),
            'max_linear_speed': LaunchConfiguration('max_speed'),
            'max_angular_speed': 1.0,
            'inference_rate_hz': 30.0,
            'batch_size': 256
        }]
    )
    
    # Build Launch Description with Timers for orderly startup
    return LaunchDescription([
        server_addr_arg,
        max_speed_arg,
        collision_dist_arg,
        
        # Core
        robot_description_launch,
        hiwonder_motor_node,
        
        # Sensors (delayed)
        TimerAction(period=5.0, actions=[realsense_launch]),
        TimerAction(period=6.0, actions=[lsm9ds1_launch]),
        
        # Control (delayed)
        TimerAction(period=8.0, actions=[vfc_node]),
        TimerAction(period=9.0, actions=[safety_monitor_node]),
        
        # AI (last)
        TimerAction(period=10.0, actions=[sac_runner_node])
    ])
