import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Helper to clean up node definitions
    return LaunchDescription([
        DeclareLaunchArgument(
            'port_name',
            default_value='/dev/ttyUSB0',
            description='Serial port name of the lidar'
        ),
        DeclareLaunchArgument(
            'frame_id',
            default_value='laser_link',
            description='Frame ID for the lidar'
        ),
        DeclareLaunchArgument(
            'lidar_model',
            default_value='LDLiDAR_LD19',
            description='LiDAR Model (LDLiDAR_LD19 for STL-19p/LD19)'
        ),

        # LDROBOT LiDAR Publisher
        Node(
            package='ldlidar_stl_ros2',
            executable='ldlidar_stl_ros2_node',
            name='ldlidar_publisher_ld19',
            output='screen',
            parameters=[
                {'product_name': LaunchConfiguration('lidar_model')},
                {'topic_name': 'scan'},
                {'frame_id': LaunchConfiguration('frame_id')},
                {'port_name': LaunchConfiguration('port_name')},
                {'port_baudrate': 230400},
                {'laser_scan_dir': True},  # Counterclockwise - arrow points forward, standard orientation
                {'enable_angle_crop_func': False},
                {'angle_crop_min': 135.0},
                {'angle_crop_max': 225.0}
            ]
        ),

        # NOTE: base_link -> laser_link is published by robot_state_publisher from
        # the URDF (laser_joint, z=0.3655 after the raised mount). Do NOT publish a
        # static transform for the same frame here — two publishers for one transform
        # makes the laser pose jump and corrupts SLAM. The URDF is the single source
        # of truth for sensor extrinsics.
    ])
