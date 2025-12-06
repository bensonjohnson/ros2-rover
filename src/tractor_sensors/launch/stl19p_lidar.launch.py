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
            default_value='LD19',
            description='LiDAR Model (LD19 for STL-19p)'
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
                {'laser_scan_dir': True},
                {'enable_angle_crop_func': False},
                {'angle_crop_min': 135.0},
                {'angle_crop_max': 225.0}
            ]
        ),
        
        # Static Transform (Base -> Laser)
        # Assuming laser is mounted on top, slightly forward?
        # Adjust these values based on actual mounting!
        # x=0.08 (forward), z=0.15 (height)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_laser_tf',
            arguments=['0.08', '0.0', '0.15', '0.0', '0.0', '0.0', 'base_link', 'laser_link']
        )
    ])
