import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'publish_tf',
            default_value='false',
            description='Whether rf2o should publish odom->base_link TF (false when using robot_localization)'
        ),

        # RF2O Laser Odometry - uses LD19 LiDAR for odometry
        Node(
            package='rf2o_laser_odometry',
            executable='rf2o_laser_odometry_node',
            name='rf2o_laser_odometry',
            output='screen',
            parameters=[{
                'laser_scan_topic': '/scan',
                'odom_topic': '/odom_rf2o',
                'publish_tf': LaunchConfiguration('publish_tf'),  # Let robot_localization handle TF
                'base_frame_id': 'base_link',
                'odom_frame_id': 'odom',
                'init_pose_from_topic': '',
                'freq': 20.0  # Hz - increased for better odometry
            }]
        )
    ])
