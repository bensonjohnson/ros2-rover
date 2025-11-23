import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get package directories
    tractor_bringup_dir = get_package_share_directory('tractor_bringup')
    
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
    
    # Include main tractor launch (sensors, motors, safety)
    tractor_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_bringup_dir, 'launch', 'tractor.launch.py')
        ),
        launch_arguments={
            'enable_realsense': 'true',
            'enable_lidar': 'false', # Only using depth camera for now
            'enable_rtabmap': 'false', # No SLAM needed for pure RL
            'enable_safety': 'true'
        }.items()
    )
    
    # PPO Episode Runner Node
    ppo_runner_node = Node(
        package='tractor_bringup',
        executable='ppo_episode_runner',
        name='ppo_episode_runner',
        output='screen',
        parameters=[{
            'server_addr': LaunchConfiguration('server_addr'),
            'max_linear_speed': LaunchConfiguration('max_speed'),
            'max_angular_speed': 1.0,
            'inference_rate_hz': 30.0,
            'batch_size': 256
        }]
    )
    
    return LaunchDescription([
        server_addr_arg,
        max_speed_arg,
        tractor_launch,
        ppo_runner_node
    ])
