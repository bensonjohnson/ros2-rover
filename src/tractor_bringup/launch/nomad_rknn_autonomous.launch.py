"""NoMaD autonomous launch.

Runs the pretrained NoMaD visual-navigation policy on the RK3588 NPU. RGB
is fed through `nomad_rknn_runner` (vision_encoder + 10-step diffusion via
noise_pred_net) and the predicted waypoint trajectory is converted to
[left_track, right_track] in the same node. The output topic /track_cmd_ai
is identical to the RLPD runner's, so the downstream lidar_safety_monitor
node gates dangerous commands the same way for both stacks.

joy_node + teleop_twist_joy stay live in autonomous mode so the operator
can take over with the RB deadman — mirrors the RLPD HIL pattern, except
NoMaD has no online learning so the intervention is not logged anywhere.

Launch args:
    goal_mode           exploration | image_goal     (default: exploration)
    goal_image_path     filesystem path to goal RGB  (default: empty)
    nominal_speed       forward speed in m/s         (default: 0.20)
    inference_rate_hz   policy tick rate             (default: 7.0)
    vision_encoder_rknn path to vision_encoder.rknn
    noise_pred_net_rknn path to noise_pred_net.rknn
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    tractor_bringup_dir = get_package_share_directory('tractor_bringup')
    realsense_dir = get_package_share_directory('realsense2_camera')
    tractor_sensors_dir = get_package_share_directory('tractor_sensors')

    # Fallback model location for direct `ros2 launch` use. start_nomad_rover.sh
    # overrides these with paths resolved against the actual checkout.
    default_models_dir = os.path.join(
        os.path.expanduser('~'), 'ros2-rover', 'models', 'nomad'
    )

    goal_mode_arg = DeclareLaunchArgument(
        'goal_mode', default_value='exploration',
        description='exploration | image_goal')
    goal_image_path_arg = DeclareLaunchArgument(
        'goal_image_path', default_value='',
        description='Filesystem path to goal RGB image (used when goal_mode=image_goal)')
    nominal_speed_arg = DeclareLaunchArgument(
        'nominal_speed', default_value='0.20',
        description='Forward linear speed in m/s')
    inference_rate_hz_arg = DeclareLaunchArgument(
        'inference_rate_hz', default_value='7.0',
        description='Policy tick rate in Hz (5-10 recommended)')
    lookahead_dist_arg = DeclareLaunchArgument(
        'lookahead_dist', default_value='0.30',
        description='Pure-pursuit lookahead in meters')
    track_width_arg = DeclareLaunchArgument(
        'track_width', default_value='0.30',
        description='Differential drive track width in meters')
    max_track_speed_arg = DeclareLaunchArgument(
        'max_track_speed', default_value='0.45',
        description='Track speed at which the normalized [-1,1] command equals +/-1')
    vision_encoder_rknn_arg = DeclareLaunchArgument(
        'vision_encoder_rknn',
        default_value=os.path.join(default_models_dir, 'vision_encoder.rknn'),
        description='Path to vision_encoder.rknn')
    noise_pred_net_rknn_arg = DeclareLaunchArgument(
        'noise_pred_net_rknn',
        default_value=os.path.join(default_models_dir, 'noise_pred_net.rknn'),
        description='Path to noise_pred_net.rknn')

    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_bringup_dir, 'launch', 'robot_description.launch.py')))

    hiwonder_motor_node = Node(
        package='tractor_control',
        executable='hiwonder_motor_driver',
        name='hiwonder_motor_driver',
        output='screen',
        parameters=[
            os.path.join(tractor_bringup_dir, 'config', 'hiwonder_motor_params.yaml'),
            {'publish_tf': False},
        ])

    # NoMaD needs RGB at full resolution (the node resizes to 96x96). Depth
    # stays on because robot_localization uses it indirectly via the IMU
    # pipeline and the safety monitor benefits from a populated camera stack.
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(realsense_dir, 'launch', 'rs_launch.py')),
        launch_arguments={
            'pointcloud.enable': 'false',
            'align_depth.enable': 'false',
            'enable_color': 'true',
            'enable_depth': 'true',
            'enable_sync': 'false',
            'device_type': '435i',
            'depth_module.depth_profile': '640x480x30',
            'rgb_camera.color_profile': '640x480x30',
            'enable_gyro': 'false',
            'enable_accel': 'false',
            'enable_imu': 'false',
        }.items())

    lsm9ds1_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_sensors_dir, 'launch', 'lsm9ds1_imu.launch.py')))

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_sensors_dir, 'launch', 'stl19p_lidar.launch.py')),
        launch_arguments={
            'port_name': '/dev/ttyUSB0',
            'frame_id': 'laser_link',
        }.items())

    rf2o_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_sensors_dir, 'launch', 'lidar_odometry.launch.py')),
        launch_arguments={'publish_tf': 'false'}.items())

    # Same safety monitor config as the RLPD launch — output contract is identical.
    safety_monitor_node = Node(
        package='tractor_bringup',
        executable='lidar_safety_monitor.py',
        name='lidar_safety_monitor',
        output='screen',
        parameters=[{
            'scan_topic': '/scan',
            'input_cmd_topic': '/cmd_vel_teleop',
            'output_cmd_topic': '/cmd_vel',
            'input_track_topic': '/track_cmd_ai',
            'output_track_topic': '/track_cmd',
            'stop_distance': 0.15,
            'slow_distance': 0.15,
            'hysteresis': 0.10,
            'min_valid_range': 0.05,
            'max_eval_distance': 5.0,
            'robot_front_offset': 0.06,
            'robot_half_width': 0.12,
            'stale_timeout': 0.2,
            'min_block_duration': 0.3,
        }])

    robot_localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_bringup_dir, 'launch', 'robot_localization.launch.py')),
        launch_arguments={'use_gps': 'false'}.items())

    # Keep joy + teleop live so RB can suppress autonomy.
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        parameters=[os.path.join(tractor_bringup_dir, 'config', 'xbox_teleop.yaml')])

    teleop_twist_node = Node(
        package='teleop_twist_joy',
        executable='teleop_node',
        name='teleop_twist_joy',
        output='screen',
        parameters=[os.path.join(tractor_bringup_dir, 'config', 'xbox_teleop.yaml')],
        remappings=[('cmd_vel', 'cmd_vel_teleop')])

    nomad_runner_node = Node(
        package='tractor_bringup',
        executable='nomad_rknn_runner',
        name='nomad_rknn_runner',
        output='screen',
        parameters=[{
            'vision_encoder_rknn': LaunchConfiguration('vision_encoder_rknn'),
            'noise_pred_net_rknn': LaunchConfiguration('noise_pred_net_rknn'),
            'goal_mode': LaunchConfiguration('goal_mode'),
            'goal_image_path': LaunchConfiguration('goal_image_path'),
            'inference_rate_hz': ParameterValue(
                LaunchConfiguration('inference_rate_hz'), value_type=float),
            'nominal_speed': ParameterValue(
                LaunchConfiguration('nominal_speed'), value_type=float),
            'lookahead_dist': ParameterValue(
                LaunchConfiguration('lookahead_dist'), value_type=float),
            'track_width': ParameterValue(
                LaunchConfiguration('track_width'), value_type=float),
            'max_track_speed': ParameterValue(
                LaunchConfiguration('max_track_speed'), value_type=float),
        }])

    return LaunchDescription([
        goal_mode_arg,
        goal_image_path_arg,
        nominal_speed_arg,
        inference_rate_hz_arg,
        lookahead_dist_arg,
        track_width_arg,
        max_track_speed_arg,
        vision_encoder_rknn_arg,
        noise_pred_net_rknn_arg,

        robot_description_launch,
        hiwonder_motor_node,

        joy_node,
        teleop_twist_node,

        TimerAction(period=2.0, actions=[lidar_launch]),
        TimerAction(period=4.0, actions=[rf2o_launch]),
        TimerAction(period=5.0, actions=[realsense_launch]),
        TimerAction(period=6.0, actions=[lsm9ds1_launch]),

        TimerAction(period=7.0, actions=[robot_localization_launch]),
        TimerAction(period=8.0, actions=[safety_monitor_node]),

        TimerAction(period=10.0, actions=[nomad_runner_node]),
    ])
