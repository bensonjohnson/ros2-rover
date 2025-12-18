#!/bin/bash
# Start ES-SAC Rover

source install/setup.bash

export ROS_DOMAIN_ID=0

echo "Starting ES-SAC Rover Runner..."
ros2 run tractor_bringup es_episode_runner.py --ros-args -p nats_server:=nats://nats.gokickrocks.org:4222
