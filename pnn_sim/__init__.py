"""Headless 2D simulator for the predictive-coding active-inference brain.

No ROS, no Gazebo: pure Python + numpy + torch. The brain modules are
imported unmodified from tractor_bringup.active_inference, so checkpoints
trained here load on the rover unchanged. See sim/README.md.
"""
