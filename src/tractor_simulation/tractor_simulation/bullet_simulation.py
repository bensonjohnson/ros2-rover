#!/usr/bin/env python3
"""
PyBullet Simulation for ROS2 Tractor with ES Training
Provides realistic physics simulation and visualization for training
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import cv2
from typing import Tuple, Dict, Any
import os

class TractorSimulation:
    def __init__(self, use_gui: bool = True, enable_visualization: bool = True):
        """
        Initialize the tractor simulation environment
        
        Args:
            use_gui: Whether to show PyBullet GUI
            enable_visualization: Whether to enable real-time visualization
        """
        self.use_gui = use_gui
        self.enable_visualization = enable_visualization
        
        # Physics parameters
        self.time_step = 1/240.0  # 240 Hz physics
        self.gravity = -9.81
        
        # Robot parameters (matching Hiwonder robot)
        self.wheel_base = 0.3  # meters between wheels
        self.wheel_radius = 0.05  # meters
        self.max_linear_velocity = 0.5  # m/s
        self.max_angular_velocity = 1.0  # rad/s
        
        # Sensor parameters (RealSense D435i)
        self.camera_fov = 69.0  # degrees
        self.camera_width = 424
        self.camera_height = 240
        self.camera_near = 0.1
        self.camera_far = 10.0
        
        # Environment state
        self.robot_id = None
        self.ground_id = None
        self.obstacles = []
        self.step_count = 0
        self.position = np.array([0.0, 0.0, 0.1])
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion
        self.linear_velocity = np.array([0.0, 0.0, 0.0])
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        
        # Initialize simulation
        self._setup_simulation()
        
    def _setup_simulation(self):
        """Setup PyBullet simulation environment"""
        if self.use_gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setGravity(0, 0, self.gravity)
        p.setRealTimeSimulation(0)
        p.setTimeStep(self.time_step)
        
        # Load ground plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground_id = p.loadURDF("plane.urdf")
        
        # Load robot (simple box for now, can be replaced with URDF)
        self._load_robot()
        
        # Setup camera
        self._setup_camera()
        
        print("Simulation initialized successfully")
        
    def _load_robot(self):
        """Load robot model into simulation"""
        # For now, create a simple box robot
        # In future, this can be replaced with a proper URDF
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=[0.15, 0.1, 0.05]  # x, y, z half dimensions
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.15, 0.1, 0.05],
            rgbaColor=[0.2, 0.6, 0.8, 1.0]  # Blue color
        )
        
        self.robot_id = p.createMultiBody(
            baseMass=5.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.position.tolist(),
            baseOrientation=self.orientation.tolist()
        )
        
        # Add wheels (simplified)
        self._add_wheels()
        
    def _add_wheels(self):
        """Add simplified wheel visualization"""
        # For now, skip wheel attachment to avoid API issues
        # Wheels are visual only and not physically simulated
        pass
        
    def _setup_camera(self):
        """Setup depth camera simulation"""
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.15, 0.0, 0.1],  # Front of robot
            cameraTargetPosition=[1.0, 0.0, 0.1],  # Looking forward
            cameraUpVector=[0, 0, 1]
        )
        
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=self.camera_width / self.camera_height,
            nearVal=self.camera_near,
            farVal=self.camera_far
        )
        
    def add_obstacles(self, obstacle_config: Dict[str, Any]):
        """
        Add obstacles to the environment
        
        Args:
            obstacle_config: Dictionary with obstacle parameters
        """
        # Example: Add walls, boxes, cylinders, etc.
        if obstacle_config.get("type") == "wall":
            self._add_wall(obstacle_config)
        elif obstacle_config.get("type") == "box":
            self._add_box(obstacle_config)
        elif obstacle_config.get("type") == "cylinder":
            self._add_cylinder(obstacle_config)
            
    def _add_wall(self, config: Dict[str, Any]):
        """Add a wall obstacle"""
        position = config.get("position", [2.0, 0.0, 0.5])
        size = config.get("size", [0.1, 3.0, 1.0])
        
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=size
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=size,
            rgbaColor=[0.8, 0.8, 0.8, 1.0]
        )
        
        wall_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        self.obstacles.append(wall_id)
        
    def _add_box(self, config: Dict[str, Any]):
        """Add a box obstacle"""
        position = config.get("position", [1.0, 1.0, 0.25])
        size = config.get("size", [0.25, 0.25, 0.25])
        
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=size
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=size,
            rgbaColor=[0.9, 0.1, 0.1, 1.0]
        )
        
        box_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        self.obstacles.append(box_id)
        
    def _add_cylinder(self, config: Dict[str, Any]):
        """Add a cylinder obstacle"""
        position = config.get("position", [1.5, -1.0, 0.3])
        radius = config.get("radius", 0.2)
        height = config.get("height", 0.6)
        
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=height
        )
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=[0.1, 0.9, 0.1, 1.0]
        )
        
        cylinder_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        self.obstacles.append(cylinder_id)
        
    def set_velocity(self, linear_velocity: float, angular_velocity: float):
        """
        Set robot velocity commands
        
        Args:
            linear_velocity: Forward/backward velocity (m/s)
            angular_velocity: Rotation velocity (rad/s)
        """
        # Clamp velocities to limits
        linear_velocity = np.clip(linear_velocity, -self.max_linear_velocity, self.max_linear_velocity)
        angular_velocity = np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity)
        
        # Convert to wheel velocities for differential drive
        left_velocity = (linear_velocity - angular_velocity * self.wheel_base / 2.0) / self.wheel_radius
        right_velocity = (linear_velocity + angular_velocity * self.wheel_base / 2.0) / self.wheel_radius
        
        # Apply velocities (simplified - in reality would apply to wheel joints)
        # For now, directly set base velocity
        velocity = [
            linear_velocity * np.cos(self._get_yaw()),
            linear_velocity * np.sin(self._get_yaw()),
            0.0
        ]
        
        p.resetBaseVelocity(self.robot_id, linearVelocity=velocity, angularVelocity=[0, 0, angular_velocity])
        
    def _get_yaw(self) -> float:
        """Get robot yaw angle from quaternion"""
        orientation = p.getBasePositionAndOrientation(self.robot_id)[1]
        euler = p.getEulerFromQuaternion(orientation)
        return euler[2]
        
    def get_depth_image(self) -> np.ndarray:
        """
        Get simulated depth image from robot camera
        
        Returns:
            Depth image as numpy array (height, width)
        """
        # Render camera image
        width, height = self.camera_width, self.camera_height
        
        # Update view matrix based on current robot position
        robot_pos, robot_orient = p.getBasePositionAndOrientation(self.robot_id)
        camera_pos = [
            robot_pos[0] + 0.15 * np.cos(self._get_yaw()),
            robot_pos[1] + 0.15 * np.sin(self._get_yaw()),
            robot_pos[2] + 0.1
        ]
        
        target_pos = [
            camera_pos[0] + np.cos(self._get_yaw()),
            camera_pos[1] + np.sin(self._get_yaw()),
            camera_pos[2]
        ]
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=[0, 0, 1]
        )
        
        # Get camera image
        camera_image = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Extract depth buffer
        depth_buffer = np.reshape(camera_image[3], [height, width])
        
        # Convert depth buffer to actual depth values
        # Depth buffer values are in range [0, 1], convert to meters
        depth_image = self.camera_far * self.camera_near / (
            self.camera_far - depth_buffer * (self.camera_far - self.camera_near)
        )
        
        # Clip to sensor range
        depth_image = np.clip(depth_image, self.camera_near, self.camera_far)
        
        return depth_image.astype(np.float32)
        
    def get_robot_state(self) -> Dict[str, Any]:
        """
        Get current robot state
        
        Returns:
            Dictionary with robot state information
        """
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        linear_velocity, angular_velocity = p.getBaseVelocity(self.robot_id)
        
        # Convert quaternion to euler for easier use
        euler_orientation = p.getEulerFromQuaternion(orientation)
        
        return {
            "position": np.array(position),
            "orientation": np.array(euler_orientation),
            "linear_velocity": np.array(linear_velocity),
            "angular_velocity": np.array(angular_velocity),
            "yaw": euler_orientation[2]
        }
        
    def step(self) -> Dict[str, Any]:
        """
        Advance simulation by one step
        
        Returns:
            Dictionary with simulation state
        """
        p.stepSimulation()
        self.step_count += 1
        
        # Get current state
        robot_state = self.get_robot_state()
        depth_image = self.get_depth_image()
        
        # Update internal state tracking
        self.position = robot_state["position"]
        self.orientation = np.array(p.getQuaternionFromEuler(robot_state["orientation"].tolist()))
        self.linear_velocity = robot_state["linear_velocity"]
        self.angular_velocity = robot_state["angular_velocity"]
        
        return {
            "robot_state": robot_state,
            "depth_image": depth_image,
            "step_count": self.step_count
        }
        
    def reset(self):
        """Reset simulation to initial state"""
        p.resetBasePositionAndOrientation(
            self.robot_id,
            self.position.tolist(),
            self.orientation.tolist()
        )
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])
        self.step_count = 0
        
    def close(self):
        """Clean up simulation"""
        p.disconnect()
        
    def add_indoor_environment(self):
        """Add a simple indoor environment with walls"""
        # Add walls to create a room
        walls = [
            {"type": "wall", "position": [0, 3, 1], "size": [5, 0.1, 1]},  # North wall
            {"type": "wall", "position": [0, -3, 1], "size": [5, 0.1, 1]},  # South wall
            {"type": "wall", "position": [5, 0, 1], "size": [0.1, 3, 1]},   # East wall
            {"type": "wall", "position": [-5, 0, 1], "size": [0.1, 3, 1]},  # West wall
        ]
        
        for wall_config in walls:
            self.add_obstacles(wall_config)
            
        # Add some obstacles
        obstacles = [
            {"type": "box", "position": [2, 1, 0.25], "size": [0.25, 0.25, 0.25]},
            {"type": "cylinder", "position": [-1, -1, 0.3], "radius": 0.3, "height": 0.6},
            {"type": "box", "position": [3, -2, 0.5], "size": [0.5, 0.3, 0.5]},
        ]
        
        for obstacle_config in obstacles:
            self.add_obstacles(obstacle_config)
            
    def add_outdoor_environment(self):
        """Add a simple outdoor environment"""
        # Add some random obstacles
        obstacles = [
            {"type": "cylinder", "position": [2, 2, 0.3], "radius": 0.4, "height": 0.6},
            {"type": "box", "position": [-2, 1, 0.25], "size": [0.3, 0.3, 0.5]},
            {"type": "cylinder", "position": [0, -3, 0.4], "radius": 0.5, "height": 0.8},
            {"type": "box", "position": [4, -1, 0.3], "size": [0.4, 0.6, 0.6]},
        ]
        
        for obstacle_config in obstacles:
            self.add_obstacles(obstacle_config)

# Simple test function
def test_simulation():
    """Test the simulation environment"""
    print("Starting simulation test...")
    
    # Create simulation with GUI
    sim = TractorSimulation(use_gui=True, enable_visualization=True)
    
    # Add environment
    sim.add_indoor_environment()
    
    # Test robot movement
    print("Testing robot movement...")
    for i in range(1000):
        # Move forward and turn slightly
        sim.set_velocity(0.2, 0.1)
        
        # Step simulation
        state = sim.step()
        
        # Print status every 100 steps
        if i % 100 == 0:
            robot_state = state["robot_state"]
            print(f"Step {i}: Position: {robot_state['position']}, Yaw: {robot_state['yaw']:.2f}")
            
        # Small delay for visualization
        time.sleep(1/60.0)
        
    # Clean up
    sim.close()
    print("Simulation test completed!")

if __name__ == "__main__":
    test_simulation()
