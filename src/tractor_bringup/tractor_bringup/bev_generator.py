#!/usr/bin/env python3
"""
Bird's Eye View (BEV) Generator for Point Cloud Processing
Converts 3D point clouds to 2D multi-channel BEV representations
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import time

class BEVGenerator:
    """
    Generates Bird's Eye View representations from point clouds
    """
    
    def __init__(self, 
                 bev_size: Tuple[int, int] = (200, 200),  # pixels (height, width)
                 bev_range: Tuple[float, float] = (10.0, 10.0),  # meters (x_range, y_range)
                 height_channels: Tuple[float, ...] = (0.2, 1.0),  # height thresholds for channels
                 enable_ground_removal: bool = True,
                 ground_ransac_iterations: int = 100,
                 ground_ransac_threshold: float = 0.05):
        """
        Initialize BEV generator
        
        Args:
            bev_size: (height, width) of BEV image in pixels
            bev_range: (x_range, y_range) in meters from robot center
            height_channels: Height thresholds for multi-channel representation
            enable_ground_removal: Whether to perform ground plane removal
            ground_ransac_iterations: Number of RANSAC iterations for ground plane fitting
            ground_ransac_threshold: Distance threshold for ground plane points
        """
        self.bev_height, self.bev_width = bev_size
        self.x_range, self.y_range = bev_range
        self.height_channels = height_channels
        self.enable_ground_removal = enable_ground_removal
        self.ground_ransac_iterations = ground_ransac_iterations
        self.ground_ransac_threshold = ground_ransac_threshold
        
        # Calculate pixel resolution
        self.x_resolution = (2 * self.x_range) / self.bev_height
        self.y_resolution = (2 * self.y_range) / self.bev_width
        
        # Precompute coordinate mappings for efficiency
        self._precompute_coordinate_mappings()
        
    def _precompute_coordinate_mappings(self):
        """Precompute coordinate mappings for faster BEV generation"""
        # Create coordinate grids
        x_coords = np.linspace(-self.x_range, self.x_range, self.bev_height)
        y_coords = np.linspace(-self.y_range, self.y_range, self.bev_width)
        self.x_grid, self.y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')
        
    def remove_ground_plane(self, points: np.ndarray) -> np.ndarray:
        """
        Remove ground plane points using RANSAC
        
        Args:
            points: Nx3 array of 3D points
            
        Returns:
            Points with ground plane removed
        """
        if not self.enable_ground_removal or len(points) < 3:
            return points
            
        # RANSAC-based ground plane removal
        best_inliers = []
        best_plane = None
        
        for _ in range(self.ground_ransac_iterations):
            # Randomly sample 3 points
            if len(points) < 3:
                break
                
            indices = np.random.choice(len(points), min(3, len(points)), replace=False)
            sample_points = points[indices]
            
            # Fit plane to sample points
            try:
                # Calculate plane normal using cross product
                v1 = sample_points[1] - sample_points[0]
                v2 = sample_points[2] - sample_points[0]
                normal = np.cross(v1, v2)
                
                if np.linalg.norm(normal) == 0:
                    continue
                    
                normal = normal / np.linalg.norm(normal)
                d = -np.dot(normal, sample_points[0])
                
                # Calculate distances to plane
                distances = np.abs(np.dot(points, normal) + d)
                
                # Find inliers
                inliers = np.where(distances < self.ground_ransac_threshold)[0]
                
                # Update best model if this one is better
                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_plane = (normal, d)
                    
            except Exception:
                continue
                
        # Remove ground points (inliers)
        if len(best_inliers) > 0:
            # Keep points that are NOT inliers (not part of ground plane)
            ground_removed = np.delete(points, best_inliers, axis=0)
            return ground_removed if len(ground_removed) > 0 else points
            
        return points
        
    def generate_bev(self, points: np.ndarray) -> np.ndarray:
        """
        Generate multi-channel BEV representation from point cloud
        
        Args:
            points: Nx3 array of 3D points (x, y, z)
            
        Returns:
            Multi-channel BEV image (height, width, channels)
        """
        # Remove ground plane if enabled
        if self.enable_ground_removal:
            points = self.remove_ground_plane(points)
            
        if len(points) == 0:
            # Return empty BEV with all channels
            return np.zeros((self.bev_height, self.bev_width, len(self.height_channels) + 2), dtype=np.float32)
            
        # Convert 3D points to 2D grid coordinates
        x_coords = points[:, 0]  # Forward/backward (x-axis)
        y_coords = points[:, 1]  # Left/right (y-axis)
        z_coords = points[:, 2]  # Up/down (z-axis)
        
        # Filter points within BEV range
        valid_x = (x_coords >= -self.x_range) & (x_coords <= self.x_range)
        valid_y = (y_coords >= -self.y_range) & (y_coords <= self.y_range)
        valid_mask = valid_x & valid_y
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return np.zeros((self.bev_height, self.bev_width, len(self.height_channels) + 2), dtype=np.float32)
            
        x_coords = x_coords[valid_indices]
        y_coords = y_coords[valid_indices]
        z_coords = z_coords[valid_indices]
        
        # Convert to pixel coordinates
        # Note: In BEV, x maps to height dimension, y maps to width dimension
        pixel_x = ((x_coords + self.x_range) / (2 * self.x_range) * self.bev_height).astype(int)
        pixel_y = ((y_coords + self.y_range) / (2 * self.y_range) * self.bev_width).astype(int)
        
        # Clip to valid range
        pixel_x = np.clip(pixel_x, 0, self.bev_height - 1)
        pixel_y = np.clip(pixel_y, 0, self.bev_width - 1)
        
        # Create BEV channels
        num_channels = len(self.height_channels) + 2  # height slices + max height + density
        bev_image = np.zeros((self.bev_height, self.bev_width, num_channels), dtype=np.float32)
        
        # Create coordinate pairs for efficient processing
        coords = np.column_stack((pixel_x, pixel_y))
        unique_coords, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
        
        # Process each unique coordinate
        for i, (px, py) in enumerate(unique_coords):
            # Find all points at this coordinate
            point_indices = np.where(inverse_indices == i)[0]
            z_values = z_coords[point_indices]
            
            # Maximum height channel (channel 0)
            bev_image[px, py, 0] = np.max(z_values)
            
            # Point density channel (channel 1)
            bev_image[px, py, 1] = len(z_values) / 100.0  # Normalize by typical count
            
            # Height slice channels
            for j, height_threshold in enumerate(self.height_channels):
                # Count points above this threshold
                above_threshold = np.sum(z_values > height_threshold)
                bev_image[px, py, j + 2] = above_threshold / 50.0  # Normalize
                
        # Normalize height channel (0-3 meters typical range)
        bev_image[:, :, 0] = np.clip(bev_image[:, :, 0] / 3.0, 0.0, 1.0)
        
        return bev_image
        
    def generate_single_channel_bev(self, points: np.ndarray, channel_type: str = "max_height") -> np.ndarray:
        """
        Generate single-channel BEV for specific visualization
        
        Args:
            points: Nx3 array of 3D points
            channel_type: "max_height", "density", or "height_slice_X" where X is index
            
        Returns:
            Single channel BEV image (height, width)
        """
        bev_multi = self.generate_bev(points)
        
        if channel_type == "max_height":
            return bev_multi[:, :, 0]
        elif channel_type == "density":
            return bev_multi[:, :, 1]
        elif channel_type.startswith("height_slice_"):
            try:
                index = int(channel_type.split("_")[-1])
                if index < len(self.height_channels):
                    return bev_multi[:, :, index + 2]
            except:
                pass
        # Default to max height
        return bev_multi[:, :, 0]

# Example usage and testing
if __name__ == "__main__":
    # Create sample point cloud
    np.random.seed(42)
    points = np.random.randn(1000, 3) * 2  # 1000 random points
    
    # Create BEV generator
    bev_gen = BEVGenerator(
        bev_size=(200, 200),
        bev_range=(10.0, 10.0),
        height_channels=(0.2, 1.0),
        enable_ground_removal=True
    )
    
    # Generate BEV
    bev_image = bev_gen.generate_bev(points)
    print(f"BEV shape: {bev_image.shape}")
    print(f"BEV range: [{np.min(bev_image):.3f}, {np.max(bev_image):.3f}]")
