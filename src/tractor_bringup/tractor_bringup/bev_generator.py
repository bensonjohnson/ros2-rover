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
                 ground_ransac_threshold: float = 0.05,
                 # Grass-specific parameters
                 enable_grass_filtering: bool = True,
                 grass_height_tolerance: float = 0.15,  # 15cm grass tolerance
                 min_obstacle_height: float = 0.25,   # Objects must be >25cm to be obstacles
                 # Performance tuning
                 ground_update_interval: int = 10,
                 enable_opencl: bool = False):
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
        
        # Grass-aware filtering parameters
        self.enable_grass_filtering = enable_grass_filtering
        self.grass_height_tolerance = grass_height_tolerance
        self.min_obstacle_height = min_obstacle_height
        
        # Performance/caching
        self.ground_update_interval = max(1, int(ground_update_interval))
        self._frame_counter = 0
        self._cached_ground_plane = None  # (normal, d)
        
        # Optional OpenCL (lazy init)
        self.enable_opencl = bool(enable_opencl)
        self._ocl_ready = False
        self._ocl_ctx = None
        self._ocl_queue = None
        self._ocl_prg = None
        
        # Calculate pixel resolution
        self.x_resolution = (2 * self.x_range) / self.bev_height
        self.y_resolution = (2 * self.y_range) / self.bev_width
        
        # Precompute coordinate mappings for efficiency
        self._precompute_coordinate_mappings()
        
    def _init_opencl(self):
        """Try to initialize OpenCL context and program. Falls back if unavailable."""
        if not self.enable_opencl or self._ocl_ready:
            return
        try:
            import pyopencl as cl
            kernel_src = """
            __kernel void bev_bins(
                __global const float *x,
                __global const float *y,
                __global const float *z,
                const int n,
                const float x_range,
                const float y_range,
                const int H,
                const int W,
                __global int *density,
                __global int *lowcnt,
                __global int *zmax_scaled,
                const float low_thresh,
                const float z_scale
            ){
                int i = get_global_id(0);
                if (i >= n) return;
                float xf = x[i];
                float yf = y[i];
                float zf = z[i];
                if (xf < -x_range || xf > x_range || yf < -y_range || yf > y_range) return;
                int px = (int)((xf + x_range) / (2.0f*x_range) * (float)H);
                int py = (int)((yf + y_range) / (2.0f*y_range) * (float)W);
                if (px < 0) px = 0; if (px >= H) px = H-1;
                if (py < 0) py = 0; if (py >= W) py = W-1;
                int idx = px * W + py;
                atomic_inc((volatile __global int *)&density[idx]);
                if (zf > low_thresh) atomic_inc((volatile __global int *)&lowcnt[idx]);
                int zscaled = (int)(zf * z_scale);
                // emulate atomic max via CAS loop on int
                volatile __global int *addr = &zmax_scaled[idx];
                int old = *addr;
                while (zscaled > old) {
                    int prev = atomic_cmpxchg(addr, old, zscaled);
                    if (prev == old) break;
                    old = prev;
                }
            }
            """
            self._ocl_ctx = cl.create_some_context(interactive=False)
            self._ocl_queue = cl.CommandQueue(self._ocl_ctx)
            self._ocl_prg = cl.Program(self._ocl_ctx, kernel_src).build()
            self._ocl_ready = True
        except Exception:
            self._ocl_ready = False
        
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
            
        # Update cadence: reuse cached plane most frames
        self._frame_counter += 1
        use_cached = self._cached_ground_plane is not None and (self._frame_counter % self.ground_update_interval != 0)
        if use_cached:
            normal, d = self._cached_ground_plane
            distances = np.abs(np.dot(points, normal) + d)
            inliers = np.where(distances < self.ground_ransac_threshold)[0]
            best_inliers = inliers
            best_plane = self._cached_ground_plane
        else:
            # RANSAC-based ground plane removal
            best_inliers = []
            best_plane = None
            for _ in range(self.ground_ransac_iterations):
                if len(points) < 3:
                    break
                indices = np.random.choice(len(points), 3, replace=False)
                sample_points = points[indices]
                try:
                    v1 = sample_points[1] - sample_points[0]
                    v2 = sample_points[2] - sample_points[0]
                    normal = np.cross(v1, v2)
                    nrm = np.linalg.norm(normal)
                    if nrm == 0:
                        continue
                    normal = normal / nrm
                    d = -np.dot(normal, sample_points[0])
                    distances = np.abs(np.dot(points, normal) + d)
                    inliers = np.where(distances < self.ground_ransac_threshold)[0]
                    if len(inliers) > len(best_inliers):
                        best_inliers = inliers
                        best_plane = (normal, d)
                except Exception:
                    continue
            # Cache plane if robust
            if best_plane is not None and len(best_inliers) > 100:
                self._cached_ground_plane = best_plane
        # Apply grass-aware ground filtering
        if self.enable_grass_filtering and len(best_inliers) > 0:
            original_count = len(points)
            filtered_points = self._grass_aware_ground_filtering(points, best_plane, best_inliers)
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
                if self._debug_counter % 100 == 0:  # Print every 100 frames
                    removed_count = original_count - len(filtered_points)
                    print(f"[GrassFilter] Removed {removed_count}/{original_count} points ({removed_count/original_count*100:.1f}%)")
            else:
                self._debug_counter = 1
            return filtered_points
        
        # Standard ground removal (inliers)
        elif len(best_inliers) > 0:
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
        
        # Create BEV channels - vectorized per-pixel aggregation (no Python loops)
        num_channels = 4
        bev_image = np.zeros((self.bev_height, self.bev_width, num_channels), dtype=np.float32)

        H, W = self.bev_height, self.bev_width
        lin_idx = (pixel_x * W + pixel_y).astype(np.int64)
        # Density per pixel
        density = np.bincount(lin_idx, minlength=H * W).astype(np.float32)
        # Low obstacles count per pixel
        low_counts = np.bincount(lin_idx, weights=(z_coords > 0.2).astype(np.float32), minlength=H * W)
        # Max height per pixel
        zmax = np.full(H * W, -1e9, dtype=np.float32)
        np.maximum.at(zmax, lin_idx, z_coords.astype(np.float32))

        # Reshape to images
        density_img = density.reshape(H, W)
        low_img = low_counts.reshape(H, W)
        zmax_img = np.clip(zmax.reshape(H, W), 0.0, None)

        # Channel 0: normalized max height (0..3m typical)
        bev_image[:, :, 0] = np.clip(zmax_img / 3.0, 0.0, 1.0)
        # Channel 1: density normalized by a typical count (20)
        bev_image[:, :, 1] = np.clip(density_img / 20.0, 0.0, 1.0)
        # Channel 2: low obstacle normalized by 10
        bev_image[:, :, 2] = np.clip(low_img / 10.0, 0.0, 1.0)
        # Channel 3: obstacle confidence = height_score * density_score
        height_score = np.clip(zmax_img / 1.0, 0.0, 1.0)
        density_score = np.clip(density_img / 20.0, 0.0, 1.0)
        bev_image[:, :, 3] = height_score * density_score

        # Optional OpenCL acceleration (experimental)
        if self.enable_opencl:
            self._init_opencl()
            if self._ocl_ready:
                try:
                    import pyopencl as cl
                    n = len(x_coords)
                    if n > 0:
                        mf = cl.mem_flags
                        ctx = self._ocl_ctx
                        queue = self._ocl_queue
                        prg = self._ocl_prg
                        x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_coords.astype(np.float32))
                        y_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_coords.astype(np.float32))
                        z_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_coords.astype(np.float32))
                        dens_buf = cl.Buffer(ctx, mf.READ_WRITE, size=H * W * 4)
                        low_buf = cl.Buffer(ctx, mf.READ_WRITE, size=H * W * 4)
                        zmax_buf = cl.Buffer(ctx, mf.READ_WRITE, size=H * W * 4)
                        cl.enqueue_fill_buffer(queue, dens_buf, np.int32(0), 0, H * W * 4)
                        cl.enqueue_fill_buffer(queue, low_buf, np.int32(0), 0, H * W * 4)
                        cl.enqueue_fill_buffer(queue, zmax_buf, np.int32(0), 0, H * W * 4)
                        prg.bev_bins(
                            queue, (int(n),), None,
                            x_buf, y_buf, z_buf, np.int32(n),
                            np.float32(self.x_range), np.float32(self.y_range),
                            np.int32(H), np.int32(W),
                            dens_buf, low_buf, zmax_buf,
                            np.float32(0.2), np.float32(1000.0)
                        )
                        dens_out = np.empty(H * W, dtype=np.int32)
                        low_out = np.empty(H * W, dtype=np.int32)
                        zmax_out = np.empty(H * W, dtype=np.int32)
                        cl.enqueue_copy(queue, dens_out, dens_buf)
                        cl.enqueue_copy(queue, low_out, low_buf)
                        cl.enqueue_copy(queue, zmax_out, zmax_buf)
                        queue.finish()
                        density_img = dens_out.astype(np.float32).reshape(H, W)
                        low_img = low_out.astype(np.float32).reshape(H, W)
                        zmax_img = (zmax_out.astype(np.float32) / 1000.0).reshape(H, W)
                        bev_image[:, :, 0] = np.clip(zmax_img / 3.0, 0.0, 1.0)
                        bev_image[:, :, 1] = np.clip(density_img / 20.0, 0.0, 1.0)
                        bev_image[:, :, 2] = np.clip(low_img / 10.0, 0.0, 1.0)
                        height_score = np.clip(zmax_img / 1.0, 0.0, 1.0)
                        density_score = np.clip(density_img / 20.0, 0.0, 1.0)
                        bev_image[:, :, 3] = height_score * density_score
                except Exception:
                    pass
        
        return bev_image
    
    def _grass_aware_ground_filtering(self, points: np.ndarray, best_plane: tuple, ground_inliers: np.ndarray) -> np.ndarray:
        """
        Advanced ground filtering that handles grass and vegetation
        
        Args:
            points: Nx3 array of 3D points
            best_plane: (normal, d) tuple defining the ground plane
            ground_inliers: indices of points classified as ground by RANSAC
            
        Returns:
            Filtered points with grass-aware ground removal
        """
        normal, d = best_plane
        
        # Calculate distances to the ground plane for all points
        distances_to_plane = np.abs(np.dot(points, normal) + d)
        signed_distances = np.dot(points, normal) + d  # Signed distances (above/below plane)
        
        # Multi-stage filtering approach
        obstacle_mask = np.ones(len(points), dtype=bool)
        
        # Stage 1: Remove obvious ground points (within tight tolerance)
        strict_ground_mask = distances_to_plane < (self.ground_ransac_threshold * 0.5)
        obstacle_mask &= ~strict_ground_mask
        
        # Stage 2: Grass-aware filtering for medium height points  
        grass_candidate_mask = (
            (distances_to_plane >= (self.ground_ransac_threshold * 0.5)) &
            (signed_distances > 0) &  # Above ground plane
            (signed_distances < self.grass_height_tolerance)  # Within grass height
        )
        
        if np.any(grass_candidate_mask):
            # For grass candidates, apply statistical filtering
            grass_candidates = points[grass_candidate_mask]
            
            # Grid-based analysis: divide area into 50cm x 50cm cells
            cell_size = 0.5  # 50cm cells
            
            # Create grid coordinates
            x_coords = grass_candidates[:, 0]
            y_coords = grass_candidates[:, 1]
            z_coords = grass_candidates[:, 2]
            
            # Find grid bounds
            if len(x_coords) > 0:
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                
                # Process each grid cell
                x_cells = int(np.ceil((x_max - x_min) / cell_size)) + 1
                y_cells = int(np.ceil((y_max - y_min) / cell_size)) + 1
                
                grass_removal_mask = np.zeros(len(grass_candidates), dtype=bool)
                
                for i in range(x_cells):
                    for j in range(y_cells):
                        # Define cell boundaries
                        x_start = x_min + i * cell_size
                        x_end = x_start + cell_size
                        y_start = y_min + j * cell_size
                        y_end = y_start + cell_size
                        
                        # Find points in this cell
                        cell_mask = (
                            (x_coords >= x_start) & (x_coords < x_end) &
                            (y_coords >= y_start) & (y_coords < y_end)
                        )
                        
                        if not np.any(cell_mask):
                            continue
                            
                        cell_heights = z_coords[cell_mask]
                        cell_count = len(cell_heights)
                        
                        if cell_count < 3:  # Not enough points for statistics
                            continue
                            
                        # Statistical analysis of heights in this cell
                        height_std = np.std(cell_heights)
                        height_range = np.max(cell_heights) - np.min(cell_heights)
                        
                        # Grass characteristics:
                        # - High density of points (many grass blades)
                        # - Low height variation (uniform grass height)
                        # - Points distributed across small height range
                        
                        is_likely_grass = (
                            (cell_count > 15) and  # Dense point cloud
                            (height_std < 0.05) and  # Low height variation (5cm)
                            (height_range < 0.12)  # Small total range (12cm)
                        )
                        
                        if is_likely_grass:
                            # Mark these points for removal (grass)
                            grass_removal_mask[cell_mask] = True
                
                # Apply grass removal to obstacle mask
                grass_indices = np.where(grass_candidate_mask)[0]
                points_to_remove = grass_indices[grass_removal_mask]
                obstacle_mask[points_to_remove] = False
        
        # Stage 3: Minimum obstacle height filter
        # Only keep points that are significantly above ground
        significant_obstacle_mask = signed_distances >= self.min_obstacle_height
        
        # Final filtering: combine all criteria
        final_mask = obstacle_mask & (significant_obstacle_mask | 
                                    (signed_distances >= self.grass_height_tolerance))
        
        return points[final_mask]
        
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
    
    # Create BEV generator with grass-aware filtering
    bev_gen = BEVGenerator(
        bev_size=(200, 200),
        bev_range=(10.0, 10.0),
        height_channels=(0.2, 1.0),
        enable_ground_removal=True,
        enable_grass_filtering=True,
        grass_height_tolerance=0.15,  # 15cm grass tolerance
        min_obstacle_height=0.25      # 25cm minimum obstacle height
    )
    
    # Generate BEV
    bev_image = bev_gen.generate_bev(points)
    print(f"BEV shape: {bev_image.shape}")
    print(f"BEV range: [{np.min(bev_image):.3f}, {np.max(bev_image):.3f}]")
