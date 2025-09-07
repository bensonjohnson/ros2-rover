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
        self._last_ransac_time = 0.0
        self._last_att_rp = (0.0, 0.0)
        
        # IMU-assisted ground model
        self._imu_up_vec: Optional[np.ndarray] = None  # unit up vector in point frame
        self._sensor_height_m: float = 0.30  # camera height above ground when level
        self._imu_bias_m: float = 0.0  # small height bias correction from occasional RANSAC
        self._imu_ransac_interval_s: float = 4.0
        self._imu_rp_threshold_deg: float = 3.0
        
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
        
    def set_imu_up(self, up_vec_body: np.ndarray):
        """Set unit up vector in the same frame as input points (forward-left-up).
        up_vec_body: shape (3,), will be normalized. If invalid, ignored.
        """
        try:
            v = np.asarray(up_vec_body, dtype=np.float32).reshape(3)
            n = np.linalg.norm(v)
            if n > 1e-6:
                self._imu_up_vec = (v / n).astype(np.float32)
        except Exception:
            pass

    def set_sensor_height(self, h_m: float):
        try:
            self._sensor_height_m = float(h_m)
        except Exception:
            pass

    def set_imu_ground_params(self, ransac_interval_s: float = None, rp_threshold_deg: float = None):
        if ransac_interval_s is not None:
            try:
                self._imu_ransac_interval_s = float(ransac_interval_s)
            except Exception:
                pass
        if rp_threshold_deg is not None:
            try:
                self._imu_rp_threshold_deg = float(rp_threshold_deg)
            except Exception:
                pass

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
        
        # Prefer IMU-based ground filtering for most frames
        if self._imu_up_vec is not None:
            return self._imu_ground_filter(points)

        # Fallback to RANSAC-only method if IMU not available
        
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

    # --- IMU-assisted ground filtering ---
    def _imu_ground_filter(self, points: np.ndarray) -> np.ndarray:
        """Fast IMU-assisted ground removal with local percentile fallback.
        - Computes z_up = dot(n_up, p) + sensor_height + bias.
        - Removes points close to ground (<= grass tolerance), keeps obstacles above min height.
        - Applies per-cell 10th percentile adjustment to handle uneven/soft ground (grass).
        - Runs occasional light RANSAC to correct bias and (optionally) fuse normal.
        """
        if len(points) == 0:
            return points
        n_up = self._imu_up_vec
        if n_up is None:
            return points

        # Signed height above ground using IMU up
        z_up = points @ n_up.astype(np.float32) + (self._sensor_height_m + self._imu_bias_m)

        # Primary fast mask
        grass_tol = float(self.grass_height_tolerance)
        min_obs = float(self.min_obstacle_height)
        keep_mask = z_up >= np.minimum(min_obs, grass_tol)

        # Local percentile fallback in coarse grid
        pts_kept = points[keep_mask]
        if len(pts_kept) == 0:
            pts_kept = points  # fallback to raw

        try:
            # Compute 10th percentile of z_up in coarse cells in (x,y)
            cell = max(self.x_resolution * 4.0, 0.20)  # ~4 pixels or at least 20cm
            x = points[:, 0]
            y = points[:, 1]
            ix = np.floor((x + self.x_range) / cell).astype(np.int32)
            iy = np.floor((y + self.y_range) / cell).astype(np.int32)
            key = (ix.astype(np.int64) << 32) ^ (iy.astype(np.int64) & 0xffffffff)
            # Group indices by cell
            order = np.argsort(key)
            key_sorted = key[order]
            z_sorted = z_up[order]
            x_sorted = x[order]
            y_sorted = y[order]
            unique_keys, starts = np.unique(key_sorted, return_index=True)
            ends = np.r_[starts[1:], len(key_sorted)]
            local_keep = np.zeros_like(z_sorted, dtype=bool)
            for s, e in zip(starts, ends):
                if e - s < 6:
                    continue
                z_cell = z_sorted[s:e]
                # 10th percentile as local ground
                p10 = np.percentile(z_cell, 10.0)
                # Keep if sufficiently above local ground
                local_keep[s:e] = (z_cell - p10) >= np.minimum(min_obs, grass_tol)
            # Map back to original order
            back = np.empty_like(local_keep)
            back[order] = local_keep
            keep_mask = keep_mask | back
        except Exception:
            pass

        filtered = points[keep_mask]
        if len(filtered) == 0:
            filtered = points  # ensure non-empty fallback

        # Occasional light RANSAC to refine bias/normal
        self._maybe_update_bias_with_ransac(points, z_up, n_up)
        return filtered

    def _maybe_update_bias_with_ransac(self, points: np.ndarray, z_up: np.ndarray, n_up: np.ndarray):
        now = time.time()
        do_time = (now - self._last_ransac_time) >= float(self._imu_ransac_interval_s)
        # Estimate roll/pitch from n_up relative to canonical up [0,0,1]
        try:
            # roll/pitch angles approximation from up vector
            ux, uy, uz = float(n_up[0]), float(n_up[1]), float(n_up[2])
            pitch = np.degrees(np.arctan2(-ux, max(uz, 1e-6)))
            roll = np.degrees(np.arctan2(uy, max(uz, 1e-6)))
        except Exception:
            roll = pitch = 0.0
        dr = abs(roll - self._last_att_rp[0])
        dp = abs(pitch - self._last_att_rp[1])
        do_att = (dr > self._imu_rp_threshold_deg) or (dp > self._imu_rp_threshold_deg)
        if not (do_time or do_att):
            return
        self._last_ransac_time = now
        self._last_att_rp = (roll, pitch)
        try:
            # Sample subset for speed
            N = len(points)
            if N < 100:
                return
            idx = np.random.choice(N, size=min(2000, N), replace=False)
            P = points[idx]
            best_inliers = []
            best_plane = None
            iters = int(min(max(30, self.ground_ransac_iterations // 3), 35))
            thr = float(max(self.ground_ransac_threshold, 0.04))
            for _ in range(iters):
                ids = np.random.choice(len(P), 3, replace=False)
                a, b, c = P[ids]
                n = np.cross(b - a, c - a)
                nrm = np.linalg.norm(n)
                if nrm < 1e-6:
                    continue
                n = n / nrm
                # Ensure normal points roughly up (same hemisphere as n_up)
                if np.dot(n, n_up) < 0:
                    n = -n
                d = -np.dot(n, a)
                dist = np.abs(P @ n + d)
                inl = np.where(dist < thr)[0]
                if len(inl) > len(best_inliers):
                    best_inliers = inl
                    best_plane = (n, d)
            if best_plane is None or len(best_inliers) < 50:
                return
            n_hat, d_hat = best_plane
            # Update bias so that plane height along n_up is at zero for ground
            # For points p on plane: n_hat·p + d_hat = 0. We want z_up = n_up·p + (h + bias) ≈ 0.
            # Approximate bias by projecting plane offset along n_up using centroid of inliers.
            Q = P[best_inliers]
            if len(Q) > 0:
                p0 = np.mean(Q, axis=0)
                # Height above ground of p0 by plane model is 0; by IMU model is n_up·p0 + (h + bias)
                est_bias = -(p0 @ n_up) - self._sensor_height_m
                # Conservative EMA update of bias
                alpha = 0.2
                self._imu_bias_m = (1 - alpha) * self._imu_bias_m + alpha * est_bias
                # Optionally blend normal a bit toward plane normal to reduce tilt drift
                blend = 0.1
                n_blend = (1 - blend) * n_up + blend * n_hat
                n_blend = n_blend / max(np.linalg.norm(n_blend), 1e-6)
                self._imu_up_vec = n_blend.astype(np.float32)
        except Exception:
            return

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
