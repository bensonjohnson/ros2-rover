import numpy as np
import cv2
from scipy import ndimage

def bresenham_line(x0, y0, x1, y1):
    """
    Fast Bresenham line algorithm for ray tracing.

    Args:
        x0, y0: Start point (robot position)
        x1, y1: End point (obstacle hit)

    Returns:
        points_x, points_y: Arrays of pixel coordinates along the line
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    points_x, points_y = [], []
    while True:
        points_x.append(x0)
        points_y.append(y0)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return np.array(points_x), np.array(points_y)

def clear_rectangular_footprint(grid, robot_row, robot_col, width_m, length_m, resolution):
    """
    Clear rectangular rover footprint from occupancy grid.

    Args:
        grid: Occupancy grid to modify
        robot_row, robot_col: Robot position in grid
        width_m: Rover width in meters
        length_m: Rover length in meters
        resolution: Grid resolution in meters/pixel

    Returns:
        Modified grid with footprint cleared
    """
    width_px = int(width_m / resolution)
    length_px = int(length_m / resolution)

    grid_size = grid.shape[0]
    row_start = max(0, robot_row - length_px)
    row_end = min(grid_size, robot_row + 1)
    col_start = max(0, robot_col - width_px // 2)
    col_end = min(grid_size, robot_col + width_px // 2 + 1)

    grid[row_start:row_end, col_start:col_end] = 0.0
    return grid

class DepthToOccupancy:
    """
    Vectorized processor to convert raw depth images to a top-down occupancy grid.
    """

    def __init__(self,
                 width=424,
                 height=240,
                 fx=386.0,
                 fy=386.0,
                 cx=212.0,
                 cy=120.0,
                 camera_height=0.187,
                 camera_tilt_deg=0.0,
                 grid_size=64,
                 grid_range=3.0,
                 obstacle_height_thresh=0.1,
                 floor_thresh=0.08
                 ):

        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.camera_height = camera_height
        self.camera_tilt = np.radians(camera_tilt_deg)
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.obstacle_thresh = obstacle_height_thresh
        self.floor_thresh = floor_thresh

        u, v = np.meshgrid(np.arange(width), np.arange(height))
        self.x_mult = (u - cx) / fx
        self.y_mult = (v - cy) / fy

    def process(self, depth_image):
        """
        Args:
            depth_image: (H, W) numpy array, float32 (meters) or uint16 (mm)
        Returns:
            grid: (64, 64) numpy array, uint8
        """
        if depth_image.dtype == np.uint16:
            depth = depth_image.astype(np.float32) * 0.001
        else:
            depth = depth_image

        z_c = depth
        x_c = z_c * self.x_mult
        y_c = z_c * self.y_mult

        points_c = np.stack([x_c, y_c, z_c], axis=-1).reshape(-1, 3)

        valid_mask = (points_c[:, 2] > 0.1) & (points_c[:, 2] < 5.0)
        points_c = points_c[valid_mask]

        if len(points_c) == 0:
            return np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        c = np.cos(self.camera_tilt)
        s = np.sin(self.camera_tilt)

        y_c_rot = points_c[:, 1] * c - points_c[:, 2] * s
        z_c_rot = points_c[:, 1] * s + points_c[:, 2] * c
        x_c_rot = points_c[:, 0]

        x_r = z_c_rot
        y_r = -x_c_rot
        z_r = -y_c_rot + self.camera_height

        is_floor = np.abs(z_r) < self.floor_thresh
        is_obstacle = z_r > self.obstacle_thresh

        scale = self.grid_size / self.grid_range
        grid_rows = self.grid_size - 1 - (x_r * scale).astype(np.int32)
        grid_cols = (self.grid_size // 2) - (y_r * scale).astype(np.int32)

        valid_indices = (grid_rows >= 0) & (grid_rows < self.grid_size) & \
                        (grid_cols >= 0) & (grid_cols < self.grid_size)

        grid_rows = grid_rows[valid_indices]
        grid_cols = grid_cols[valid_indices]
        is_floor = is_floor[valid_indices]
        is_obstacle = is_obstacle[valid_indices]

        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        grid[grid_rows[is_floor], grid_cols[is_floor]] = 128
        grid[grid_rows[is_obstacle], grid_cols[is_obstacle]] = 255

        kernel = np.ones((3,3), np.uint8)
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)

        return grid

class ScanToOccupancy:
    """
    Vectorized processor to convert 2D Laser Scan to a top-down occupancy grid.
    """

    def __init__(self,
                 grid_size=64,
                 grid_range=3.0,
                 resolution=0.046875,
                 obstacle_dilation_cm=5,
                 rover_width_m=0.30,
                 rover_length_m=0.40,
                 enable_ray_tracing=True
                 ):
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.resolution = resolution
        self.obstacle_dilation_cm = obstacle_dilation_cm
        self.rover_width_m = rover_width_m
        self.rover_length_m = rover_length_m
        self.enable_ray_tracing = enable_ray_tracing

        self.robot_row = grid_size - 1
        self.robot_col = grid_size // 2

    def process(self, ranges, angle_min, angle_increment):
        """
        Args:
            ranges: List or numpy array of float ranges
            angle_min: float
            angle_increment: float
        Returns:
            grid: (64, 64) numpy array, uint8 (0=unknown, 255=obstacle, 128=free)
        """
        if ranges is None or len(ranges) == 0:
            return np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        ranges = np.array(ranges)

        if not np.any(np.isfinite(ranges)):
            return np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        valid_mask = (ranges > 0.15) & (ranges < self.grid_range) & np.isfinite(ranges)

        if not np.any(valid_mask):
            return np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        angles = angle_min + np.arange(len(ranges)) * angle_increment

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        x = x[valid_mask]
        y = y[valid_mask]

        rows = self.robot_row - (x / self.resolution).astype(np.int32)
        cols = self.robot_col - (y / self.resolution).astype(np.int32)

        valid_indices = (rows >= 0) & (rows < self.grid_size) & \
                        (cols >= 0) & (cols < self.grid_size)

        rows = rows[valid_indices]
        cols = cols[valid_indices]

        if len(rows) == 0:
            return np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        if self.enable_ray_tracing:
            for end_r, end_c in zip(rows, cols):
                cv2.line(grid, (self.robot_col, self.robot_row),
                        (end_c, end_r), 128, 1)

        grid[rows, cols] = 255

        if self.obstacle_dilation_cm > 0:
            kernel_size = max(3, int(self.obstacle_dilation_cm / (self.resolution * 100)))
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            grid = cv2.dilate(grid, kernel, iterations=1)

        grid = clear_rectangular_footprint(grid, self.robot_row, self.robot_col,
                                          self.rover_width_m, self.rover_length_m,
                                          self.resolution).astype(np.uint8)

        return grid

class LocalMapper:
    """
    Maintains a persistent local map by stitching successive frames using odometry.
    
    Features:
    - Shifts/Rotates map based on robot motion.
    - Fades old data to handle drift/dynamic obstacles.
    - Crops center for model input.
    """
    def __init__(self, 
                 map_size=256, 
                 resolution=0.047,
                 decay_rate=0.98):
        
        self.map_size = map_size
        self.resolution = resolution
        self.decay_rate = decay_rate
        
        self.global_map = np.zeros((map_size, map_size), dtype=np.float32)
        self.center = map_size // 2
        
    def update(self, new_observation, dx, dy, dtheta):
        """
        Update the map with motion and new data.
        
        Args:
            new_observation: (64, 64) uint8 grid (Robot at bottom-center)
            dx, dy: Translation in ROBOT frame (meters)
            dtheta: Rotation in radians (counter-clockwise)
        """
        # Convert to pixels
        dx_px = dx / self.resolution
        dy_px = dy / self.resolution
        
        # Apply inverse motion: rotate map opposite to robot rotation, then translate
        M_rot = cv2.getRotationMatrix2D((self.center, self.center), -np.degrees(dtheta), 1.0)
        M_rot[0, 2] += dy_px
        M_rot[1, 2] += dx_px
        
        self.global_map = cv2.warpAffine(
            self.global_map, 
            M_rot, 
            (self.map_size, self.map_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # Fade map to handle drift/dynamic obstacles
        self.global_map *= self.decay_rate
        
        # Stamp new observation (64x64) into center of global map (256x256)
        # Robot at bottom-center (63,32) in observation aligns with center (128,128)
        rows = slice(65, 129)   # 128 - 63 = 65 to 128 + 1 = 129
        cols = slice(96, 160)   # 128 - 32 = 96 to 128 + 32 = 160
        
        obs = new_observation.astype(np.float32)
        mask = (new_observation > 0)
        
        self.global_map[rows, cols][mask] = obs[mask]
        
    def get_model_input(self):
        """
        Return the 64x64 center crop for the model.
        Returns:
            grid: (64, 64) uint8
        """
        # Crop region that positions robot at bottom-center of output (63,32)
        start_row = 128 - 63
        end_row = start_row + 64
        
        start_col = 128 - 32
        end_col = start_col + 64
        
        crop = self.global_map[start_row:end_row, start_col:end_col]
        return crop.astype(np.uint8)


class MultiChannelOccupancy:
    """
    Generates 4-channel 128x128 observation grid for enhanced SAC training:
    - Channel 0: Distance to nearest obstacle [0.0, 1.0] normalized
    - Channel 1: Exploration history [0.0, 1.0] with decay
    - Channel 2: Obstacle confidence [0.0, 1.0] sensor reliability
    - Channel 3: Terrain height [0.0, 1.0] normalized from ground plane

    Improvements over binary 64x64 grid:
    - 4x higher resolution (3.125 cm/pixel vs 4.69 cm/pixel)
    - Continuous distance values preserve gradient information for Q-function
    - Exploration history provides temporal context without LSTM
    - Confidence channel handles sensor uncertainty
    """

    def __init__(self,
                 grid_size=128,
                 range_m=4.0,
                 # Camera parameters (RealSense D435i)
                 width=424,
                 height=240,
                 fx=386.0,
                 fy=386.0,
                 cx=212.0,
                 cy=120.0,
                 camera_height=0.187, # 174mm (bottom) + 12.5mm (to optical center)
                 camera_tilt_deg=2.0,  # Slight downward tilt (adjust if needed)
                 # Thresholds - INCREASED to avoid ground plane false positives
                 # Only consider objects > 15cm above ground as obstacles
                 obstacle_height_thresh=0.15,
                 # Anything within ±12cm of ground level is floor (more tolerance)
                 floor_thresh=0.12,
                 # Max depth range for reliable floor detection (depth sensor degrades beyond this)
                 max_depth_for_floor=2.5):
        
        self.grid_size = grid_size
        self.range_m = range_m
        self.resolution = range_m / grid_size  # 3.125 cm/pixel for 128x128

        # Camera intrinsics
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.camera_height = camera_height
        self.camera_tilt = np.radians(camera_tilt_deg)
        self.obstacle_thresh = obstacle_height_thresh
        self.floor_thresh = floor_thresh
        self.max_depth_for_floor = max_depth_for_floor

        # Pre-compute unprojection matrices
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        self.x_mult = (u - cx) / fx
        self.y_mult = (v - cy) / fy

        # Exploration history tracking (larger persistent map)
        self.exploration_map = np.zeros((512, 512), dtype=np.float32)
        self.decay_rate = 0.998  # Slower decay for larger map
        self.exploration_center = 256  # Robot always at center

        # For odometry-based stitching
        self.last_pose = None  # (x, y, theta)

    def process(self, depth_img, laser_scan, robot_pose=None):
        """
        Process sensor data into 4-channel observation.

        Args:
            depth_img: (H, W) numpy array, uint16 (mm) or float32 (meters)
            laser_scan: LaserScan message or dict with 'ranges', 'angle_min', 'angle_increment'
            robot_pose: Optional (x, y, theta) for odometry stitching

        Returns:
            obs: (4, 128, 128) float32 array, normalized to [0, 1]
        """
        # Channel 0: Continuous distance map
        distance_grid = self._compute_distance_field(depth_img, laser_scan)

        # Channel 1: Exploration history
        if robot_pose is not None:
            self._update_exploration_history(robot_pose)
        history_crop = self._get_centered_crop(self.exploration_map, self.exploration_center, self.grid_size)

        # Channel 2: Obstacle confidence
        confidence_grid = self._compute_confidence(depth_img, laser_scan)

        # Channel 3: Terrain height
        height_grid = self._compute_terrain_height(depth_img)

        # Stack channels and return
        obs = np.stack([distance_grid, history_crop, confidence_grid, height_grid], axis=0)
        return obs.astype(np.float32)

    def _compute_distance_field(self, depth_img, laser_scan):
        """
        Convert depth + LiDAR to continuous distance map.
        Returns grid where each cell contains distance to nearest obstacle.
        Normalized to [0, 1] where 0 = obstacle at 0m, 1 = free space at range_m.
        """
        # Initialize with maximum distance
        grid = np.full((self.grid_size, self.grid_size), self.range_m, dtype=np.float32)

        # Process depth camera
        if depth_img is not None and depth_img.size > 0:
            # Handle uint16 input
            if depth_img.dtype == np.uint16:
                depth = depth_img.astype(np.float32) * 0.001
            else:
                depth = depth_img

            # Unproject to 3D camera frame
            z_c = depth
            x_c = z_c * self.x_mult
            y_c = z_c * self.y_mult

            # Flatten for processing
            points_c = np.stack([x_c, y_c, z_c], axis=-1).reshape(-1, 3)

            # Filter valid depth - split into near and far ranges
            # Near range: high confidence for floor detection
            # Far range: depth sensor less reliable, more conservative obstacle detection
            valid_near = (points_c[:, 2] > 0.1) & (points_c[:, 2] <= self.max_depth_for_floor)
            valid_far = (points_c[:, 2] > self.max_depth_for_floor) & (points_c[:, 2] < self.range_m)

            points_near = points_c[valid_near]
            points_far = points_c[valid_far]

            # Initialize arrays
            x_r_near = np.array([])
            y_r_near = np.array([])
            depths_near = np.array([])
            is_obstacle_near = np.array([], dtype=bool)
            x_r_far = np.array([])
            y_r_far = np.array([])
            depths_far = np.array([])
            is_obstacle_far = np.array([], dtype=bool)

            # Process near-range points (reliable floor detection)
            if len(points_near) > 0:
                # Transform to rover frame
                c = np.cos(self.camera_tilt)
                s = np.sin(self.camera_tilt)

                y_c_rot = points_near[:, 1] * c - points_near[:, 2] * s
                z_c_rot = points_near[:, 1] * s + points_near[:, 2] * c
                x_c_rot = points_near[:, 0]

                x_r_near = z_c_rot
                y_r_near = -x_c_rot
                z_r_near = -y_c_rot + self.camera_height
                depths_near = z_c_rot  # Store depth values

                # Standard obstacle detection for near range
                is_obstacle_near = z_r_near > self.obstacle_thresh

            # Process far-range points (conservative - higher threshold to avoid floor false positives)
            if len(points_far) > 0:
                # Transform to rover frame
                c = np.cos(self.camera_tilt)
                s = np.sin(self.camera_tilt)

                y_c_rot_far = points_far[:, 1] * c - points_far[:, 2] * s
                z_c_rot_far = points_far[:, 1] * s + points_far[:, 2] * c
                x_c_rot_far = points_far[:, 0]

                x_r_far = z_c_rot_far
                y_r_far = -x_c_rot_far
                z_r_far = -y_c_rot_far + self.camera_height
                depths_far = z_c_rot_far  # Store depth values

                # Higher threshold for far range to avoid floor misclassification
                # At 3-4m, depth errors can cause floor to appear 20-30cm high
                is_obstacle_far = z_r_far > (self.obstacle_thresh + 0.20)  # +20cm tolerance

            # Combine near and far points
            if len(points_near) > 0 or len(points_far) > 0:
                x_r = np.concatenate([x_r_near, x_r_far])
                y_r = np.concatenate([y_r_near, y_r_far])
                depths = np.concatenate([depths_near, depths_far])
                is_obstacle = np.concatenate([is_obstacle_near, is_obstacle_far])

                if np.any(is_obstacle):
                    # Project to grid
                    scale = self.grid_size / self.range_m
                    grid_rows = self.grid_size - 1 - (x_r[is_obstacle] * scale).astype(np.int32)
                    grid_cols = (self.grid_size // 2) - (y_r[is_obstacle] * scale).astype(np.int32)

                    # Clip to bounds
                    valid = (grid_rows >= 0) & (grid_rows < self.grid_size) & \
                            (grid_cols >= 0) & (grid_cols < self.grid_size)

                    grid_rows = grid_rows[valid]
                    grid_cols = grid_cols[valid]
                    obstacle_depths = depths[is_obstacle][valid]

                    # Keep minimum distance per cell - VECTORIZED (no loops)
                    np.minimum.at(grid, (grid_rows, grid_cols), obstacle_depths)

        # Process LiDAR scan
        # NOTE: LiDAR provides 2D horizontal slice at a fixed height (~10cm off ground)
        # It's excellent for detecting walls/obstacles but may produce artifacts at edges
        # We use it ONLY for obstacle detection, not for filling the entire grid
        if laser_scan is not None:
            if hasattr(laser_scan, 'ranges'):
                ranges = np.array(laser_scan.ranges)
                angle_min = laser_scan.angle_min
                angle_increment = laser_scan.angle_increment
            else:
                ranges = np.array(laser_scan['ranges'])
                angle_min = laser_scan['angle_min']
                angle_increment = laser_scan['angle_increment']

            # Filter valid ranges - 0.15m to avoid self-hits from rover body
            valid_mask = (ranges > 0.15) & (ranges < self.range_m) & np.isfinite(ranges)

            if np.any(valid_mask):
                # Polar to Cartesian
                # LiDAR coordinate system: X forward, Y left (standard ROS)
                angles = angle_min + np.arange(len(ranges)) * angle_increment
                x = ranges * np.cos(angles)
                y = ranges * np.sin(angles)

                # Apply mask
                x = x[valid_mask]
                y = y[valid_mask]
                ranges_filtered = ranges[valid_mask]

                # Project to grid
                # Grid coordinate system: Robot at (127, 64), +X is UP (row 0), +Y is LEFT (col 0)
                scale = self.grid_size / self.range_m
                rows = self.grid_size - 1 - (x * scale).astype(np.int32)
                cols = (self.grid_size // 2) - (y * scale).astype(np.int32)

                # Clip to bounds - also exclude the robot footprint region
                valid = (rows >= 0) & (rows < self.grid_size - 15) & \
                        (cols >= 0) & (cols < self.grid_size)

                rows = rows[valid]
                cols = cols[valid]
                ranges_filtered = ranges_filtered[valid]

                # Keep minimum distance - VECTORIZED (no loops)
                np.minimum.at(grid, (rows, cols), ranges_filtered)

        # Force Clear Robot Footprint (Rectangular rover dimensions)
        # Robot is at CENTER (64, 64) for 128x128 grid
        r_center, c_center = self.grid_size // 2, self.grid_size // 2
        grid = clear_rectangular_footprint(grid, r_center, c_center,
                                           0.30, 0.40, self.resolution)  # 30cm x 40cm rover
        grid[grid == 0.0] = self.range_m  # Set cleared area to max distance (free)

        # Apply distance transform for smooth gradients
        # First create binary obstacle mask
        # Any cell with a value significantly less than range_m contained a hit
        # We use a 10cm buffer to avoid potential float noise from initialization
        occupied = grid < (self.range_m - 0.1)

        # Compute Euclidean distance transform
        if np.any(occupied):
            # EDT gives distance in pixels, multiply by resolution to get meters
            distance_grid = ndimage.distance_transform_edt(~occupied) * self.resolution
            
            # SAFETY CHECK: If EDT produces all zeros (shouldn't happen, but defensive)
            # or NaN/Inf values, fall back to using the original grid values
            if np.all(distance_grid == 0) or np.any(np.isnan(distance_grid)) or np.any(np.isinf(distance_grid)):
                # Use the original distance values from grid (pre-EDT)
                # These are actual measured distances to obstacles
                distance_grid = grid.copy()
        else:
            # No obstacles detected - everything is free space
            distance_grid = np.full_like(grid, self.range_m)

        # Normalize to [0, 1] for network
        # Ensure no negative values and no values > range_m
        distance_grid = np.clip(distance_grid, 0.0, self.range_m)
        distance_grid = distance_grid / self.range_m

        # FINAL SAFETY CHECK: Ensure center region (robot footprint) is never 0
        # Robot is at CENTER (64, 64). The 10x10 patch around robot should be clear
        r_center, c_center = self.grid_size // 2, self.grid_size // 2
        safety_patch = distance_grid[r_center-5:r_center+6, c_center-5:c_center+6]
        if np.min(safety_patch) < 0.05:  # Less than 20cm (0.05 * 4.0m)
            # Something is wrong - footprint should be clear
            # Set to a safe minimum value (at least 0.5m)
            min_safe_val = 0.5 / self.range_m  # 0.125 normalized
            distance_grid[r_center-5:r_center+6, c_center-5:c_center+6] = np.maximum(
                distance_grid[r_center-5:r_center+6, c_center-5:c_center+6],
                min_safe_val
            )

        return distance_grid

    def _update_exploration_history(self, robot_pose):
        """
        Update persistent exploration map with current position.

        Args:
            robot_pose: (x, y, theta) in meters and radians
        """
        # Apply decay to entire map
        self.exploration_map *= self.decay_rate

        # Handle odometry updates
        if self.last_pose is not None:
            # Compute delta
            dx = robot_pose[0] - self.last_pose[0]
            dy = robot_pose[1] - self.last_pose[1]
            dtheta = robot_pose[2] - self.last_pose[2]

            # Shift map (inverse motion)
            dx_px = dx / self.resolution
            dy_px = dy / self.resolution

            # Create affine transform
            M_rot = cv2.getRotationMatrix2D(
                (self.exploration_center, self.exploration_center),
                -np.degrees(dtheta),
                1.0
            )
            M_rot[0, 2] += dy_px  # Col shift
            M_rot[1, 2] += dx_px  # Row shift

            # Apply warp
            self.exploration_map = cv2.warpAffine(
                self.exploration_map,
                M_rot,
                (512, 512),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0
            )

        self.last_pose = robot_pose

        # Mark current position as visited
        grid_x, grid_y = self.exploration_center, self.exploration_center
        radius = 8  # ~25cm radius at 3.125cm/pixel

        # Create circular mask
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x**2 + y**2 <= radius**2

        # Mark as visited
        y_min = max(0, grid_y - radius)
        y_max = min(512, grid_y + radius + 1)
        x_min = max(0, grid_x - radius)
        x_max = min(512, grid_x + radius + 1)

        mask_y_min = radius - (grid_y - y_min)
        mask_y_max = radius + (y_max - grid_y)
        mask_x_min = radius - (grid_x - x_min)
        mask_x_max = radius + (x_max - grid_x)

        self.exploration_map[y_min:y_max, x_min:x_max][
            mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max]
        ] = 1.0

    def _compute_confidence(self, depth_img, laser_scan):
        """
        Compute obstacle confidence map.
        Higher confidence where both depth and LiDAR agree.
        Normalized to [0, 1].
        """
        confidence = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Depth camera contributes 0.5
        if depth_img is not None and depth_img.size > 0:
            if depth_img.dtype == np.uint16:
                depth = depth_img.astype(np.float32) * 0.001
            else:
                depth = depth_img

            # Valid depth mask (simple version: non-zero)
            valid_depth = (depth > 0.1) & (depth < self.range_m)

            if np.any(valid_depth):
                # Project valid regions to grid and mark confidence
                z_c = depth
                x_c = z_c * self.x_mult
                y_c = z_c * self.y_mult

                points_c = np.stack([x_c, y_c, z_c], axis=-1).reshape(-1, 3)
                valid_mask = (points_c[:, 2] > 0.1) & (points_c[:, 2] < self.range_m)
                points_c = points_c[valid_mask]

                if len(points_c) > 0:
                    # Transform to rover frame
                    c = np.cos(self.camera_tilt)
                    s = np.sin(self.camera_tilt)

                    y_c_rot = points_c[:, 1] * c - points_c[:, 2] * s
                    z_c_rot = points_c[:, 1] * s + points_c[:, 2] * c
                    x_c_rot = points_c[:, 0]

                    x_r = z_c_rot
                    y_r = -x_c_rot

                    # Project to grid
                    scale = self.grid_size / self.range_m
                    grid_rows = self.grid_size - 1 - (x_r * scale).astype(np.int32)
                    grid_cols = (self.grid_size // 2) - (y_r * scale).astype(np.int32)

                    valid = (grid_rows >= 0) & (grid_rows < self.grid_size) & \
                            (grid_cols >= 0) & (grid_cols < self.grid_size)

                    grid_rows = grid_rows[valid]
                    grid_cols = grid_cols[valid]

                    # Mark with confidence 0.5
                    confidence[grid_rows, grid_cols] = 0.5

        # LiDAR contributes additional 0.5
        if laser_scan is not None:
            if hasattr(laser_scan, 'ranges'):
                ranges = np.array(laser_scan.ranges)
                angle_min = laser_scan.angle_min
                angle_increment = laser_scan.angle_increment
            else:
                ranges = np.array(laser_scan['ranges'])
                angle_min = laser_scan['angle_min']
                angle_increment = laser_scan['angle_increment']

            valid_mask = (ranges > 0.05) & (ranges < self.range_m)

            if np.any(valid_mask):
                angles = angle_min + np.arange(len(ranges)) * angle_increment
                x = ranges * np.cos(angles)
                y = ranges * np.sin(angles)

                x = x[valid_mask]
                y = y[valid_mask]

                scale = self.grid_size / self.range_m
                rows = self.grid_size - 1 - (x * scale).astype(np.int32)
                cols = (self.grid_size // 2) - (y * scale).astype(np.int32)

                valid = (rows >= 0) & (rows < self.grid_size) & \
                        (cols >= 0) & (cols < self.grid_size)

                rows = rows[valid]
                cols = cols[valid]

                # Add 0.5 confidence (max 1.0 when both sensors agree)
                confidence[rows, cols] = np.minimum(confidence[rows, cols] + 0.5, 1.0)

        return confidence

    def _compute_terrain_height(self, depth_img):
        """
        Compute terrain height map relative to ground plane.
        Normalized to [0, 1] where 0 = ground level, 1 = max obstacle height.
        """
        height_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        if depth_img is None or depth_img.size == 0:
            return height_grid

        # Handle uint16 input
        if depth_img.dtype == np.uint16:
            depth = depth_img.astype(np.float32) * 0.001
        else:
            depth = depth_img

        # Unproject to 3D
        z_c = depth
        x_c = z_c * self.x_mult
        y_c = z_c * self.y_mult

        points_c = np.stack([x_c, y_c, z_c], axis=-1).reshape(-1, 3)

        # Filter valid depth
        valid_mask = (points_c[:, 2] > 0.1) & (points_c[:, 2] < self.range_m)
        points_c = points_c[valid_mask]

        if len(points_c) == 0:
            return height_grid

        # Transform to rover frame
        c = np.cos(self.camera_tilt)
        s = np.sin(self.camera_tilt)

        y_c_rot = points_c[:, 1] * c - points_c[:, 2] * s
        z_c_rot = points_c[:, 1] * s + points_c[:, 2] * c
        x_c_rot = points_c[:, 0]

        x_r = z_c_rot
        y_r = -x_c_rot
        z_r = -y_c_rot + self.camera_height

        # Project to grid
        scale = self.grid_size / self.range_m
        grid_rows = self.grid_size - 1 - (x_r * scale).astype(np.int32)
        grid_cols = (self.grid_size // 2) - (y_r * scale).astype(np.int32)

        # Clip to bounds
        valid = (grid_rows >= 0) & (grid_rows < self.grid_size) & \
                (grid_cols >= 0) & (grid_cols < self.grid_size)

        grid_rows = grid_rows[valid]
        grid_cols = grid_cols[valid]
        heights = z_r[valid]

        # Keep maximum height per cell - VECTORIZED (no loops)
        np.maximum.at(height_grid, (grid_rows, grid_cols), heights)

        # Normalize to [0, 1] range
        # Floor level = 0, max obstacle (0.5m) = 1.0
        height_grid = np.clip((height_grid + 0.1) / 0.6, 0.0, 1.0)

        return height_grid

    def _get_centered_crop(self, map_array, center, crop_size):
        """
        Extract centered crop from map with robot at bottom-center.

        Args:
            map_array: Large persistent map
            center: Robot position in map
            crop_size: Output grid size

        Returns:
            crop: (crop_size, crop_size) array
        """
        # Robot should be at bottom-center of crop
        # For 128x128 crop, robot at row 127, col 64
        robot_row_in_crop = crop_size - 1
        robot_col_in_crop = crop_size // 2

        # Calculate map region
        start_row = center - robot_row_in_crop
        end_row = start_row + crop_size
        start_col = center - robot_col_in_crop
        end_col = start_col + crop_size

        # Handle boundaries
        if start_row < 0 or end_row > map_array.shape[0] or \
           start_col < 0 or end_col > map_array.shape[1]:
            # Create padded crop
            crop = np.zeros((crop_size, crop_size), dtype=map_array.dtype)

            # Calculate valid region
            src_r_start = max(0, start_row)
            src_r_end = min(map_array.shape[0], end_row)
            src_c_start = max(0, start_col)
            src_c_end = min(map_array.shape[1], end_col)

            dst_r_start = src_r_start - start_row
            dst_r_end = dst_r_start + (src_r_end - src_r_start)
            dst_c_start = src_c_start - start_col
            dst_c_end = dst_c_start + (src_c_end - src_c_start)

            crop[dst_r_start:dst_r_end, dst_c_start:dst_c_end] = \
                map_array[src_r_start:src_r_end, src_c_start:src_c_end]

            return crop
        else:
            return map_array[start_row:end_row, start_col:end_col].copy()


class RGBDProcessor:
    """Process RGB and Depth into 4-channel RGBA input for model."""

    def __init__(self, max_range=4.0):
        self.max_range = max_range

    def process(self, rgb_img, depth_img):
        """
        Args:
            rgb_img: (240, 424, 3) uint8 RGB image
            depth_img: (240, 424) uint16 depth image (mm)

        Returns:
            rgbd: (4, 240, 424) float32 [0,1] normalized
        """
        # Validate input shapes
        if rgb_img is None or depth_img is None:
            raise ValueError(f"Invalid input: rgb_img={rgb_img is not None}, depth_img={depth_img is not None}")

        if len(rgb_img.shape) != 3 or rgb_img.shape[2] != 3:
            raise ValueError(f"Expected RGB image shape (H, W, 3), got {rgb_img.shape}")

        if len(depth_img.shape) != 2:
            raise ValueError(f"Expected depth image shape (H, W), got {depth_img.shape}")

        if rgb_img.shape[:2] != depth_img.shape[:2]:
            raise ValueError(
                f"Shape mismatch: RGB {rgb_img.shape} vs Depth {depth_img.shape}. "
                f"RGB and depth must have same spatial dimensions."
            )

        # Convert RGB to float32 and normalize
        rgb_normalized = rgb_img.astype(np.float32) / 255.0

        # Process depth (same as current _process_depth)
        if depth_img.dtype == np.uint16:
            depth = depth_img.astype(np.float32) * 0.001
        else:
            depth = depth_img.astype(np.float32)

        # Apply median filter
        depth_filtered = cv2.medianBlur(depth_img, 3).astype(np.float32) * 0.001
        depth_normalized = np.clip(depth_filtered, 0.0, self.max_range) / self.max_range
        depth_normalized[depth == 0.0] = 1.0

        # Stack channels: RGB + Depth
        rgbd = np.stack([
            rgb_normalized[..., 0],  # R
            rgb_normalized[..., 1],  # G
            rgb_normalized[..., 2],  # B
            depth_normalized          # Depth
        ], axis=0)

        return rgbd.astype(np.float32)

class RawSensorProcessor:
    """
    Simplified processor for dual-encoder SAC architecture.

    Processes sensors into raw representations for separate encoders:
    - Laser: 128×128 binary occupancy grid
    - Depth: 424×240 full-resolution normalized depth image

    No EDT, no complex fusion - let the CNN learn features directly.

    Improvements:
    - Configurable decay rate for persistence buffer
    - Standardized robot position (bottom-center)
    - Rectangular footprint clearing
    - Improved min range filtering (0.15m)
    - Input validation
    """

    def __init__(self, grid_size=128, max_range=4.0, scan_decay_rate=0.85):
        """
        Args:
            grid_size: Size of laser occupancy grid (default 128×128)
            max_range: Maximum sensor range in meters (for normalization)
            scan_decay_rate: Decay rate for persistence buffer (0.85 = slower, 0.6 = faster)
        """
        self.grid_size = grid_size
        self.max_range = max_range
        self.resolution = max_range / grid_size  # 4m / 128 = 3.125 cm/px
        self.scan_decay_rate = scan_decay_rate

        # Persistence buffer for laser scan (Decay)
        self.scan_buffer = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Robot position: CENTER for full 360° awareness
        # Rover now at (64, 64) for 128x128 grid - can see in all directions
        self.robot_row = grid_size // 2  # 64 for 128x128
        self.robot_col = grid_size // 2  # 64 for 128x128
        
        # Rover dimensions for footprint marker (30cm x 40cm)
        self.rover_width_px = int(0.30 / self.resolution)   # ~10 pixels
        self.rover_length_px = int(0.40 / self.resolution)  # ~13 pixels

    def process(self, depth_img, laser_scan):
        """
        Process raw sensor data into dual inputs for CNN.

        Args:
            depth_img: (424, 240) uint16 depth image in mm, or float32 in meters
            laser_scan: LaserScan message with ranges, angle_min, angle_increment

        Returns:
            laser_grid: (128, 128) float32 [0.0, 1.0] - Binary occupancy
            depth_processed: (424, 240) float32 [0.0, 1.0] - Normalized depth
        """
        # Process laser to 128×128 occupancy
        laser_grid = self._process_laser(laser_scan)

        # Process depth to FULL RESOLUTION 424×240
        depth_processed = self._process_depth(depth_img)

        return laser_grid, depth_processed

    def _process_laser(self, scan):
        """
        Convert LiDAR polar scan to 128×128 binary occupancy grid.

        Args:
            scan: LaserScan message or dict with 'ranges', 'angle_min', 'angle_increment'

        Returns:
            grid: (128, 128) float32 binary occupancy (0=free, 1=occupied)
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        if scan is None:
            return grid

        # Extract scan data with validation
        if hasattr(scan, 'ranges'):
            ranges = np.array(scan.ranges)
            angle_min = scan.angle_min
            angle_increment = scan.angle_increment
        else:
            ranges = np.array(scan.get('ranges', []))
            if len(ranges) == 0:
                return grid
            angle_min = scan.get('angle_min', -np.pi)
            angle_increment = scan.get('angle_increment', 0.01)

        # Validate ranges
        if len(ranges) == 0 or not np.any(np.isfinite(ranges)):
            return grid

        # Filter valid ranges (0.15m min to avoid self-hits, improved from 0.25m)
        valid = (ranges > 0.15) & (ranges < self.max_range) & np.isfinite(ranges)

        if not np.any(valid):
            return grid

        # Polar → Cartesian conversion
        # LiDAR coordinate system: X forward, Y left (standard ROS)
        angles = angle_min + np.arange(len(ranges)) * angle_increment
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        # Apply valid mask
        x = x[valid]
        y = y[valid]

        # Project to grid
        # Robot at CENTER (row 64, col 64) for 360° awareness
        scale = self.grid_size / self.max_range  # ~32 pixels/meter

        rows = self.robot_row - (x * scale).astype(np.int32)
        cols = self.robot_col - (y * scale).astype(np.int32)

        # Clip to bounds
        valid_idx = (rows >= 0) & (rows < self.grid_size) & \
                    (cols >= 0) & (cols < self.grid_size)

        rows = rows[valid_idx]
        cols = cols[valid_idx]

        if len(rows) == 0:
            return grid

        # Binary occupancy: 1.0 where obstacles detected
        current_view = np.zeros_like(grid)
        current_view[rows, cols] = 1.0

        # Accumulate into buffer with max-hold
        self.scan_buffer = np.maximum(self.scan_buffer, current_view)

        # Output is buffer state
        grid = self.scan_buffer.copy()

        # Decay buffer for next frame with configurable rate
        # Default 0.85 decay -> survives ~4 frames at 30Hz
        # Frame 0: 1.0 (Scan arrives at 10Hz)
        # Frame 1: 0.85
        # Frame 2: 0.72
        # Frame 3: 0.61 (Next scan arrives)
        self.scan_buffer *= self.scan_decay_rate

        # Clear robot footprint (rectangular rover dimensions)
        grid = clear_rectangular_footprint(grid, self.robot_row, self.robot_col,
                                           0.30, 0.40, self.resolution)

        # Add rover footprint marker (unique 0.5 value to distinguish from obstacles)
        # This helps the model understand ego-centric perspective
        # Rover is oriented facing "up" (row 0 direction) with forward arrow
        r_center, c_center = self.robot_row, self.robot_col
        half_w = self.rover_width_px // 2
        half_l = self.rover_length_px // 2
        
        # Draw rover body as 0.5 (distinct from 0.0=free, 1.0=obstacle)
        grid[r_center - half_l:r_center + half_l + 1,
             c_center - half_w:c_center + half_w + 1] = 0.5
        
        # Draw forward arrow/triangle at front of rover (rows < center)
        arrow_len = 3
        for i in range(arrow_len):
            grid[r_center - half_l - 1 - i, c_center - i:c_center + i + 1] = 0.7

        return grid

    def _process_depth(self, depth_img):
        """
        Process raw depth image to normalized full-resolution format.

        Args:
            depth_img: (H, W) numpy array, uint16 (mm) or float32 (meters)

        Returns:
            depth_normalized: (424, 240) float32 [0.0, 1.0]
                - 0.0 = very close
                - 1.0 = far/invalid
        """
        if depth_img is None or depth_img.size == 0:
            # Return empty grid at native resolution
            return np.ones((240, 424), dtype=np.float32)  # All far/unknown

        # Convert uint16 mm → float32 meters
        if depth_img.dtype == np.uint16:
            depth = depth_img.astype(np.float32) * 0.001
        else:
            depth = depth_img.astype(np.float32)

        # Apply 3×3 median filter to reduce speckle noise
        # medianBlur requires uint8 or uint16 input
        if depth_img.dtype == np.uint16:
            depth_filtered = cv2.medianBlur(depth_img, 3).astype(np.float32) * 0.001
        else:
            # Convert back to uint16 for filtering
            depth_uint16 = np.clip(depth * 1000, 0, 65535).astype(np.uint16)
            depth_filtered = cv2.medianBlur(depth_uint16, 3).astype(np.float32) * 0.001

        depth = depth_filtered

        # NO RESIZING - Keep native 424×240 resolution
        # Normalize to [0.0, 1.0]
        depth_clipped = np.clip(depth, 0.0, self.max_range)
        depth_normalized = depth_clipped / self.max_range

        # Invalid/zero depth → 1.0 (far/unknown, same as max range)
        depth_normalized[depth == 0.0] = 1.0

        return depth_normalized  # Shape: (240, 424)


class UnifiedBEVProcessor:
    """
    Unified Birds-Eye-View (BEV) processor that fuses LiDAR and Depth camera
    into a single 2-channel 256×256 grid.

    Channel 0: LiDAR occupancy (360° top-down binary occupancy)
    Channel 1: Depth-projected occupancy (front 87° arc projected to BEV)

    This replaces the dual-input architecture (LaserEncoder + DepthEncoder)
    with a single unified representation that's spatially aligned.

    Advantages:
    - Single encoder instead of two (smaller model)
    - Explicit spatial alignment between modalities
    - Higher resolution: 256×256 at 4m = 1.56cm/pixel
    - LiDAR provides 360° coverage, depth reinforces front arc
    """

    def __init__(self,
                 grid_size=128,
                 max_range=4.0,
                 # D435i camera parameters
                 depth_width=848,
                 depth_height=100,
                 fx=421.0,   # D435i focal length at 848x100
                 fy=421.0,
                 cx=424.0,   # Center x (848/2)
                 cy=50.0,    # Center y (100/2)
                 camera_height=0.187,      # Camera height above ground
                 camera_tilt_deg=0.0,      # Camera tilt angle
                 obstacle_height_thresh=0.10,  # Min height to be obstacle (10cm)
                 scan_decay_rate=0.85):
        """
        Args:
            grid_size: Output grid size (256×256)
            max_range: Maximum sensor range in meters
            depth_width, depth_height: D435i depth image dimensions
            fx, fy, cx, cy: Camera intrinsics
            camera_height: Height of camera above ground plane (meters)
            camera_tilt_deg: Downward tilt of camera (degrees)
            obstacle_height_thresh: Minimum height above ground to be obstacle
            scan_decay_rate: Decay rate for LiDAR persistence buffer
        """
        self.grid_size = grid_size
        self.max_range = max_range
        self.resolution = max_range / grid_size  # 4m / 256 = 1.5625 cm/px

        # Camera parameters
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.camera_height = camera_height
        self.camera_tilt = np.radians(camera_tilt_deg)
        self.obstacle_thresh = obstacle_height_thresh

        # LiDAR persistence buffer (for temporal smoothing)
        self.scan_decay_rate = scan_decay_rate
        self.scan_buffer = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Robot position: CENTER for full 360° awareness
        self.robot_row = grid_size // 2  # 64 for 128x128
        self.robot_col = grid_size // 2  # 64 for 128x128
        
        # Rover dimensions for footprint marker (30cm x 40cm)
        self.rover_width_px = int(0.30 / self.resolution)   # ~10 pixels
        self.rover_length_px = int(0.40 / self.resolution)  # ~13 pixels

        # Pre-compute depth unprojection matrices (for 848 x 100 depth)
        u, v = np.meshgrid(np.arange(depth_width), np.arange(depth_height))
        self.x_mult = (u - cx) / fx
        self.y_mult = (v - cy) / fy

    def process(self, depth_img, laser_scan):
        """
        Process raw sensor data into unified 2-channel BEV grid.

        Args:
            depth_img: (100, 848) or (H, W) uint16 depth image in mm, or float32 in meters
            laser_scan: LaserScan message with ranges, angle_min, angle_increment

        Returns:
            bev_grid: (2, 256, 256) float32 [0.0, 1.0]
                - Channel 0: LiDAR occupancy (0=free, 1=occupied)
                - Channel 1: Depth occupancy (0=free, 1=occupied)
        """
        # Channel 0: LiDAR occupancy
        laser_bev = self._process_laser(laser_scan)

        # Channel 1: Depth-projected occupancy
        depth_bev = self._process_depth_to_bev(depth_img)

        # Stack channels: (2, 256, 256)
        bev_grid = np.stack([laser_bev, depth_bev], axis=0)

        return bev_grid.astype(np.float32)

    def _process_laser(self, scan):
        """
        Convert LiDAR polar scan to 256×256 binary occupancy grid.

        Args:
            scan: LaserScan message or dict with 'ranges', 'angle_min', 'angle_increment'

        Returns:
            grid: (256, 256) float32 binary occupancy (0=free, 1=occupied)
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        if scan is None:
            return grid

        # Extract scan data with validation
        if hasattr(scan, 'ranges'):
            ranges = np.array(scan.ranges)
            angle_min = scan.angle_min
            angle_increment = scan.angle_increment
        else:
            ranges = np.array(scan.get('ranges', []))
            if len(ranges) == 0:
                return grid
            angle_min = scan.get('angle_min', -np.pi)
            angle_increment = scan.get('angle_increment', 0.01)

        # Validate ranges
        if len(ranges) == 0 or not np.any(np.isfinite(ranges)):
            return grid

        # Filter valid ranges (0.15m min to avoid self-hits)
        valid = (ranges > 0.15) & (ranges < self.max_range) & np.isfinite(ranges)

        if not np.any(valid):
            return grid

        # Polar → Cartesian conversion
        # LiDAR coordinate system: X forward, Y left (standard ROS)
        angles = angle_min + np.arange(len(ranges)) * angle_increment
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        # Apply valid mask
        x = x[valid]
        y = y[valid]

        # Project to grid
        # Robot at bottom-center (row 255, col 128) for standardization
        scale = self.grid_size / self.max_range  # 64 pixels/meter for 256/4

        rows = self.robot_row - (x * scale).astype(np.int32)
        cols = self.robot_col - (y * scale).astype(np.int32)

        # Clip to bounds
        valid_idx = (rows >= 0) & (rows < self.grid_size) & \
                    (cols >= 0) & (cols < self.grid_size)

        rows = rows[valid_idx]
        cols = cols[valid_idx]

        if len(rows) == 0:
            return grid

        # Binary occupancy: 1.0 where obstacles detected
        current_view = np.zeros_like(grid)
        current_view[rows, cols] = 1.0

        # Accumulate into buffer with max-hold
        self.scan_buffer = np.maximum(self.scan_buffer, current_view)

        # Output is buffer state
        grid = self.scan_buffer.copy()

        # Decay buffer for next frame
        self.scan_buffer *= self.scan_decay_rate

        # Clear robot footprint (30cm x 40cm rover)
        grid = clear_rectangular_footprint(grid, self.robot_row, self.robot_col,
                                           0.30, 0.40, self.resolution)

        # Add rover footprint marker (unique 0.5 value to distinguish from obstacles)
        # This helps the model understand ego-centric perspective
        r_center, c_center = self.robot_row, self.robot_col
        half_w = self.rover_width_px // 2
        half_l = self.rover_length_px // 2
        
        # Draw rover body as 0.5 (distinct from 0.0=free, 1.0=obstacle)
        grid[r_center - half_l:r_center + half_l + 1,
             c_center - half_w:c_center + half_w + 1] = 0.5
        
        # Draw forward arrow at front of rover (rows < center)
        arrow_len = 3
        for i in range(arrow_len):
            grid[r_center - half_l - 1 - i, c_center - i:c_center + i + 1] = 0.7

        return grid

    def _process_depth_to_bev(self, depth_img):
        """
        Project depth camera image to 256×256 BEV occupancy grid.

        Uses pinhole camera model to unproject depth pixels to 3D,
        then projects to top-down view.

        Args:
            depth_img: (H, W) numpy array, uint16 (mm) or float32 (meters)

        Returns:
            grid: (256, 256) float32 binary occupancy (0=free, 1=occupied)
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        if depth_img is None or depth_img.size == 0:
            return grid

        # Handle input shape mismatch (resize if needed)
        h, w = depth_img.shape[:2]
        if (h, w) != (self.depth_height, self.depth_width):
            # Resize to expected dimensions
            if depth_img.dtype == np.uint16:
                depth_img = cv2.resize(depth_img, (self.depth_width, self.depth_height), 
                                       interpolation=cv2.INTER_NEAREST)
            else:
                depth_img = cv2.resize(depth_img, (self.depth_width, self.depth_height),
                                       interpolation=cv2.INTER_LINEAR)
            # Update unprojection matrices for new size
            u, v = np.meshgrid(np.arange(self.depth_width), np.arange(self.depth_height))
            x_mult = (u - self.cx) / self.fx
            y_mult = (v - self.cy) / self.fy
        else:
            x_mult = self.x_mult
            y_mult = self.y_mult

        # Convert uint16 mm → float32 meters
        if depth_img.dtype == np.uint16:
            depth = depth_img.astype(np.float32) * 0.001
        else:
            depth = depth_img.astype(np.float32)

        # 1. Unproject to 3D (Camera Frame)
        z_c = depth
        x_c = z_c * x_mult
        y_c = z_c * y_mult

        # Flatten for point cloud processing
        points_c = np.stack([x_c, y_c, z_c], axis=-1).reshape(-1, 3)

        # Filter invalid depth
        valid_mask = (points_c[:, 2] > 0.15) & (points_c[:, 2] < self.max_range)
        points_c = points_c[valid_mask]

        if len(points_c) == 0:
            return grid

        # 2. Transform to Rover Frame
        # Camera frame: Z forward, X right, Y down
        # Rover frame: X forward, Y left, Z up
        c = np.cos(self.camera_tilt)
        s = np.sin(self.camera_tilt)

        # Apply camera tilt rotation around X-axis
        y_c_rot = points_c[:, 1] * c - points_c[:, 2] * s
        z_c_rot = points_c[:, 1] * s + points_c[:, 2] * c
        x_c_rot = points_c[:, 0]  # Unchanged

        # Map to Rover Frame
        # X_r = Z_c_rot (forward)
        # Y_r = -X_c_rot (left)
        # Z_r = -Y_c_rot + height (up)
        x_r = z_c_rot
        y_r = -x_c_rot
        z_r = -y_c_rot + self.camera_height

        # 3. Filter for obstacles (above ground threshold)
        is_obstacle = z_r > self.obstacle_thresh

        if not np.any(is_obstacle):
            return grid

        # 4. Project obstacles to grid
        scale = self.grid_size / self.max_range  # 64 pixels/meter

        # Robot at bottom-center (255, 128)
        grid_rows = self.robot_row - (x_r[is_obstacle] * scale).astype(np.int32)
        grid_cols = self.robot_col - (y_r[is_obstacle] * scale).astype(np.int32)

        # Clip to bounds
        valid_idx = (grid_rows >= 0) & (grid_rows < self.grid_size) & \
                    (grid_cols >= 0) & (grid_cols < self.grid_size)

        grid_rows = grid_rows[valid_idx]
        grid_cols = grid_cols[valid_idx]

        if len(grid_rows) == 0:
            return grid

        # Mark obstacle cells as 1.0
        grid[grid_rows, grid_cols] = 1.0

        # Morphological closing to fill gaps
        kernel = np.ones((3, 3), np.uint8)
        grid = cv2.morphologyEx((grid * 255).astype(np.uint8), 
                                cv2.MORPH_CLOSE, kernel).astype(np.float32) / 255.0

        # Clear robot footprint
        grid = clear_rectangular_footprint(grid, self.robot_row, self.robot_col,
                                           0.30, 0.40, self.resolution)

        return grid
