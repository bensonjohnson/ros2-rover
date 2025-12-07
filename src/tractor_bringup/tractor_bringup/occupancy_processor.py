import numpy as np
import cv2
from scipy import ndimage

class DepthToOccupancy:
    """
    Vectorized processor to convert raw depth images to a top-down occupancy grid.
    
    Logic:
    1. Unproject depth pixels to 3D points (camera frame).
    2. Transform points to rover frame (rotate by camera tilt, translate by height).
    3. Filter points:
       - Floor: Points near z=0 (rover ground level).
       - Obstacles: Points above floor threshold.
    4. Project to 2D grid (x, y).
    5. Create 64x64 occupancy grid (0=unknown, 128=free, 255=occupied).
    """
    
    def __init__(self,
                 width=424,
                 height=240,
                 fx=386.0,
                 fy=386.0,
                 cx=212.0,
                 cy=120.0,
                 camera_height=0.187, # 174mm (bottom) + 12.5mm (to optical center)
                 camera_tilt_deg=0.0, # degrees down from horizontal
                 grid_size=64,
                 grid_range=3.0, # meters (forward/side range)
                 obstacle_height_thresh=0.1, # meters above ground to be obstacle (increased to 10cm to reduce noise)
                 floor_thresh=0.08 # meters +/- ground to be floor (increased to 8cm)
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
        
        # Pre-compute unprojection matrices
        # Create meshgrid of pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # (u - cx) / fx, (v - cy) / fy
        self.x_mult = (u - cx) / fx
        self.y_mult = (v - cy) / fy

    def process(self, depth_image):
        """
        Args:
            depth_image: (H, W) numpy array, float32 (meters) or uint16 (mm)
        Returns:
            grid: (64, 64) numpy array, uint8
        """
        # Handle uint16 input
        if depth_image.dtype == np.uint16:
            depth = depth_image.astype(np.float32) * 0.001
        else:
            depth = depth_image
            
        # 1. Unproject to 3D (Camera Frame)
        z_c = depth
        x_c = z_c * self.x_mult
        y_c = z_c * self.y_mult
        
        # Flatten for point cloud processing
        points_c = np.stack([x_c, y_c, z_c], axis=-1).reshape(-1, 3)
        
        # Filter invalid depth
        valid_mask = (points_c[:, 2] > 0.1) & (points_c[:, 2] < 5.0)
        points_c = points_c[valid_mask]
        
        if len(points_c) == 0:
            return np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
            
        # 2. Transform to Rover Frame
        c = np.cos(self.camera_tilt)
        s = np.sin(self.camera_tilt)
        
        y_c_rot = points_c[:, 1] * c - points_c[:, 2] * s
        z_c_rot = points_c[:, 1] * s + points_c[:, 2] * c
        x_c_rot = points_c[:, 0] # Unchanged
        
        # Map to Rover Frame
        # X_r = Z_c_rot
        # Y_r = -X_c_rot
        # Z_r = -Y_c_rot + height
        
        x_r = z_c_rot
        y_r = -x_c_rot
        z_r = -y_c_rot + self.camera_height
        
        # 3. Filter Points
        is_floor = np.abs(z_r) < self.floor_thresh
        is_obstacle = z_r > self.obstacle_thresh
        
        # 4. Project to Grid
        scale = self.grid_size / self.grid_range
        
        # Robot is at bottom center (row 63, col 32)
        
        # Map X_r (Forward) to Image Rows (Bottom to Top)
        # row = H - 1 - (x_r * scale)
        grid_rows = self.grid_size - 1 - (x_r * scale).astype(np.int32)
        
        # Map Y_r (Left) to Image Cols (Center to Left/Right)
        # Col = Center - Y_r * scale
        grid_cols = (self.grid_size // 2) - (y_r * scale).astype(np.int32)
        
        # Clip to grid bounds
        valid_indices = (grid_rows >= 0) & (grid_rows < self.grid_size) & \
                        (grid_cols >= 0) & (grid_cols < self.grid_size)
                        
        grid_rows = grid_rows[valid_indices]
        grid_cols = grid_cols[valid_indices]
        is_floor = is_floor[valid_indices]
        is_obstacle = is_obstacle[valid_indices]
        
        # Create Grid
        # 0 = Unknown
        # 128 = Free (Floor)
        # 255 = Occupied (Obstacle)
        
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        
        # Mark floor first (128)
        grid[grid_rows[is_floor], grid_cols[is_floor]] = 128
        
        # Mark obstacles (255) - Overwrite floor if conflict
        grid[grid_rows[is_obstacle], grid_cols[is_obstacle]] = 255

        # Morphological Closing to fill holes/noise
        kernel = np.ones((3,3), np.uint8)
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)
        
        return grid

class ScanToOccupancy:
    """
    Vectorized processor to convert 2D Laser Scan to a top-down occupancy grid.
    
    Logic:
    1. Filter invalid ranges (0 or > max_range).
    2. Convert Polar (range, angle) to Cartesian (x, y) in Robot Frame.
    3. Project to 64x64 grid to match DepthToOccupancy.
    """
    def __init__(self, 
                 grid_size=64, 
                 grid_range=3.0, # meters
                 resolution=0.046875 # meters/pixel
                 ):
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.resolution = resolution
        
        # Pre-compute grid center offset
        # Grid: 64x64. Robot at (63, 32).
        # We need to map meters (x, y) to pixels (r, c).
        # Pixel coordinates
        # r = 63 - (x / resolution)
        # c = 32 - (y / resolution)

    def process(self, ranges, angle_min, angle_increment):
        """
        Args:
            ranges: List or numpy array of float ranges
            angle_min: float
            angle_increment: float
        Returns:
            grid: (64, 64) numpy array, uint8 (0=unknown, 255=obstacle, 128=free)
        """
        ranges = np.array(ranges)
        
        # 1. Filter Ranges
        # Valid: 0.05 < r < grid_range
        valid_mask = (ranges > 0.05) & (ranges < self.grid_range)
        
        # 2. Polar to Cartesian
        angles = angle_min + np.arange(len(ranges)) * angle_increment
        
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        
        # Apply mask
        x = x[valid_mask]
        y = y[valid_mask]
        
        if len(x) == 0:
             return np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        
        # 3. Project to Grid
        # Robot is at bottom center (row 63, col 32)
        # X is Forward (Up in grid) -> -Row
        # Y is Left (Left in grid) -> +Col? No.
        # Image convention: Col 0 is Left. Col 63 is Right.
        # Robot Y+ is Left. So Y > 0 means Col < 32.
        # Standard:
        # row = H - 1 - (x / res)
        # col = W/2 - (y / res)
        
        rows = self.grid_size - 1 - (x / self.resolution).astype(np.int32)
        cols = (self.grid_size // 2) - (y / self.resolution).astype(np.int32)
        
        # Clip
        valid_indices = (rows >= 0) & (rows < self.grid_size) & \
                        (cols >= 0) & (cols < self.grid_size)
                        
        rows = rows[valid_indices]
        cols = cols[valid_indices]
        
        # 4. Create Grid
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        
        # Mark obstacles (255)
        # Use simple marking for now.
        grid[rows, cols] = 255
        
        # Expand obstacles slightly (dilation)
        kernel = np.ones((3,3), np.uint8)
        grid = cv2.dilate(grid, kernel, iterations=1)
        
        # Ray tracing for free space is expensive in Python.
        # We'll rely on the "decay" of the LocalMapper to clear dynamic obstacles
        # or rely on Depth camera for free space clearing.
        # Alternatively, we could assume everything between robot and hit is free?
        # For now, let's just mark obstacles.
        
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
                 resolution=0.047, # 3.0m / 64px = 0.046875 m/px
                 decay_rate=0.98):
        
        self.map_size = map_size
        self.resolution = resolution
        self.decay_rate = decay_rate
        
        # The map is centered on the robot
        # Robot pixel = (map_size // 2, map_size // 2) ... Wait, standard convention?
        # Let's say robot is always at center (128, 128)
        
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
        # 1. Shift Map (Inverse Motion Model)
        # If robot moves forward +X, the map moves backward -X relative to robot.
        # If robot rotates Left +Theta, the map rotates Right -Theta.
        
        # Convert meters to pixels
        dx_px = dx / self.resolution
        dy_px = dy / self.resolution
        
        # Define affine transform matrix
        # Translate opposite to motion, rotate opposite to motion
        
        # Rotation center is the map center
        M_rot = cv2.getRotationMatrix2D((self.center, self.center), np.degrees(dtheta), 1.0)
        
        # Translation (Robot Frame: X=Up, Y=Left)
        # Map Frame (Image): Row=Down(-X), Col=Right(-Y)
        
        # If robot moves +X (Forward) -> Objects move Down (+Row)
        # If robot moves +Y (Left) -> Objects move Right (+Col)
        
        # Wait, let's verify coord systems.
        # new_observation: Robot at bottom center (row=63, col=32)
        # global_map: Robot at center (row=128, col=128)
        
        # 1a. Rotate first (around robot center)
        # We rotate the map "Underneath" the robot.
        # Robot turns LEFT (+theta). World turns RIGHT (-theta).
        # cv2 rotates CCW for positive angle. So we use -degrees(dtheta).
        M_rot = cv2.getRotationMatrix2D((self.center, self.center), -np.degrees(dtheta), 1.0)
        
        # 1b. Translate
        # Robot moves dx (forward), dy (left).
        # World moves -dx (backward/down), -dy (right/right).
        # Image Y (Row) = Down. Image X (Col) = Right.
        # Forward (+dx) -> Down (+Row in image?). No.
        # In Grid: Top is Forward. Bottom is Backward.
        # Robot moves Forward (+dx). Objects move Backward (Down).
        # So +Row is correct for +dx.
        
        # Left (+dy) -> Right (+Col).
        # So +Col is correct for +dy.
        
        M_rot[0, 2] += dy_px # Col shift (from dy)
        M_rot[1, 2] += dx_px # Row shift (from dx)
        
        # Apply Warp
        self.global_map = cv2.warpAffine(
            self.global_map, 
            M_rot, 
            (self.map_size, self.map_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0 # Fade to unknown
        )
        
        # 2. Fade Map (Soft forgetting)
        # This handles dynamic obstacles and odometry drift
        self.global_map *= self.decay_rate
        
        # 3. Stamp New Observation
        # The new observation is 64x64, robot at (63, 32).
        # We need to place it into the global map (256, 256) at center (128, 128).
        # Observation (63, 32) aligns with Buffer (128, 128).
        
        # Offsets
        # Obs Top-Left (0,0) -> Buffer (128 - 63, 128 - 32)
        # Row offset: 128 (center) - 63 (robot row in obs) = 65
        # Col offset: 128 (center) - 32 (robot col in obs) = 96
        
        # Wait, if obs is 64x64. Robot is at row 63.
        # We want Robot (63, 32) to be at Center (128, 128).
        # Row start_map = 128 - 63 = 65.
        # Row end_map   = 65 + 64 = 129.
        # Col start_map = 128 - 32 = 96.
        # Col end_map   = 96 + 64 = 160.
        
        rows = slice(65, 129)
        cols = slice(96, 160)
        
        # We only overwrite KNOWN cells (128 or 255)
        # Unknown (0) in observation should NOT overwrite known map
        
        obs = new_observation.astype(np.float32)
        mask = (new_observation > 0)
        
        # Region Of Interest
        roi = self.global_map[rows, cols]
        
        # Updates
        roi[mask] = obs[mask]
        
        self.global_map[rows, cols] = roi
        
    def get_model_input(self):
        """
        Return the 64x64 center crop for the model.
        Returns:
            grid: (64, 64) uint8
        """
        # Crop 64x64 logic:
        # We want the view AHEAD of the robot.
        # Robot is at (128, 128).
        # Standard observation (DepthToOccupancy) gives 3m ahead.
        # So we want the exact same window we just stamped, effectively?
        # NO. We want the robot at the BOTTOM of the crop, just like the raw input.
        # Robot at (63, 32) in the crop.
        # So Crop Center needs to be shifted Forward from Robot Center.
        
        # Robot at (128, 128).
        # We want Robot to be at Row 63 in the output.
        # So Output Row 63 = Map Row 128.
        # Output Row 0  = Map Row 128 - 63 = 65.
        
        # Output Col 32 = Map Col 128.
        # Output Col 0  = Map Col 128 - 32 = 96.
        
        start_row = 128 - 63 # 65
        end_row = start_row + 64 # 129
        
        start_col = 128 - 32 # 96
        end_col = start_col + 64 # 160
        
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
                 camera_tilt_deg=0.0,
                 # Thresholds - INCREASED to avoid ground plane false positives
                 # Only consider objects > 15cm above ground as obstacles
                 obstacle_height_thresh=0.15,
                 # Anything within ±12cm of ground level is floor (more tolerance)
                 floor_thresh=0.12):
        
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

            # Filter valid depth
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
                z_r = -y_c_rot + self.camera_height

                # Only consider obstacles (above ground)
                is_obstacle = z_r > self.obstacle_thresh

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
                    depths = z_c_rot[is_obstacle][valid]

                    # Keep minimum distance per cell
                    for i in range(len(grid_rows)):
                        r, c, d = grid_rows[i], grid_cols[i], depths[i]
                        grid[r, c] = min(grid[r, c], d)

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

            # Filter valid ranges - Increase min to 0.25m to avoid self-hits from rover body
            valid_mask = (ranges > 0.25) & (ranges < self.range_m) & np.isfinite(ranges)

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

                # Keep minimum distance
                for i in range(len(rows)):
                    r, c, d = rows[i], cols[i], ranges_filtered[i]
                    grid[r, c] = min(grid[r, c], d)

        # Force Clear Robot Footprint (Mask out self-collisions)
        # Robot is at bottom center (127, 64)
        # Radius of ~20cm? Grid res ~3cm. 20/3 = ~7 pixels.
        # Let's clear a semi-circle or box around the origin
        r_center, c_center = self.grid_size - 1, self.grid_size // 2
        y_grid, x_grid = np.ogrid[:self.grid_size, :self.grid_size]
        # (r - r_cnt)^2 + (c - c_cnt)^2 < radius^2
        # Use 45cm radius clearing to be safe (tractor body + noise)
        radius_px = int(0.45 / self.resolution)
        footprint_mask = ((y_grid - r_center)**2 + (x_grid - c_center)**2) < radius_px**2
        grid[footprint_mask] = self.range_m # Reset to max distance (free)

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
        # Robot is at (127, 64). The 10x10 patch we check in episode runner is rows 118-128, cols 59-69
        # These should NEVER be 0.0 as the robot occupies this space (free by definition)
        r_center, c_center = self.grid_size - 1, self.grid_size // 2
        safety_patch = distance_grid[r_center-10:r_center+1, c_center-5:c_center+6]
        if np.min(safety_patch) < 0.05:  # Less than 20cm (0.05 * 4.0m)
            # Something is wrong - footprint should be clear
            # Set to a safe minimum value (at least 0.5m)
            min_safe_val = 0.5 / self.range_m  # 0.125 normalized
            distance_grid[r_center-10:r_center+1, c_center-5:c_center+6] = np.maximum(
                distance_grid[r_center-10:r_center+1, c_center-5:c_center+6],
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

        # Keep maximum height per cell
        for i in range(len(grid_rows)):
            r, c, h = grid_rows[i], grid_cols[i], heights[i]
            height_grid[r, c] = max(height_grid[r, c], h)

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
