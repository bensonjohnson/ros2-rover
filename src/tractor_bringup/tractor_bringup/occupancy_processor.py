import numpy as np
import cv2

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
                 camera_height=0.123, # meters (calculated from URDF)
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
