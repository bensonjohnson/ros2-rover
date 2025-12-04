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
        
        # Pre-compute rotation matrix for camera tilt
        # Camera frame: Z forward, X right, Y down
        # Rover frame: X forward, Y left, Z up
        
        # Rotation: Pitch down by 'tilt'
        # R_pitch = [
        #   [1, 0, 0],
        #   [0, cos, -sin],
        #   [0, sin, cos]
        # ]
        
        # Coordinate transform (Camera -> Rover):
        # Rover X = Camera Z
        # Rover Y = -Camera X
        # Rover Z = -Camera Y + height
        
        # Combined transform matrix (3x3) applied to (X_c, Y_c, Z_c)
        # We want (X_r, Y_r, Z_r)
        
        # Let's do it explicitly in process() for clarity first, then optimize if needed.
        # Actually, pre-computing the rotation is better.
        
        c = np.cos(self.camera_tilt)
        s = np.sin(self.camera_tilt)
        
        # Camera to Rover rotation
        # If camera is untilted:
        # X_r = Z_c
        # Y_r = -X_c
        # Z_r = -Y_c + h
        
        # With tilt (pitch down):
        # Camera Y' = Y*c - Z*s
        # Camera Z' = Y*s + Z*c
        # Then map to rover
        
        pass

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
        # Z_c = depth
        # X_c = Z_c * ((u - cx) / fx)
        # Y_c = Z_c * ((v - cy) / fy)
        
        # Vectorized
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
        # Camera Frame: X right, Y down, Z forward
        # Rover Frame: X forward, Y left, Z up
        
        # Apply Tilt (Rotation around X axis of camera)
        # Y_c' = Y_c * cos(t) - Z_c * sin(t)
        # Z_c' = Y_c * sin(t) + Z_c * cos(t)
        
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
        # Floor: |z_r| < floor_thresh
        # Obstacle: z_r > obstacle_thresh
        
        is_floor = np.abs(z_r) < self.floor_thresh
        is_obstacle = z_r > self.obstacle_thresh
        
        # 4. Project to Grid
        # Grid X: 0 to grid_range (mapped to 0..64)
        # Grid Y: -grid_range/2 to grid_range/2 (mapped to 0..64)
        
        # Scale to grid coordinates
        # X: [0, range] -> [63, 0] (Top-down view: Robot at bottom center)
        # Actually, let's put robot at bottom center (row 63, col 32)
        # X (forward) maps to rows (decreasing)
        # Y (left) maps to cols (increasing)
        
        # x_grid = grid_size - (x_r / range * grid_size)
        # y_grid = (y_r + range/2) / range * grid_size
        
        scale = self.grid_size / self.grid_range
        
        # Robot is at (0,0) in rover frame.
        # In grid image:
        # Row 0 is top (max X)
        # Row 63 is bottom (min X, near robot)
        # Col 0 is left (max Y)
        # Col 63 is right (min Y)
        
        # Wait, Y_r is left positive.
        # Image X is column (left-right).
        # Image Y is row (top-down).
        
        # Map X_r (Forward) to Image Rows (Bottom to Top)
        # row = H - 1 - (x_r * scale)
        grid_rows = self.grid_size - 1 - (x_r * scale).astype(np.int32)
        
        # Map Y_r (Left) to Image Cols (Center to Left/Right)
        # Y_r positive = LEFT in rover frame
        # Col 0 = LEFT in image, Col 63 = RIGHT in image
        # If Y_r = +1m (Left), Col should be low (e.g. 0)
        # If Y_r = -1m (Right), Col should be high (e.g. 63)
        # Therefore: Col = Center - Y_r * scale
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
        
        # Optional: Morphological operations to clean up
        # kernel = np.ones((3,3), np.uint8)
        # grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)
        
        return grid
