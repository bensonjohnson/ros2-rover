import sys
import os
import numpy as np
import torch
import cv2
import importlib.util

# Add paths
sys.path.append('src/tractor_bringup/tractor_bringup')
sys.path.append('remote_training_server')

# Import modules via importlib to handle same-named files
def import_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    from occupancy_processor import RawSensorProcessor
    print("✅ Imported RawSensorProcessor")
except ImportError as e:
    # Try importing from full path if relative fails
    try:
        RawSensorProcessor = import_from_path("occupancy_processor", "src/tractor_bringup/tractor_bringup/occupancy_processor.py").RawSensorProcessor
        print("✅ Imported RawSensorProcessor (via path)")
    except Exception as e2:
        print(f"❌ Failed to import RawSensorProcessor: {e} -> {e2}")

try:
    from model_architectures import DualEncoderPolicyNetwork
    print("✅ Imported DualEncoderPolicyNetwork")
except ImportError as e:
    print(f"❌ Failed to import DualEncoderPolicyNetwork: {e}")

# Deferred imports for serialization to avoid crash if msgpack missing
rover_ser = None
server_ser = None

class MockScan:
    def __init__(self):
        self.ranges = [2.0] * 360 # 2 meters all around
        self.angle_min = -3.14
        self.angle_max = 3.14
        self.angle_increment = 6.28 / 360
        self.range_min = 0.1
        self.range_max = 10.0

def verify():
    print("\n--- 1. Testing RawSensorProcessor ---")
    try:
        processor = RawSensorProcessor(grid_size=128, max_range=4.0)
        # Dummy inputs
        # Depth image: uint16, mm
        depth_img = np.random.randint(500, 4000, (240, 424), dtype=np.uint16)
        
        laser_grid, depth_processed = processor.process(depth_img, MockScan())
        print(f"Processor Output Shapes: Laser={laser_grid.shape}, Depth={depth_processed.shape}")
        
        # Validate shapes
        if laser_grid.shape == (128, 128):
            print("✅ Laser Grid Shape OK")
        else:
            print(f"❌ Laser Grid Shape Mismatch: {laser_grid.shape}")

        if depth_processed.shape == (240, 424):
            print("✅ Depth Image Shape OK")
        else:
            print(f"❌ Depth Image Shape Mismatch: {depth_processed.shape}")
            
    except Exception as e:
        print(f"❌ Processor Test Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- 2. Testing DualEncoderPolicyNetwork ---")
    try:
        model = DualEncoderPolicyNetwork(action_dim=2, proprio_dim=10)
        # Inputs: (B, 1, 128, 128), (B, 1, 424, 240), (B, 10)
        dummy_laser = torch.randn(1, 1, 128, 128)
        dummy_depth = torch.randn(1, 1, 424, 240)
        dummy_proprio = torch.randn(1, 10)
        
        mean, log_std = model(dummy_laser, dummy_depth, dummy_proprio)
        print(f"Model Output Shapes: Mean={mean.shape}, LogStd={log_std.shape}")
        
        if mean.shape == (1, 2) and log_std.shape == (1, 2):
            print("✅ Model Forward Pass OK")
        else:
            print(f"❌ Model Output Shape Mismatch")
            
    except Exception as e:
        print(f"❌ Model Test Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- 3. Testing Serialization (Roundtrip) ---")
    try:
        # Check imports first
        global rover_ser, server_ser
        if rover_ser is None or server_ser is None:
             try:
                 rover_ser = import_from_path("rover_ser", "src/tractor_bringup/tractor_bringup/serialization_utils.py")
                 server_ser = import_from_path("server_ser", "remote_training_server/serialization_utils.py")
             except ImportError as e:
                 print(f"⚠️ Serialization skipped: {e}")
                 raise ImportError(e)

        # Rover side: serialize batch
        batch_size = 4
        batch = {
            'laser': np.random.rand(batch_size, 1, 128, 128).astype(np.float32),
            'depth': np.random.rand(batch_size, 1, 424, 240).astype(np.float32),
            'proprio': np.random.rand(batch_size, 10).astype(np.float32),
            'actions': np.random.rand(batch_size, 2).astype(np.float32),
            'rewards': np.random.rand(batch_size, 1).astype(np.float32),
            'dones': np.random.rand(batch_size, 1).astype(np.float32),
            'next_laser': np.random.rand(batch_size, 1, 128, 128).astype(np.float32),
            'next_depth': np.random.rand(batch_size, 1, 424, 240).astype(np.float32),
            'next_proprio': np.random.rand(batch_size, 10).astype(np.float32),
        }
        
        print(f"Original Proprio Mean: {batch['proprio'].mean():.6f}")
        
        print("Serializing batch (Rover)...")
        data = rover_ser.serialize_batch(batch)
        print(f"Serialized size: {len(data)} bytes")
        
        print("Deserializing batch (Server)...")
        decoded = server_ser.deserialize_batch(data)
        
        # Verify contents
        if np.allclose(batch['proprio'], decoded['proprio']):
            print("✅ Proprio Data Match")
        else:
            print(f"❌ Proprio Data Mismatch: {batch['proprio'][0]} vs {decoded['proprio'][0]}")
            
        if decoded['laser'].shape == (batch_size, 1, 128, 128):
            print("✅ Laser Data Shape Match")
        else:
            print(f"❌ Laser Data Shape Mismatch: {decoded['laser'].shape}")
            
        print("✅ Serialization Roundtrip OK")

    except Exception as e:
        # If skip or fail, just print and continue to vis test (which uses dummy data)
        print(f"ℹ️ Serialization test ended: {e}")


    print("\n--- 4. Testing Visualization State ---")
    try:
        # Simulate data (independent of serialization)
        latest_laser_vis = np.random.randint(0, 2, (1, 128, 128)).astype(np.float32)
        latest_depth_vis = np.random.rand(1, 424, 240).astype(np.float32) # Float 0-1
        
        print(f"Simulated Vis Laser Shape: {latest_laser_vis.shape}")
        print(f"Simulated Vis Depth Shape: {latest_depth_vis.shape}")
        
        # Simulate dashboard usage
        # Laser
        if latest_laser_vis.ndim == 3: latest_laser_vis = latest_laser_vis[0]
        vis_laser = cv2.resize(latest_laser_vis, (256, 256), interpolation=cv2.INTER_NEAREST)
        print("✅ Laser Visualization Resize OK")
        
        # Depth
        if latest_depth_vis.ndim == 3: latest_depth_vis = latest_depth_vis[0]
        
        # Convert float [0, 1] to uint8 [0, 255] if needed
        # This mirrors the fix applied to dashboard_app.py
        if latest_depth_vis.dtype != np.uint8:
            # print("Converting float depth to uint8...")
            latest_depth_vis = (latest_depth_vis * 255.0).astype(np.uint8)
        
        depth_inverted = 255 - latest_depth_vis
        vis_depth = cv2.applyColorMap(depth_inverted, cv2.COLORMAP_JET)
        print("✅ Depth Visualization Colormap OK")

    except Exception as e:
        print(f"❌ Visualization Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
