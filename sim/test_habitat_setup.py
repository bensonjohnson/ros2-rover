#!/usr/bin/env python3
"""Test script to verify Habitat integration is working."""

import sys
from pathlib import Path

print("=" * 60)
print("Testing Habitat Setup for ROS2 Rover")
print("=" * 60)
print()

# Test 1: Import dependencies
print("1. Testing imports...")
try:
    import habitat
    import habitat_sim
    import torch
    import zmq
    import zstandard
    print("   ✓ All required packages imported successfully")
    print(f"   - habitat-lab: {habitat.__version__}")
    print(f"   - habitat-sim: {habitat_sim.__version__}")
    print(f"   - PyTorch: {torch.__version__}")
    print(f"   - PyZMQ: {zmq.zmq_version()}")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

print()

# Test 2: Check GPU
print("2. Checking GPU availability...")
if torch.cuda.is_available():
    print(f"   ✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
else:
    print("   ⚠ No GPU detected - will use CPU (slower)")

print()

# Test 3: Load Habitat config
print("3. Testing Habitat config loading...")
try:
    habitat_lab_path = Path(habitat.__file__).parent
    config_file = "benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    configs_dir = str(habitat_lab_path / "config")
    config_full_path = habitat_lab_path / "config" / config_file

    if not config_full_path.exists():
        print(f"   ❌ Config file not found: {config_full_path}")
        sys.exit(1)

    config = habitat.get_config(
        config_path=config_file,
        configs_dir=configs_dir
    )
    print(f"   ✓ Config loaded from: {config_file}")
except Exception as e:
    print(f"   ❌ Failed to load config: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Check test scenes
print("4. Checking test scenes...")
test_scenes_path = Path("/home/benson/habitat-lab/data/versioned_data/habitat_test_scenes")
if test_scenes_path.exists():
    scene_files = list(test_scenes_path.glob("*.glb"))
    print(f"   ✓ Found {len(scene_files)} test scenes:")
    for scene in scene_files:
        size_mb = scene.stat().st_size / (1024 * 1024)
        print(f"     - {scene.name} ({size_mb:.1f} MB)")
else:
    print(f"   ⚠ Test scenes not found at: {test_scenes_path}")
    print("     Run: python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/")

print()

# Test 5: Check model architecture imports
print("5. Testing model architecture imports...")
try:
    sys.path.append(str(Path(__file__).parent.parent / 'remote_training_server'))
    from model_architectures import RGBDEncoder, PolicyHead
    print("   ✓ Model architectures imported successfully")
except ImportError as e:
    print(f"   ⚠ Model architectures not found: {e}")
    print("     This is OK if you're just testing Habitat")

print()
print("=" * 60)
print("✅ Habitat setup test complete!")
print("=" * 60)
print()
print("Next steps:")
print("1. Make sure the V620 training server is running")
print("2. Run: ./start_habitat_training.sh")
print()
