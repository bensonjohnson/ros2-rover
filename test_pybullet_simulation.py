#!/usr/bin/env python3
"""
Simple test script to verify PyBullet simulation functionality
"""

import sys
import os
import time

# Add the simulation module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'tractor_simulation', 'tractor_simulation'))

def test_pybullet_import():
    """Test PyBullet import"""
    try:
        import pybullet as p
        import pybullet_data
        print("✓ PyBullet imported successfully")
        print(f"  Version: {p.getAPIVersion()}")
        return True
    except Exception as e:
        print(f"✗ Failed to import PyBullet: {e}")
        return False

def test_simulation_class():
    """Test the TractorSimulation class"""
    try:
        from bullet_simulation import TractorSimulation
        print("✓ TractorSimulation class imported successfully")
        
        # Test creating simulation instance (headless)
        print("  Creating headless simulation...")
        sim = TractorSimulation(use_gui=False, enable_visualization=False)
        print("  ✓ Headless simulation created")
        
        # Test adding environment
        print("  Adding indoor environment...")
        sim.add_indoor_environment()
        print("  ✓ Indoor environment added")
        
        # Test a few simulation steps
        print("  Running simulation steps...")
        for i in range(10):
            state = sim.step()
            if i % 5 == 0:
                print(f"    Step {i}: Position = {state['robot_state']['position'][:2]}")
        
        # Test getting depth image
        depth_image = sim.get_depth_image()
        print(f"  ✓ Depth image shape: {depth_image.shape}")
        
        # Test robot state
        robot_state = sim.get_robot_state()
        print(f"  ✓ Robot position: {robot_state['position'][:2]}")
        print(f"  ✓ Robot yaw: {robot_state['yaw']:.2f}")
        
        # Clean up
        sim.close()
        print("  ✓ Simulation cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test TractorSimulation class: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simulation_with_gui():
    """Test simulation with GUI (if possible)"""
    try:
        from bullet_simulation import TractorSimulation
        print("✓ Testing GUI simulation...")
        
        # This will show a window - run for a few seconds then close
        print("  Creating GUI simulation (will close automatically in 5 seconds)...")
        sim = TractorSimulation(use_gui=True, enable_visualization=True)
        sim.add_indoor_environment()
        
        # Run for 5 seconds
        start_time = time.time()
        step_count = 0
        while time.time() - start_time < 5:
            sim.set_velocity(0.2, 0.1)  # Move forward and turn
            state = sim.step()
            step_count += 1
            time.sleep(0.01)  # Small delay for visualization
            
        print(f"  ✓ Ran {step_count} steps in GUI mode")
        sim.close()
        
        return True
        
    except Exception as e:
        print(f"⚠ GUI test failed (this is OK if running headless): {e}")
        return True  # Don't fail the overall test for GUI issues

def main():
    """Main test function"""
    print("🧪 Testing PyBullet Simulation Components")
    print("=" * 50)
    
    tests = [
        ("PyBullet Import", test_pybullet_import),
        ("Simulation Class", test_simulation_class),
        ("GUI Simulation", test_simulation_with_gui)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            if test_func():
                passed += 1
                print(f"  ✓ {test_name} passed")
            else:
                print(f"  ✗ {test_name} failed")
        except Exception as e:
            print(f"  ✗ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All PyBullet tests passed!")
        print("\nNext steps:")
        print("  1. Run './setup_simulation.sh' to build the full Docker environment")
        print("  2. Run './run_simulation.sh' to start ES training")
    else:
        print("⚠ Some tests failed. Check the output above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
