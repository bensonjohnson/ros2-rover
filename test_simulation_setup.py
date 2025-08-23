#!/usr/bin/env python3
"""
Test script to verify the simulation setup and components
"""

import sys
import os
import subprocess
import json

def test_python_imports():
    """Test that all required Python modules can be imported"""
    print("Testing Python imports...")
    
    modules_to_test = [
        "pybullet",
        "numpy",
        "torch",
        "cv2",
        "rclpy",
        "es_trainer_depth",
        "bullet_simulation"
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            if module == "es_trainer_depth":
                # Special handling for our custom module
                sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'tractor_bringup', 'tractor_bringup'))
                import es_trainer_depth
                print(f"  ✓ {module}")
            elif module == "bullet_simulation":
                # Special handling for our custom module
                sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'tractor_simulation', 'tractor_simulation'))
                import bullet_simulation
                print(f"  ✓ {module}")
            else:
                __import__(module)
                print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module} - {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_docker_setup():
    """Test Docker installation and permissions"""
    print("\nTesting Docker setup...")
    
    # Check if Docker is installed
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"  ✓ Docker installed: {result.stdout.strip()}")
        else:
            print(f"  ✗ Docker not installed or not working: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ✗ Docker not found: {e}")
        return False
    
    # Check if Docker Compose is installed
    try:
        result = subprocess.run(["docker-compose", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"  ✓ Docker Compose installed: {result.stdout.strip()}")
        else:
            print(f"  ⚠ Docker Compose not installed: {result.stderr}")
    except Exception as e:
        print(f"  ⚠ Docker Compose not found: {e}")
    
    # Check if user is in docker group
    try:
        result = subprocess.run(["groups"], capture_output=True, text=True, timeout=10)
        if "docker" in result.stdout:
            print("  ✓ User is in docker group")
        else:
            print("  ⚠ User is not in docker group (may need sudo for Docker commands)")
    except Exception as e:
        print(f"  ⚠ Could not check user groups: {e}")
    
    return True

def test_rocm_support():
    """Test ROCm support and GPU availability"""
    print("\nTesting ROCm support...")
    
    # Check if ROCm tools are available
    try:
        result = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("  ✓ ROCm tools available")
            # Show first few lines of rocminfo
            lines = result.stdout.split('\n')[:10]
            for line in lines:
                if line.strip():
                    print(f"    {line}")
        else:
            print(f"  ⚠ ROCm tools not available: {result.stderr}")
    except Exception as e:
        print(f"  ⚠ ROCm tools not found: {e}")
    
    # Check PyTorch ROCm support
    try:
        import torch
        print(f"  ✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print("  ✓ CUDA available (for NVIDIA GPUs)")
        else:
            print("  ⚠ CUDA not available")
            
        if hasattr(torch.version, 'hip') and torch.version.hip:
            print(f"  ✓ ROCm available: {torch.version.hip}")
        else:
            print("  ⚠ ROCm not available in PyTorch")
            
    except ImportError:
        print("  ✗ PyTorch not available")
        return False
    
    return True

def test_file_structure():
    """Test that required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        "src/tractor_simulation/tractor_simulation/bullet_simulation.py",
        "src/tractor_simulation/tractor_simulation/es_simulation_trainer.py",
        "Dockerfile.pytorch-rocm",
        "docker-compose.simulation.yml",
        "setup_simulation.sh",
        "run_simulation.sh",
        "dashboard/index.html",
        "README_SIMULATION.md"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def test_directory_structure():
    """Test that required directories exist and are writable"""
    print("\nTesting directory structure...")
    
    required_dirs = [
        "models",
        "logs",
        "sim_data",
        "dashboard",
        "src/tractor_simulation/tractor_simulation"
    ]
    
    problematic_dirs = []
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            if os.access(dir_path, os.W_OK):
                print(f"  ✓ {dir_path} (writable)")
            else:
                print(f"  ⚠ {dir_path} (not writable)")
                problematic_dirs.append(dir_path)
        else:
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"  ✓ {dir_path} (created)")
            except Exception as e:
                print(f"  ✗ {dir_path} (could not create: {e})")
                problematic_dirs.append(dir_path)
    
    return len(problematic_dirs) == 0

def main():
    """Main test function"""
    print("🚜 Testing Tractor ES Simulation Setup")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run all tests
    tests = [
        ("File Structure", test_file_structure),
        ("Directory Structure", test_directory_structure),
        ("Python Imports", test_python_imports),
        ("Docker Setup", test_docker_setup),
        ("ROCm Support", test_rocm_support)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            result = test_func()
            if not result:
                all_tests_passed = False
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! Your simulation setup is ready.")
        print("\nNext steps:")
        print("  1. Run './setup_simulation.sh' to build Docker images")
        print("  2. Run './run_simulation.sh' to start training")
        print("  3. Check the dashboard at http://localhost:3000")
    else:
        print("⚠ Some tests failed. Please check the output above and fix issues.")
        print("\nCommon fixes:")
        print("  - Install missing Python packages")
        print("  - Add user to docker group: sudo usermod -aG docker $USER")
        print("  - Install ROCm drivers and tools")
        print("  - Check file permissions")
    
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    sys.exit(main())
