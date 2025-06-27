#!/usr/bin/env python3
"""
Example usage of the tractor coverage planning system.
Demonstrates different coverage patterns for mowing and spraying.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point32, Polygon
from tractor_coverage.coverage_planner import CoveragePlanner
from tractor_coverage.coverage_client import CoverageClient
import math
import time


def create_test_areas():
    """Create various test areas for coverage planning"""
    
    areas = {}
    
    # 1. Simple rectangular lawn
    rect_boundary = Polygon()
    rect_points = [
        (0, 0), (20, 0), (20, 15), (0, 15), (0, 0)  # 20x15m rectangle
    ]
    for x, y in rect_points:
        point = Point32()
        point.x = float(x)
        point.y = float(y)
        point.z = 0.0
        rect_boundary.points.append(point)
    areas['rectangular_lawn'] = rect_boundary
    
    # 2. L-shaped yard
    l_boundary = Polygon()
    l_points = [
        (0, 0), (15, 0), (15, 8), (8, 8), (8, 12), (0, 12), (0, 0)
    ]
    for x, y in l_points:
        point = Point32()
        point.x = float(x)
        point.y = float(y)
        point.z = 0.0
        l_boundary.points.append(point)
    areas['l_shaped_yard'] = l_boundary
    
    # 3. Circular field
    circular_boundary = Polygon()
    center_x, center_y, radius = 0, 0, 10
    num_points = 24
    for i in range(num_points + 1):
        angle = 2 * math.pi * i / num_points
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        point = Point32()
        point.x = float(x)
        point.y = float(y)
        point.z = 0.0
        circular_boundary.points.append(point)
    areas['circular_field'] = circular_boundary
    
    # 4. Complex polygon with obstacles
    complex_boundary = Polygon()
    complex_points = [
        (0, 0), (25, 0), (25, 20), (20, 25), (5, 25), (0, 20), (0, 0)
    ]
    for x, y in complex_points:
        point = Point32()
        point.x = float(x)
        point.y = float(y)
        point.z = 0.0
        complex_boundary.points.append(point)
    areas['complex_field'] = complex_boundary
    
    return areas


def create_obstacles():
    """Create obstacle examples"""
    obstacles = {}
    
    # Tree cluster
    tree_obstacle = []
    tree_points = [
        (8, 6), (10, 6), (10, 8), (8, 8), (8, 6)  # 2x2m square obstacle
    ]
    for x, y in tree_points:
        point = Point32()
        point.x = float(x)
        point.y = float(y)
        point.z = 0.0
        tree_obstacle.append(point)
    obstacles['tree_cluster'] = tree_obstacle
    
    # Building
    building_obstacle = []
    building_points = [
        (12, 10), (18, 10), (18, 16), (12, 16), (12, 10)  # 6x6m building
    ]
    for x, y in building_points:
        point = Point32()
        point.x = float(x)
        point.y = float(y)
        point.z = 0.0
        building_obstacle.append(point)
    obstacles['building'] = building_obstacle
    
    return obstacles


def demonstrate_coverage_planning():
    """Demonstrate coverage planning algorithms"""
    
    print("=== Coverage Planning Demonstration ===\n")
    
    # Create planner
    planner = CoveragePlanner()
    
    # Get test areas
    areas = create_test_areas()
    obstacles = create_obstacles()
    
    # Test different configurations
    test_configs = [
        {
            'name': 'Lawn Mowing - Standard',
            'area': 'rectangular_lawn',
            'tool_width': 1.0,
            'overlap': 0.1,
            'obstacles': []
        },
        {
            'name': 'Lawn Mowing - Wide Cut',
            'area': 'rectangular_lawn', 
            'tool_width': 1.5,
            'overlap': 0.15,
            'obstacles': []
        },
        {
            'name': 'Spraying - Circular Field',
            'area': 'circular_field',
            'tool_width': 2.0,
            'overlap': 0.05,
            'obstacles': []
        },
        {
            'name': 'Complex Field with Obstacles',
            'area': 'complex_field',
            'tool_width': 1.2,
            'overlap': 0.1,
            'obstacles': ['tree_cluster', 'building']
        }
    ]
    
    for config in test_configs:
        print(f"Testing: {config['name']}")
        print("-" * 40)
        
        # Setup planner
        planner.set_tool_parameters(config['tool_width'], config['overlap'])
        
        # Get boundary
        boundary_polygon = areas[config['area']]
        boundary_points = [Point(x=p.x, y=p.y, z=0.0) for p in boundary_polygon.points]
        
        # Get obstacles
        obstacle_list = []
        for obs_name in config['obstacles']:
            if obs_name in obstacles:
                obs_points = [Point(x=p.x, y=p.y, z=0.0) for p in obstacles[obs_name]]
                obstacle_list.append(obs_points)
        
        # Plan coverage
        start_time = time.time()
        coverage_path = planner.plan_coverage_path(boundary_points, obstacle_list)
        planning_time = time.time() - start_time
        
        # Calculate statistics
        area = planner.calculate_coverage_area(boundary_points, obstacle_list)
        path_length = 0.0
        if len(coverage_path) > 1:
            for i in range(1, len(coverage_path)):
                dx = coverage_path[i].x - coverage_path[i-1].x
                dy = coverage_path[i].y - coverage_path[i-1].y
                path_length += (dx*dx + dy*dy)**0.5
        
        estimated_time = planner.estimate_coverage_time(path_length, work_speed=1.0)
        
        # Print results
        print(f"  Boundary area: {area:.1f} m²")
        print(f"  Tool width: {config['tool_width']:.1f} m")
        print(f"  Overlap: {config['overlap']*100:.1f}%")
        print(f"  Waypoints generated: {len(coverage_path)}")
        print(f"  Total path length: {path_length:.1f} m")
        print(f"  Planning time: {planning_time:.3f} seconds")
        print(f"  Estimated work time: {estimated_time/60:.1f} minutes")
        print(f"  Efficiency: {area/path_length*100:.1f}% (area/distance)")
        print()


def demonstrate_ros_integration():
    """Demonstrate ROS integration with coverage client"""
    
    print("=== ROS Integration Demonstration ===\n")
    
    rclpy.init()
    
    try:
        # Create coverage client
        client = CoverageClient()
        
        print("Creating test coverage operations...")
        
        # Example 1: Rectangular mowing
        print("\n1. Rectangular Mowing Operation:")
        rect_boundary = client.create_rectangular_boundary(10, 5, 20, 12)
        print(f"   Boundary: 20x12m rectangle at (10,5)")
        print(f"   Tool width: 1.0m, Overlap: 10%")
        
        # Don't actually start - just demonstrate the setup
        print("   Command: client.start_mowing_operation(boundary, tool_width=1.0, overlap=0.1)")
        
        # Example 2: Circular spraying
        print("\n2. Circular Spraying Operation:")
        circle_boundary = client.create_circular_boundary(0, 0, 8)
        print(f"   Boundary: Circle with 8m radius at origin")
        print(f"   Tool width: 1.5m, Overlap: 5%")
        print("   Command: client.start_spraying_operation(boundary, tool_width=1.5, overlap=0.05)")
        
        print("\nTo actually execute operations:")
        print("1. Launch the coverage system: ros2 launch tractor_coverage coverage_system.launch.py")
        print("2. Use the client in interactive mode: ros2 run tractor_coverage coverage_client")
        print("3. Or create custom scripts using the CoverageClient class")
        
    finally:
        rclpy.shutdown()


def print_usage_instructions():
    """Print comprehensive usage instructions"""
    
    print("\n" + "="*60)
    print("TRACTOR COVERAGE SYSTEM USAGE GUIDE")
    print("="*60)
    
    print("\n1. SYSTEM STARTUP:")
    print("   # Launch complete tractor system")
    print("   ros2 launch tractor_bringup tractor_bringup.launch.py")
    print("   ")
    print("   # Launch navigation (in separate terminal)")
    print("   ros2 launch tractor_bringup navigation.launch.py")
    print("   ")
    print("   # Launch coverage system")
    print("   ros2 launch tractor_coverage coverage_system.launch.py")
    
    print("\n2. DEFINE COVERAGE AREA:")
    print("   Method A - Interactive Client:")
    print("   ros2 run tractor_coverage coverage_client")
    print("   ")
    print("   Method B - Custom Script:")
    print("   # Create boundary polygon")
    print("   # Use CoverageClient.start_mowing_operation() or start_spraying_operation()")
    
    print("\n3. MONITOR EXECUTION:")
    print("   # View in RViz (coverage markers and path)")
    print("   rviz2")
    print("   ")
    print("   # Or use built-in visualizer")
    print("   ros2 run tractor_coverage coverage_visualizer")
    
    print("\n4. OPERATION TYPES:")
    print("   MOWING:")
    print("   - Tool width: 0.8-1.5m (typical)")
    print("   - Overlap: 10-15% (for good coverage)")
    print("   - Speed: 0.3-0.8 m/s")
    print("   - Includes perimeter pass")
    print("   ")
    print("   SPRAYING:")
    print("   - Tool width: 1.5-3.0m (boom width)")
    print("   - Overlap: 5-10% (to avoid waste)")
    print("   - Speed: 0.5-1.2 m/s")
    print("   - Variable rate application based on speed")
    
    print("\n5. SAFETY FEATURES:")
    print("   - Automatic boundary validation")
    print("   - Obstacle avoidance integration")
    print("   - Emergency stop capability")
    print("   - Implement safety interlocks")
    print("   - GPS fence protection")
    
    print("\n6. ADVANCED FEATURES:")
    print("   - Path optimization for efficiency")
    print("   - Resume capability after interruption")
    print("   - Coverage progress tracking")
    print("   - Integration with implement controllers")
    print("   - Adaptive speed control")


if __name__ == "__main__":
    print("Tractor Coverage Planning System Examples")
    print("========================================\n")
    
    # Run demonstrations
    demonstrate_coverage_planning()
    demonstrate_ros_integration()
    print_usage_instructions()
    
    print("\nDemo complete! Check the generated coverage paths and try the system.")
    print("For interactive testing, run: ros2 run tractor_coverage coverage_client")