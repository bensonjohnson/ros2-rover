#!/usr/bin/env python3
"""URDF to Habitat configuration utility.

Extracts robot dimensions from XACRO/URDF and generates Habitat-compatible configs.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import math


def parse_xacro_properties(xacro_file: str) -> dict:
    """Parse XACRO properties from tractor.urdf.xacro.

    Returns:
        Dictionary of property names to values (in meters)
    """
    tree = ET.parse(xacro_file)
    root = tree.getroot()

    # Find all xacro:property elements
    properties = {}

    # XACRO namespace
    ns = {'xacro': 'http://www.ros.org/wiki/xacro'}

    for prop in root.findall('.//xacro:property', ns):
        name = prop.get('name')
        value = prop.get('value')

        # Try to evaluate simple expressions
        if value:
            try:
                # Handle PI constant
                value = value.replace('${PI}', str(math.pi))

                # Simple eval for math expressions
                if '/' in value or '*' in value or '+' in value or '-' in value:
                    properties[name] = eval(value, {"__builtins__": {}}, properties)
                else:
                    properties[name] = float(value)
            except:
                properties[name] = value

    return properties


def extract_robot_dimensions(xacro_file: str) -> dict:
    """Extract robot dimensions for Habitat configuration.

    Args:
        xacro_file: Path to tractor.urdf.xacro

    Returns:
        Dictionary with Habitat-compatible dimensions
    """
    props = parse_xacro_properties(xacro_file)

    # Calculate total dimensions
    # Robot footprint: base + tracks
    base_length = props.get('base_length', 0.2667)
    base_width = props.get('base_width', 0.102)
    base_height = props.get('base_height', 0.058)
    track_width = props.get('track_width', 0.042)
    track_height = props.get('track_height', 0.090)
    platform_offset_z = props.get('platform_offset_z', 0.05)
    platform_height = props.get('platform_height', 0.0025)

    # Total robot dimensions
    total_width = base_width + 2 * track_width  # 0.102 + 2*0.042 = 0.186m
    robot_radius = total_width / 2  # 0.093m
    robot_height = track_height  # Ground to top of tracks: 0.090m

    # Camera mounting
    # Camera is on platform_link at front
    # Platform height: base_height/2 + platform_offset_z + platform_height/2
    platform_z = base_height / 2 + platform_offset_z + platform_height / 2
    # Camera is 25mm box, centered on platform
    camera_height = platform_z + 0.025 / 2  # ~0.0925m

    # But wait - let me check the actual camera position from joint
    # camera_joint origin: xyz="${base_length/2 - 0.09/2} 0 ${platform_height/2 + 0.025/2}"
    # Relative to platform_link, which is already elevated
    # So actual camera Z from ground:
    camera_z_from_ground = track_height / 2 - 0.016 + base_height / 2 + platform_offset_z + platform_height / 2 + platform_height / 2 + 0.025 / 2
    # Simplify: track_height/2 is base ground level
    # Let me use the simpler calculation

    # From URDF:
    # base_footprint is at ground (0, 0, track_height/2 - 0.016)
    # base_link is +track_height/2 - 0.016 above ground
    # platform_link is +base_height/2 + platform_offset_z + platform_height/2 above base_link
    # camera is +platform_height/2 + 0.025/2 above platform_link

    total_camera_height = (track_height / 2 - 0.016) + (base_height / 2 + platform_offset_z + platform_height / 2) + (platform_height / 2 + 0.025 / 2)
    # = 0.045 - 0.016 + 0.029 + 0.05 + 0.00125 + 0.00125 + 0.0125
    # = 0.029 + 0.0925 = 0.1215m ≈ 122mm

    # Inertial properties
    mass = 5.0  # Base mass (chassis)
    track_mass = 2.0  # Per track
    total_mass = mass + 2 * track_mass  # 9kg

    # Wheel/track parameters
    wheel_radius = props.get('wheel_radius', 0.0485)
    wheel_separation = props.get('wheel_separation', 0.144)

    return {
        # Agent dimensions for Habitat
        'agent_radius': robot_radius,  # 0.093m (half of total width)
        'agent_height': robot_height,  # 0.090m (track height)

        # Camera sensor position (relative to agent center)
        'camera_height': total_camera_height,  # 0.122m from ground
        'camera_offset_forward': base_length / 2 - 0.09 / 2,  # Forward offset from center

        # Physical properties
        'total_mass': total_mass,  # 9kg
        'base_length': base_length,  # 0.267m
        'base_width': total_width,  # 0.186m
        'base_height': robot_height,  # 0.090m

        # Differential drive parameters
        'wheel_radius': wheel_radius,  # 0.0485m
        'wheel_separation': wheel_separation,  # 0.144m

        # Track properties (for friction modeling)
        'track_width': track_width,  # 0.042m per track
        'track_length': base_length,  # 0.267m

        # Raw properties (for reference)
        'xacro_properties': props,
    }


def generate_habitat_config(dimensions: dict, output_file: str = None):
    """Generate Habitat configuration YAML from robot dimensions.

    Args:
        dimensions: Robot dimensions from extract_robot_dimensions()
        output_file: Optional path to write YAML config
    """
    config_yaml = f"""# Habitat configuration generated from tractor.urdf.xacro
# Robot dimensions extracted from URDF

SIMULATOR:
  AGENT_0:
    HEIGHT: {dimensions['agent_height']:.4f}  # {dimensions['agent_height']*1000:.1f}mm (track height)
    RADIUS: {dimensions['agent_radius']:.4f}  # {dimensions['agent_radius']*1000:.1f}mm (half of total width including tracks)

  # RGB Camera (RealSense D435i)
  RGB_SENSOR:
    WIDTH: 640
    HEIGHT: 480
    HFOV: 69  # Horizontal FOV in degrees
    POSITION: [0, {dimensions['camera_height']:.4f}, 0]  # [x, y, z] in meters - camera height from ground
    ORIENTATION: [0, 0, 0]  # [roll, pitch, yaw]

  # Depth Camera (RealSense D435i)
  DEPTH_SENSOR:
    WIDTH: 640
    HEIGHT: 480
    HFOV: 69
    MIN_DEPTH: 0.0
    MAX_DEPTH: 10.0
    POSITION: [0, {dimensions['camera_height']:.4f}, 0]
    ORIENTATION: [0, 0, 0]

# Physical properties for reference
# Total mass: {dimensions['total_mass']:.1f} kg
# Base dimensions: {dimensions['base_length']*1000:.1f}mm × {dimensions['base_width']*1000:.1f}mm × {dimensions['base_height']*1000:.1f}mm
# Wheel separation: {dimensions['wheel_separation']*1000:.1f}mm
# Wheel radius: {dimensions['wheel_radius']*1000:.1f}mm
"""

    if output_file:
        with open(output_file, 'w') as f:
            f.write(config_yaml)
        print(f"✓ Habitat config written to: {output_file}")

    return config_yaml


def main():
    """Extract dimensions and generate config."""
    # Path to URDF
    repo_root = Path(__file__).parent.parent
    xacro_file = repo_root / 'src/tractor_bringup/urdf/tractor.urdf.xacro'

    if not xacro_file.exists():
        print(f"❌ URDF not found: {xacro_file}")
        return

    print(f"Reading URDF: {xacro_file}")
    print()

    # Extract dimensions
    dims = extract_robot_dimensions(str(xacro_file))

    # Print summary
    print("=" * 60)
    print("Robot Dimensions from URDF")
    print("=" * 60)
    print()
    print("Agent Configuration (for Habitat):")
    print(f"  Radius: {dims['agent_radius']:.4f} m ({dims['agent_radius']*1000:.1f} mm)")
    print(f"  Height: {dims['agent_height']:.4f} m ({dims['agent_height']*1000:.1f} mm)")
    print()
    print("Camera Configuration:")
    print(f"  Height from ground: {dims['camera_height']:.4f} m ({dims['camera_height']*1000:.1f} mm)")
    print(f"  Forward offset: {dims['camera_offset_forward']:.4f} m ({dims['camera_offset_forward']*1000:.1f} mm)")
    print()
    print("Physical Properties:")
    print(f"  Total mass: {dims['total_mass']:.1f} kg")
    print(f"  Base: {dims['base_length']*1000:.1f} × {dims['base_width']*1000:.1f} × {dims['base_height']*1000:.1f} mm")
    print(f"  Wheel separation: {dims['wheel_separation']*1000:.1f} mm")
    print(f"  Wheel radius: {dims['wheel_radius']*1000:.1f} mm")
    print()

    # Generate config
    output_file = Path(__file__).parent / 'habitat_config_from_urdf.yaml'
    config = generate_habitat_config(dims, str(output_file))

    print()
    print("=" * 60)
    print("Comparison with Current Habitat Config")
    print("=" * 60)
    print()
    print("Current (hardcoded)     →  URDF-based")
    print(f"Radius: 0.180m (180mm)  →  {dims['agent_radius']:.3f}m ({dims['agent_radius']*1000:.0f}mm)  ⚠ 2x too large!")
    print(f"Height: 0.880m (880mm)  →  {dims['agent_height']:.3f}m ({dims['agent_height']*1000:.0f}mm)  ⚠ 10x too large!")
    print(f"Camera: 0.880m          →  {dims['camera_height']:.3f}m ({dims['camera_height']*1000:.0f}mm)  ⚠ 7x too large!")
    print()
    print("⚠ WARNING: Current Habitat config does NOT match URDF!")
    print("  The hardcoded values are much larger than the actual robot.")
    print("  Update habitat_episode_runner.py to use URDF-based dimensions.")
    print()


if __name__ == '__main__':
    main()
