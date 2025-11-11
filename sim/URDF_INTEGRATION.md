# URDF Integration with Habitat Simulation

## Overview

The Habitat simulation now **automatically extracts robot dimensions from the URDF** to ensure accurate physics and sensor matching. This eliminates manual configuration errors and keeps the simulation synchronized with the physical robot definition.

## Robot Dimensions (from URDF)

From `src/tractor_bringup/urdf/tractor.urdf.xacro`:

| Property | Value | Notes |
|----------|-------|-------|
| **Agent Radius** | **93mm** | Half of total width (102mm base + 2×42mm tracks) |
| **Agent Height** | **90mm** | Track height (ground clearance) |
| **Camera Height** | **123mm** | RealSense D435i mounting height from ground |
| **Base Length** | **267mm** | Chassis length |
| **Base Width** | **186mm** | Total width including tracks |
| **Wheel Separation** | **144mm** | Center-to-center distance |
| **Wheel Radius** | **48.5mm** | Effective wheel radius |
| **Total Mass** | **9.0kg** | 5kg chassis + 2×2kg tracks |

## How It Works

### 1. URDF Parsing

`urdf_to_habitat.py` parses the XACRO file and extracts dimensions:

```python
from urdf_to_habitat import extract_robot_dimensions

dims = extract_robot_dimensions('src/tractor_bringup/urdf/tractor.urdf.xacro')

# Returns:
{
    'agent_radius': 0.093,      # 93mm
    'agent_height': 0.090,      # 90mm
    'camera_height': 0.123,     # 123mm
    'total_mass': 9.0,          # kg
    # ... more properties
}
```

### 2. Automatic Configuration

`habitat_episode_runner.py` automatically loads URDF dimensions on startup:

```python
runner = HabitatEpisodeRunner(
    use_urdf_dimensions=True  # Default: automatically use URDF
)

# Output:
# ✓ Loaded URDF dimensions:
#   Agent radius: 93mm
#   Agent height: 90mm
#   Camera height: 123mm
```

### 3. Habitat Environment Setup

Dimensions are applied to the Habitat environment:

```python
# Agent collision geometry
config.SIMULATOR.AGENT_0.HEIGHT = 0.090  # From URDF
config.SIMULATOR.AGENT_0.RADIUS = 0.093  # From URDF

# Camera sensors
config.SIMULATOR.RGB_SENSOR.POSITION = [0, 0.123, 0]   # From URDF
config.SIMULATOR.DEPTH_SENSOR.POSITION = [0, 0.123, 0] # From URDF
```

## Why This Matters

### Before (Hardcoded - WRONG!)

```python
# Old habitat_episode_runner.py (INCORRECT)
agent_radius = 0.18   # 180mm - 2x too large!
agent_height = 0.88   # 880mm - 10x too large!
camera_height = 0.88  # 880mm - 7x too large!
```

**Problems:**
- ❌ Agent was **human-sized** instead of RC-tank-sized
- ❌ Collision radius 2x too large → unrealistic obstacle avoidance
- ❌ Camera 7x too high → completely different perspective
- ❌ Manually maintained (prone to drift from URDF)

### After (URDF-based - CORRECT!)

```python
# New habitat_episode_runner.py (CORRECT)
# Automatically loaded from URDF
agent_radius = 0.093   # 93mm - matches physical tank
agent_height = 0.090   # 90mm - matches physical tank
camera_height = 0.123  # 123mm - matches RealSense mount
```

**Benefits:**
- ✅ **Accurate simulation** of small RC tank
- ✅ **Realistic collisions** with proper radius
- ✅ **Correct camera viewpoint** matching hardware
- ✅ **Single source of truth** (URDF)
- ✅ **Automatic updates** when URDF changes

## Impact on Training

### Collision Detection

With correct radius (93mm vs 180mm):
- **More realistic gap navigation** - can fit through tighter spaces
- **Better sim-to-real transfer** - obstacle distances match
- **Accurate collision penalty** - fitness reflects true safety margins

### Camera Perspective

With correct height (123mm vs 880mm):
- **Ground-level view** instead of overhead view
- **More realistic depth sensing** - objects appear larger, closer
- **Better sim-to-real transfer** - visual features match

### Scene Scale

The robot is **very small** compared to typical Habitat scenes:
- Gibson/Matterport3D scenes are human-scale (~2m doorways)
- Tank is only 90mm tall (3.6 inches)
- **This is intentional** - simulates navigating large environments

To the tank, a typical room looks like:
- Standard doorway (0.9m): **10× robot width**
- Table height (0.75m): **8× robot height**
- Room (4m × 4m): **43× robot length**

This creates realistic navigation challenges for the small rover.

## Manual Configuration (Optional)

To disable URDF loading and use custom dimensions:

```python
runner = HabitatEpisodeRunner(
    use_urdf_dimensions=False,  # Disable URDF
    # Will use fallback defaults (still URDF-based, but not dynamic)
)
```

Or modify `_create_habitat_env()` to hardcode custom values.

## Viewing URDF Dimensions

Run the utility script to see extracted dimensions:

```bash
cd sim
python3 urdf_to_habitat.py
```

Output:
```
============================================================
Robot Dimensions from URDF
============================================================

Agent Configuration (for Habitat):
  Radius: 0.0930 m (93.0 mm)
  Height: 0.0900 m (90.0 mm)

Camera Configuration:
  Height from ground: 0.1230 m (123.0 mm)
  Forward offset: 0.0883 m (88.3 mm)

Physical Properties:
  Total mass: 9.0 kg
  Base: 266.7 × 186.0 × 90.0 mm
  Wheel separation: 144.0 mm
  Wheel radius: 48.5 mm

✓ Habitat config written to: habitat_config_from_urdf.yaml
```

## URDF → Habitat Mapping

### Coordinate Systems

**URDF (ROS convention):**
- X: forward
- Y: left
- Z: up

**Habitat:**
- X: right
- Y: up
- Z: forward

The `urdf_to_habitat.py` handles these transformations.

### Height Calculation

Camera height is computed from URDF joint chain:

```
base_footprint (ground)
  └─ base_link (+29mm)
      └─ platform_link (+50mm)
          └─ camera_link (+12.5mm)

Total: 29 + 50 + 12.5 = 91.5mm ≈ 92mm
```

(Actual calculation includes track offset: 123mm final height)

### Collision Geometry

URDF defines box geometries for tracks and chassis. Habitat uses cylinder approximation:

- **Radius**: Max half-width = 93mm (conservative)
- **Height**: Track height = 90mm (ground clearance)

This is a **conservative approximation** - slightly larger than the tightest fit, but close to the actual tank footprint.

## Future Enhancements

Potential improvements for even better URDF integration:

1. **Full URDF Import**
   - Load complete robot mesh into Habitat
   - Preserve exact geometry (not cylinder approximation)
   - Requires Habitat mesh support

2. **Mass/Inertia Properties**
   - Extract inertial parameters from URDF
   - Apply to Habitat physics simulation
   - More realistic dynamics

3. **Track Dynamics**
   - Model differential drive explicitly
   - Use URDF wheel_separation for turning radius
   - Simulate track slip/skid

4. **Sensor Mounting**
   - Extract all sensor positions from URDF
   - Support IMU, GPS (if re-added to URDF)
   - Match sensor noise models

5. **Auto-Update**
   - Watch URDF file for changes
   - Hot-reload dimensions during development
   - Continuous validation

## Troubleshooting

### "URDF not found"

```bash
# Check URDF path
ls -la /root/ros2-rover/src/tractor_bringup/urdf/tractor.urdf.xacro

# If missing, check repo structure
# Or manually specify path in habitat_episode_runner.py
```

### "Dimensions seem wrong"

```bash
# Verify URDF parsing
cd sim
python3 urdf_to_habitat.py

# Compare with physical measurements
# Update URDF if measurements changed
```

### "Agent stuck in walls"

- Agent radius (93mm) may be too small for some scenes
- Increase `collision_distance` parameter (default: 120mm)
- Or scale agent radius by 1.1-1.2× for safety margin

## References

- URDF Specification: http://wiki.ros.org/urdf
- XACRO Tutorial: http://wiki.ros.org/xacro
- Habitat Sim Docs: https://aihabitat.org/docs/habitat-sim/
- Physical Robot: `src/tractor_bringup/urdf/tractor.urdf.xacro`
