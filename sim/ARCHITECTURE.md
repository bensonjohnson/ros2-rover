# Habitat Simulation Architecture

## Protocol Compatibility

The Habitat simulation uses **exactly the same ZeroMQ protocol** as the physical rover, making them interchangeable from the V620 trainer's perspective.

### Message Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    MAP-Elites Trainer                       │
│              (v620_map_elites_trainer.py)                   │
│                                                             │
│  - Maintains population of models (10-25 individuals)      │
│  - Runs tournament selection (75-500 candidates)           │
│  - Computes fitness from episode metrics                   │
│  - Performs gradient refinement on trajectories            │
└──────────────┬──────────────────────────────────────────────┘
               │
               │ ZeroMQ REQ-REP (tcp://*:5556)
               │
    ┌──────────┴───────────┐
    │                      │
┌───▼──────────────┐  ┌────▼────────────────┐
│  Physical Rover  │  │ Habitat Simulator   │
│ (ROS2 Episode    │  │ (habitat_episode_   │
│  Runner)         │  │  runner.py)         │
└──────────────────┘  └─────────────────────┘
```

## Message Types

### 1. Request Model

**Rover/Sim → Trainer:**
```python
{
    'type': 'request_model'
}
```

**Trainer → Rover/Sim:**
```python
{
    'type': 'model',
    'model_id': 42,
    'model_bytes': b'...',  # PyTorch state_dict serialized
    'generation_type': 'mutation'  # or 'random', 'tournament', etc.
}
```

### 2. Episode Result

**Rover/Sim → Trainer:**
```python
{
    'type': 'episode_result',
    'model_id': 42,
    'total_distance': 3.45,        # meters
    'collision_count': 2,
    'avg_speed': 0.12,             # m/s
    'avg_clearance': 0.85,         # m
    'duration': 62.3,              # seconds
    'action_smoothness': 0.15,     # std of angular velocity changes
    'avg_linear_action': 0.4,      # average forward intent
    'avg_angular_action': 0.2,     # average rotation intent
    'turn_efficiency': 2.1,        # distance per heading change
    'stationary_rotation_time': 8.5,  # seconds pivoting in place
    'track_slip_detected': False
}
```

**Trainer → Rover/Sim:**
```python
{
    'type': 'ack',
    'collect_trajectory': True,     # Request trajectory for refinement
    'model_id': 42,                 # Which model to re-run
    'suggested_episode_duration': 60.0  # Variable duration
}
```

### 3. Trajectory Data (Optional)

**Rover/Sim → Trainer:**
```python
{
    'type': 'trajectory_data',
    'model_id': 42,
    'compressed': True,
    'compression': 'zstd',
    'trajectory': {
        'rgb': b'...',              # Compressed (N, H, W, 3) uint8
        'rgb_shape': (450, 480, 640, 3),
        'depth': b'...',            # Compressed (N, H, W) float32
        'depth_shape': (450, 480, 640),
        'proprio': ndarray,         # (N, 6) [vel, ang_vel, roll, pitch, accel, clearance]
        'actions': ndarray,         # (N, 2) [linear, angular] in [-1, 1]
    }
}
```

**Trainer → Rover/Sim:**
```python
{
    'type': 'ack'
}
```

## Key Differences: Rover vs Simulation

### Rover (map_elites_episode_runner.py)

**Advantages:**
- ✅ Real-world physics (track friction, inertia, terrain)
- ✅ True sensor noise (depth errors, motion blur)
- ✅ Actual lighting conditions
- ✅ Real collision consequences

**Limitations:**
- ⏱️ Slow (1 episode per minute)
- 🔋 Battery limited
- 🛠️ Hardware wear (track damage)
- 🐌 RKNN conversion overhead (30s per model)
- 🏠 Requires physical space

**Pipeline:**
```
PyTorch model → ONNX → RKNN → NPU inference
(on V620)              (on rover)
```

### Habitat Simulation (habitat_episode_runner.py)

**Advantages:**
- ⚡ Fast (8-12 episodes per minute, 10-20x realtime)
- 🔄 Unlimited episodes (no battery, no wear)
- 🎯 Perfect reproducibility
- 💻 Direct PyTorch inference (no conversion)
- 🌍 Diverse environments (change scenes easily)
- 📊 Parallelizable (run multiple instances)

**Limitations:**
- ⚠️ Simplified physics (no track dynamics)
- 📷 Perfect sensors (unless noise added)
- 🎨 Synthetic visuals (sim-to-real gap)
- 💾 Requires scene data (~10GB+)

**Pipeline:**
```
PyTorch model → GPU inference
(on V620)       (on sim machine)
```

## Implementation Details

### Sensor Matching

Both rover and sim use identical sensor configurations:

| Parameter | Value | Source |
|-----------|-------|--------|
| RGB Resolution | 640x480 | RealSense D435i |
| Depth Resolution | 640x480 | RealSense D435i |
| Horizontal FOV | 69° | RealSense D435i |
| Depth Range | 0-10m | RealSense D435i |
| Camera Height | 0.88m | Tank mounting |

### Agent Matching

| Parameter | Value | Notes |
|-----------|-------|-------|
| Radius | 0.18m | Tank half-width |
| Height | 0.88m | Tank height |
| Max Linear Speed | 0.18 m/s | Tank motor limit |
| Max Angular Speed | 1.0 rad/s | Tank steering |
| Collision Distance | 0.12m | Safety threshold |
| Control Rate | 30Hz | ROS2 timer rate |

### Model Architecture

Both use the same `ActorNetwork`:
- **Encoder**: RGBDEncoder (shared ResNet-based vision)
- **Policy Head**: MLP with proprioception fusion
- **Output**: `tanh`-squashed actions `[-1, 1]²`

### Fitness Function

Identical fitness computation:
```python
fitness = distance * 2.0
         - collision_penalty
         + clearance_bonus
         + pivot_turn_reward
         - spinning_penalty
         + smoothness_bonus
         - track_slip_penalty
         + exploration_bonus
         + diversity_bonus  # Only if enabled
```

## Performance Comparison

| Metric | Physical Rover | Habitat Sim | Speedup |
|--------|---------------|-------------|---------|
| Episode runtime (60s) | 60-90s | 6-12s | 10x |
| Model conversion | 30s | 0s | ∞ |
| Throughput | 1 ep/min | 8-12 ep/min | 10x |
| Trajectory upload | 5-8s | 1-2s | 4x |
| Power consumption | 30W (rover) | 200W (GPU) | 0.15x |
| Wear and tear | Yes | None | N/A |

**Training time for 1000 episodes:**
- Rover: ~16 hours
- Sim: ~1.5 hours
- **Speedup: 11x**

## Recommended Workflow

### Hybrid Training Strategy

```
Phase 1: Rapid Exploration (Episodes 0-200)
  → Use Habitat simulation
  → Goal: Build diverse population quickly
  → Time: ~30 minutes

Phase 2: Real-world Validation (Episodes 200-400)
  → Switch to physical rover
  → Goal: Discover sim-to-real gaps
  → Time: ~3 hours

Phase 3: Focused Refinement (Episodes 400-600)
  → Use Habitat for bulk evaluations
  → Periodically validate on rover (every 50 episodes)
  → Goal: Optimize with real-world checkpoints
  → Time: ~45 minutes + 30 min validation

Phase 4: Final Polish (Episodes 600-1000)
  → Physical rover for final 400 episodes
  → Goal: Real-world robustness
  → Time: ~6 hours

Total time: ~10.5 hours (vs 16 hours rover-only)
```

### Parallel Training

Run both simultaneously:

```bash
# Terminal 1: Start trainer
cd remote_training_server
./start_map_elites_server.sh 5556 1000

# Terminal 2: Start rover
cd src/tractor_bringup
./start_map_elites_rover.sh

# Terminal 3: Start simulation
cd sim
./start_habitat_training.sh tcp://10.0.0.200:5556 1000
```

The trainer will distribute models to both clients, mixing real and simulated episodes!

## Future Enhancements

### Potential Improvements

1. **Domain Randomization**
   - Vary lighting, textures, sensor noise
   - Add artificial depth errors
   - Simulate IMU drift

2. **Physics Matching**
   - Add track slip model
   - Simulate motor lag/inertia
   - Match mass/acceleration curves

3. **Multi-Environment Training**
   - Rotate through Gibson, Matterport3D, HM3D
   - Test generalization across scene types
   - Curriculum learning (easy → hard scenes)

4. **Distributed Simulation**
   - Run 10+ parallel Habitat instances
   - Share ZeroMQ server across machines
   - Target: 100+ episodes/minute

5. **Visualization**
   - Real-time 3D viewer
   - Trajectory replay
   - Behavior archive visualization

## See Also

- [V620 MAP-Elites Trainer](../remote_training_server/v620_map_elites_trainer.py)
- [Rover Episode Runner](../src/tractor_bringup/tractor_bringup/map_elites_episode_runner.py)
- [Habitat Documentation](https://aihabitat.org/docs/habitat-lab/)
