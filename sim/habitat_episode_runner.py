#!/usr/bin/env python3
"""Habitat 3.0 autonomous episode runner for MAP-Elites training.

Runs navigation episodes in Habitat simulator and reports results to V620 server.
Compatible with the same ZeroMQ protocol as the physical rover.
"""

import io
import time
from pathlib import Path
from typing import Tuple, Optional, Dict
import argparse

import numpy as np
import torch
import zmq

# Habitat imports
try:
    import habitat
    from habitat.config.default import get_config as get_habitat_config
    from habitat.core.embodied_task import Metrics
    from habitat.tasks.nav.nav import NavigationTask
    HAS_HABITAT = True
except ImportError:
    HAS_HABITAT = False
    print("❌ Habitat not available - please install habitat-sim and habitat-lab")
    print("   pip install habitat-sim habitat-lab")
    exit(1)

# Import the model architecture from remote_training_server
import sys
sys.path.append(str(Path(__file__).parent.parent / 'remote_training_server'))
from model_architectures import RGBDEncoder, PolicyHead


class ActorNetwork(torch.nn.Module):
    """Actor-only network for MAP-Elites (matches V620 trainer)."""

    def __init__(self, proprio_dim: int = 6):
        super().__init__()
        self.encoder = RGBDEncoder()
        self.policy_head = PolicyHead(self.encoder.output_dim, proprio_dim)

    def forward(self, rgb, depth, proprio):
        """Forward pass.

        Args:
            rgb: (B, 3, H, W) RGB image
            depth: (B, 1, H, W) Depth image
            proprio: (B, 6) Proprioception

        Returns:
            action: (B, 2) [linear_vel, angular_vel] in [-1, 1] range
        """
        features = self.encoder(rgb, depth)
        action = self.policy_head(features, proprio)
        return torch.tanh(action)


class HabitatEpisodeRunner:
    """Habitat simulation episode runner for MAP-Elites."""

    def __init__(
        self,
        server_addr: str = 'tcp://localhost:5556',
        episode_duration: float = 60.0,
        max_linear_speed: float = 0.18,  # Match tank speed
        max_angular_speed: float = 1.0,
        collision_distance: float = 0.12,  # Match tank collision threshold
        inference_rate_hz: float = 30.0,
        scene_dataset: str = 'default',
        device: str = 'cuda',
        num_parallel_episodes: int = 1,
        use_urdf_dimensions: bool = True,
    ):
        """Initialize Habitat episode runner.

        Args:
            server_addr: ZeroMQ server address (V620 trainer)
            episode_duration: Max episode length in seconds
            max_linear_speed: Max forward velocity (m/s)
            max_angular_speed: Max angular velocity (rad/s)
            collision_distance: Collision threshold (m)
            inference_rate_hz: Inference frequency
            scene_dataset: Habitat scene dataset to use
            device: PyTorch device (cuda/cpu)
            num_parallel_episodes: Number of parallel environments (future)
            use_urdf_dimensions: Use URDF-based robot dimensions (recommended)
        """
        self.server_addr = server_addr
        self.episode_duration = episode_duration
        self.max_linear = max_linear_speed
        self.max_angular = max_angular_speed
        self.collision_dist = collision_distance
        self.inference_rate = inference_rate_hz
        self.dt = 1.0 / inference_rate_hz  # Timestep
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load robot dimensions from URDF
        if use_urdf_dimensions:
            from urdf_to_habitat import extract_robot_dimensions
            urdf_path = Path(__file__).parent.parent / 'src/tractor_bringup/urdf/tractor.urdf.xacro'
            if urdf_path.exists():
                self.robot_dims = extract_robot_dimensions(str(urdf_path))
                print(f"✓ Loaded URDF dimensions:")
                print(f"  Agent radius: {self.robot_dims['agent_radius']*1000:.0f}mm")
                print(f"  Agent height: {self.robot_dims['agent_height']*1000:.0f}mm")
                print(f"  Camera height: {self.robot_dims['camera_height']*1000:.0f}mm")
            else:
                print(f"⚠ URDF not found, using default dimensions")
                self.robot_dims = None
        else:
            self.robot_dims = None

        print(f"Device: {self.device}")
        print(f"Episode duration: {episode_duration}s")
        print(f"Inference rate: {inference_rate_hz} Hz")

        # Initialize Habitat environment
        self.env = self._create_habitat_env(scene_dataset)
        print(f"✓ Habitat environment initialized")

        # State
        self.current_model = None
        self.current_model_id = None

        # ZeroMQ REQ socket for bidirectional communication with V620
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_socket.connect(server_addr)
        self.zmq_socket.setsockopt(zmq.LINGER, 0)
        print(f"✓ Connected to V620 server: {server_addr}")

    def _create_habitat_env(self, scene_dataset: str):
        """Create Habitat environment with navigation task."""
        # Create basic navigation config
        config = habitat.get_config(
            config_paths="benchmark/nav/pointnav/pointnav_habitat_test.yaml"
        )

        # Customize config for our needs
        config.defrost()

        # Simulator settings
        config.SIMULATOR.TYPE = "Sim-v0"
        config.SIMULATOR.ACTION_SPACE_CONFIG = "v0"
        config.SIMULATOR.FORWARD_STEP_SIZE = 0.25  # Not used (we use velocity control)
        config.SIMULATOR.TURN_ANGLE = 10  # Not used

        # Get robot dimensions (from URDF if available, otherwise defaults)
        if self.robot_dims:
            agent_height = self.robot_dims['agent_height']
            agent_radius = self.robot_dims['agent_radius']
            camera_height = self.robot_dims['camera_height']
        else:
            # Fallback to URDF-based defaults (NOT the old incorrect values)
            agent_height = 0.090  # 90mm
            agent_radius = 0.093  # 93mm
            camera_height = 0.123  # 123mm

        # RGB sensor (match RealSense D435i resolution)
        config.SIMULATOR.RGB_SENSOR.WIDTH = 640
        config.SIMULATOR.RGB_SENSOR.HEIGHT = 480
        config.SIMULATOR.RGB_SENSOR.HFOV = 69  # D435i horizontal FOV
        config.SIMULATOR.RGB_SENSOR.POSITION = [0, camera_height, 0]  # URDF-based height

        # Depth sensor
        config.SIMULATOR.DEPTH_SENSOR.WIDTH = 640
        config.SIMULATOR.DEPTH_SENSOR.HEIGHT = 480
        config.SIMULATOR.DEPTH_SENSOR.HFOV = 69
        config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.0
        config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 10.0
        config.SIMULATOR.DEPTH_SENSOR.POSITION = [0, camera_height, 0]  # URDF-based height

        # Agent configuration (URDF-based tank dimensions)
        config.SIMULATOR.AGENT_0.HEIGHT = agent_height  # 90mm - actual tank height
        config.SIMULATOR.AGENT_0.RADIUS = agent_radius  # 93mm - actual tank radius

        # Task settings
        config.TASK.TYPE = "Nav-v0"
        config.TASK.SUCCESS_DISTANCE = 0.2
        config.TASK.MEASUREMENTS = ['DISTANCE_TO_GOAL', 'SUCCESS', 'SPL', 'COLLISIONS']

        # Environment settings
        config.ENVIRONMENT.MAX_EPISODE_STEPS = int(self.episode_duration * self.inference_rate)
        config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True

        config.freeze()

        # Create environment
        env = habitat.Env(config=config)
        return env

    def request_new_model(self) -> bool:
        """Request new model from V620 server.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Send model request
            request = {'type': 'request_model'}
            self.zmq_socket.send_pyobj(request)
            print('\n→ Requesting new model from V620...')

            # Wait for response
            if self.zmq_socket.poll(timeout=-1):  # Infinite timeout
                response = self.zmq_socket.recv_pyobj()

                if response['type'] == 'model':
                    self.current_model_id = response['model_id']
                    model_bytes = response['model_bytes']

                    print(f'✓ Received model #{self.current_model_id} '
                          f'({response["generation_type"]}) - {len(model_bytes)/1024:.1f} KB')

                    # Load model from bytes
                    buffer = io.BytesIO(model_bytes)
                    model_state = torch.load(buffer, map_location='cpu')

                    # Load into ActorNetwork
                    self.current_model = ActorNetwork().to(self.device)
                    self.current_model.load_state_dict(model_state)
                    self.current_model.eval()

                    print(f'✓ Model loaded on {self.device}')
                    return True

                else:
                    print(f'❌ Unexpected response type: {response.get("type")}')
                    return False

            else:
                print('❌ Timeout waiting for model from V620')
                return False

        except Exception as e:
            print(f'❌ Failed to request model: {e}')
            import traceback
            traceback.print_exc()
            return False

    def run_episode(self) -> Dict:
        """Run single autonomous episode in Habitat.

        Returns:
            Episode metrics dictionary
        """
        print(f'\n🚀 Starting episode #{self.current_model_id}...')

        # Reset environment
        observations = self.env.reset()

        # Episode state
        episode_start_time = time.time()
        start_position = self.env.sim.get_agent_state().position.copy()

        total_distance = 0.0
        collision_count = 0
        speed_samples = []
        clearance_samples = []
        action_samples = []
        heading_samples = []

        # Tank-specific metrics
        stationary_rotation_time = 0.0
        track_slip_detected = False

        # Trajectory collection
        trajectory_rgb = []
        trajectory_depth = []
        trajectory_proprio = []
        trajectory_actions = []

        # Agent velocity state (simulated)
        current_linear_vel = 0.0
        current_angular_vel = 0.0
        current_heading = 0.0

        step_count = 0
        max_steps = int(self.episode_duration * self.inference_rate)

        with torch.no_grad():
            for step in range(max_steps):
                # Extract sensor data
                rgb = observations['rgb']  # (H, W, 3) uint8
                depth = observations['depth']  # (H, W, 1) float32

                # Compute minimum forward distance (from depth sensor)
                # Get center column of depth map
                center_col = depth.shape[1] // 2
                center_strip = depth[:, center_col-50:center_col+50, 0]  # 100px wide strip
                min_forward_dist = float(np.min(center_strip))

                # Prepare model inputs
                rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0  # (1, 3, H, W)
                depth_tensor = torch.from_numpy(depth).permute(2, 0, 1).unsqueeze(0).float().to(self.device)  # (1, 1, H, W)

                # Proprioception: [lin_vel, ang_vel, roll, pitch, accel, clearance]
                proprio = torch.tensor([[
                    current_linear_vel,
                    current_angular_vel,
                    0.0,  # roll (flat ground)
                    0.0,  # pitch
                    0.0,  # acceleration (not used)
                    min_forward_dist
                ]], dtype=torch.float32, device=self.device)

                # Run model inference
                action = self.current_model(rgb_tensor, depth_tensor, proprio)
                action_np = action.cpu().numpy()[0]  # (2,) [linear, angular] in [-1, 1]

                # Scale to velocity commands
                linear_cmd = float(np.clip(action_np[0], -1.0, 1.0) * self.max_linear)
                angular_cmd = float(np.clip(action_np[1], -1.0, 1.0) * self.max_angular)

                # Safety: stop if too close to obstacle
                if min_forward_dist < self.collision_dist * 2:
                    linear_cmd = 0.0

                # Simulate velocity dynamics (simplified tank model)
                # Instant velocity change for sim (real tank has inertia)
                current_linear_vel = linear_cmd
                current_angular_vel = angular_cmd
                current_heading += angular_cmd * self.dt

                # Apply velocity control to agent
                # Habitat uses discrete actions, so we approximate with position updates
                forward_delta = current_linear_vel * self.dt
                rotation_delta = current_angular_vel * self.dt

                # Get current agent state
                agent_state = self.env.sim.get_agent_state()

                # Update position based on velocity
                # Forward movement in agent's heading direction
                dx = forward_delta * np.sin(current_heading)
                dy = 0.0  # No vertical movement
                dz = forward_delta * np.cos(current_heading)

                new_position = agent_state.position + np.array([dx, dy, dz])

                # Update rotation
                new_rotation = agent_state.rotation  # Keep for now (simplified)

                # Set new agent state
                agent_state.position = new_position
                agent_state.rotation = new_rotation
                self.env.sim.set_agent_state(agent_state.position, agent_state.rotation)

                # Get new observations after movement
                observations = self.env.sim.get_observations_at(
                    agent_state.position,
                    agent_state.rotation
                )

                # Check for collision
                if self.env.sim.previous_step_collided:
                    collision_count += 1

                # Compute distance traveled
                current_position = agent_state.position
                step_distance = np.linalg.norm(current_position - start_position)
                total_distance = max(total_distance, step_distance)

                # Record metrics
                speed_samples.append(abs(current_linear_vel))
                clearance_samples.append(min_forward_dist)
                action_samples.append([action_np[0], action_np[1]])
                heading_samples.append(current_heading)

                # Tank-specific metrics
                if abs(current_linear_vel) < 0.02 and abs(angular_cmd) > 0.1:
                    stationary_rotation_time += self.dt

                # Trajectory collection (subsample to match rover: ~1800 samples max)
                if step % 1 == 0 and len(trajectory_rgb) < 1800:
                    trajectory_rgb.append(rgb.copy())
                    trajectory_depth.append(depth[:, :, 0].copy())  # Remove channel dim
                    trajectory_proprio.append([
                        current_linear_vel,
                        current_angular_vel,
                        0.0, 0.0, 0.0,
                        min_forward_dist
                    ])
                    trajectory_actions.append([
                        linear_cmd / self.max_linear,
                        angular_cmd / self.max_angular
                    ])

                step_count += 1

        # Episode complete
        duration = time.time() - episode_start_time

        # Compute metrics
        avg_speed = np.mean(speed_samples) if speed_samples else 0.0
        avg_clearance = np.mean(clearance_samples) if clearance_samples else 0.0

        # Turn efficiency
        turn_efficiency = 0.0
        if len(heading_samples) > 1 and total_distance > 0.1:
            total_heading_change = np.sum(np.abs(np.diff(heading_samples)))
            turn_efficiency = total_distance / (total_heading_change + 1e-6)

        # Action smoothness
        action_smoothness = 0.0
        avg_linear_action = 0.0
        avg_angular_action = 0.0
        if len(action_samples) > 1:
            actions = np.array(action_samples)
            angular_actions = actions[:, 1]
            angular_diffs = np.diff(angular_actions)
            action_smoothness = float(np.std(angular_diffs))
            avg_linear_action = float(np.mean(np.abs(actions[:, 0])))
            avg_angular_action = float(np.mean(np.abs(actions[:, 1])))

        # Cache trajectory
        trajectory_data = None
        if len(trajectory_rgb) > 0:
            trajectory_data = {
                'rgb': np.array(trajectory_rgb, dtype=np.uint8),
                'depth': np.array(trajectory_depth, dtype=np.float32),
                'proprio': np.array(trajectory_proprio, dtype=np.float32),
                'actions': np.array(trajectory_actions, dtype=np.float32),
            }

        # Package results
        results = {
            'type': 'episode_result',
            'model_id': self.current_model_id,
            'total_distance': float(total_distance),
            'collision_count': int(collision_count),
            'avg_speed': float(avg_speed),
            'avg_clearance': float(avg_clearance),
            'duration': float(duration),
            'action_smoothness': float(action_smoothness),
            'avg_linear_action': float(avg_linear_action),
            'avg_angular_action': float(avg_angular_action),
            'turn_efficiency': float(turn_efficiency),
            'stationary_rotation_time': float(stationary_rotation_time),
            'track_slip_detected': bool(track_slip_detected),
        }

        print(f'✓ Episode complete: dist={total_distance:.2f}m, '
              f'collisions={collision_count}, '
              f'avg_speed={avg_speed:.3f}m/s, '
              f'duration={duration:.1f}s (sim), '
              f'steps={step_count}')

        return results, trajectory_data

    def send_results(self, results: Dict, trajectory_data: Optional[Dict]) -> bool:
        """Send episode results to V620 server.

        Args:
            results: Episode metrics
            trajectory_data: Optional trajectory data for refinement

        Returns:
            True if acknowledged
        """
        try:
            # Send results
            self.zmq_socket.send_pyobj(results)
            print(f'→ Sent results for model #{self.current_model_id}')

            # Wait for acknowledgment
            if self.zmq_socket.poll(timeout=10000):  # 10s timeout
                ack = self.zmq_socket.recv_pyobj()

                if ack.get('type') == 'ack':
                    print('✓ Results acknowledged by V620')

                    # Check if server wants trajectory
                    if ack.get('collect_trajectory', False):
                        model_id = ack.get('model_id', self.current_model_id)
                        print(f'→ Server requested trajectory for model #{model_id}')

                        if trajectory_data is not None:
                            self.send_trajectory(model_id, trajectory_data)
                        else:
                            print('⚠ No trajectory data available')

                    return True
                else:
                    print(f'⚠ Unexpected ack: {ack}')
                    return False
            else:
                print('❌ Timeout waiting for acknowledgment')
                return False

        except Exception as e:
            print(f'❌ Failed to send results: {e}')
            import traceback
            traceback.print_exc()
            return False

    def send_trajectory(self, model_id: int, trajectory_data: Dict):
        """Send trajectory data to V620 server (with compression)."""
        try:
            import zstandard as zstd

            print(f'  Compressing trajectory: {len(trajectory_data["actions"])} samples...')
            start_time = time.time()

            # Compress with Zstandard
            cctx = zstd.ZstdCompressor(level=10)
            rgb_compressed = cctx.compress(trajectory_data['rgb'].tobytes())
            depth_compressed = cctx.compress(trajectory_data['depth'].tobytes())

            original_mb = (trajectory_data['rgb'].nbytes + trajectory_data['depth'].nbytes) / 1024 / 1024
            compressed_mb = (len(rgb_compressed) + len(depth_compressed)) / 1024 / 1024

            print(f'  Compressed: {original_mb:.1f} MB → {compressed_mb:.1f} MB '
                  f'({original_mb/compressed_mb:.1f}x)')

            # Prepare message
            trajectory_message = {
                'type': 'trajectory_data',
                'model_id': model_id,
                'compressed': True,
                'compression': 'zstd',
                'trajectory': {
                    'rgb': rgb_compressed,
                    'rgb_shape': trajectory_data['rgb'].shape,
                    'depth': depth_compressed,
                    'depth_shape': trajectory_data['depth'].shape,
                    'proprio': trajectory_data['proprio'],
                    'actions': trajectory_data['actions'],
                }
            }

            # Send
            self.zmq_socket.send_pyobj(trajectory_message)
            send_time = time.time() - start_time
            print(f'  Sent in {send_time:.1f}s, waiting for tournament...')

            # Wait for acknowledgment (tournament can take time)
            if self.zmq_socket.poll(timeout=-1):  # Infinite timeout
                ack = self.zmq_socket.recv_pyobj()
                total_time = time.time() - start_time
                if ack.get('type') == 'ack':
                    print(f'✓ Tournament complete ({total_time:.1f}s total)')
                else:
                    print(f'⚠ Unexpected ack: {ack}')

        except Exception as e:
            print(f'❌ Failed to send trajectory: {e}')
            import traceback
            traceback.print_exc()

    def run(self, num_episodes: int = 1000):
        """Run training loop.

        Args:
            num_episodes: Number of episodes to run
        """
        print("\n" + "=" * 60)
        print("Starting Habitat MAP-Elites Training")
        print("=" * 60)
        print(f"Target episodes: {num_episodes}")
        print(f"Server: {self.server_addr}")
        print()

        for episode_idx in range(num_episodes):
            try:
                # Request new model
                if not self.request_new_model():
                    print("⚠ Failed to get model, retrying in 5s...")
                    time.sleep(5.0)
                    continue

                # Run episode
                results, trajectory_data = self.run_episode()

                # Send results
                self.send_results(results, trajectory_data)

                # Brief pause between episodes
                time.sleep(1.0)

            except KeyboardInterrupt:
                print("\n🛑 Training interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error in episode {episode_idx}: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5.0)
                continue

        print("\n" + "=" * 60)
        print("✅ Habitat Training Complete!")
        print("=" * 60)

    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'env'):
            self.env.close()
        if hasattr(self, 'zmq_socket'):
            self.zmq_socket.close()
        if hasattr(self, 'zmq_context'):
            self.zmq_context.term()


def main():
    parser = argparse.ArgumentParser(description='Habitat episode runner for MAP-Elites')
    parser.add_argument('--server', type=str, default='tcp://localhost:5556',
                        help='V620 server address')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to run')
    parser.add_argument('--duration', type=float, default=60.0,
                        help='Episode duration (seconds)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='PyTorch device (cuda/cpu)')
    parser.add_argument('--scene', type=str, default='default',
                        help='Habitat scene dataset')
    args = parser.parse_args()

    runner = HabitatEpisodeRunner(
        server_addr=args.server,
        episode_duration=args.duration,
        device=args.device,
        scene_dataset=args.scene,
    )

    try:
        runner.run(num_episodes=args.episodes)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        runner.cleanup()


if __name__ == '__main__':
    main()
