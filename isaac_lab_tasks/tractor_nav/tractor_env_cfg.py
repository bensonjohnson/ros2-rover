"""Config for Isaac-Tractor-Nav-v0.

Direct-style RL env (subclass of DirectRLEnv). RGB-128x72 + proprio
observations, 4-wheel velocity action (skid-steer), pure exploration
reward (forward velocity − collisions − action jerk − tilt).
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg


# Path to the converted USD. Run convert_urdf.py once, then set this env var
# or drop the file at the default location.
TRACTOR_USD = os.environ.get(
    "TRACTOR_USD",
    os.path.expanduser("~/Documents/ros2-rover/tractor_isaac.usd"),
)


TRACTOR_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Tractor",
    spawn=sim_utils.UsdFileCfg(
        usd_path=TRACTOR_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.08),
        joint_pos={".*_wheel_joint": 0.0},
        joint_vel={".*_wheel_joint": 0.0},
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            velocity_limit=20.0,   # rad/s; ~0.9 m/s at r=0.045
            effort_limit=4.0,      # N·m per wheel
            stiffness=0.0,         # velocity control: no position stiffness
            damping=15.0,
        ),
    },
)


@configclass
class TractorSceneCfg(InteractiveSceneCfg):
    """Flat ground + tractor + RGB camera + contact sensor on chassis."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.95,
            dynamic_friction=0.9,
        ),
    )

    robot: ArticulationCfg = TRACTOR_CFG

    # Front RGB camera, matched to URDF camera_link pose.
    # Mounted on the tractor so it moves with the chassis.
    camera = CameraCfg(
        prim_path="/World/envs/env_.*/Tractor/base_link/front_camera",
        update_period=0.1,          # 10 Hz render (policy step rate)
        height=72,
        width=128,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.93,      # mm — D435i / typical Arducam wide
            focus_distance=400.0,
            horizontal_aperture=2.65,
            clipping_range=(0.05, 30.0),
        ),
        # Position relative to base_link, in meters. Combines platform offset
        # (0.08025) and camera_joint (0.12085, 0, 0.25225) from the URDF.
        offset=CameraCfg.OffsetCfg(
            pos=(0.12085, 0.0, 0.3325),
            rot=(0.5, -0.5, 0.5, -0.5),  # camera optical: x=right, y=down, z=fwd
            convention="ros",
        ),
    )

    # Contact sensor on the chassis — anything touching base_link is a collision.
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Tractor/base_link",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )

    # Light so the camera sees something.
    dome_light = sim_utils.DomeLightCfg(
        intensity=2500.0,
        color=(0.9, 0.95, 1.0),
    )

    num_envs: int = 4096
    env_spacing: float = 6.0


@configclass
class TractorNavEnvCfg(DirectRLEnvCfg):
    # Episode
    episode_length_s: float = 20.0
    decimation: int = 5             # 100 Hz physics / 5 = 20 Hz policy
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 100.0, render_interval=5)

    # Scene
    scene: TractorSceneCfg = TractorSceneCfg(num_envs=4096, env_spacing=6.0)

    # Action / observation shapes
    action_space: int = 4           # [FL, FR, RL, RR] wheel velocity in [-1, 1]
    observation_space: int = 0      # filled in by env (dict obs: image + proprio)
    state_space: int = 0

    # Action scaling
    max_wheel_vel: float = 18.0     # rad/s — leaves headroom under velocity_limit

    # Reward weights
    rew_forward_vel: float = 1.0
    rew_alive: float = 0.05
    pen_action_rate: float = -0.01
    pen_lateral_vel: float = -0.05
    pen_tilt: float = -0.2
    pen_collision: float = -2.0

    # Termination thresholds
    tilt_limit_rad: float = 0.6     # ~34° — flipped over
    collision_force_threshold: float = 1.0  # N

    # Domain randomization (camera image augmentation handled inside env)
    randomize_friction_range: tuple[float, float] = (0.6, 1.1)
    randomize_mass_range_kg: tuple[float, float] = (-0.5, 1.5)
