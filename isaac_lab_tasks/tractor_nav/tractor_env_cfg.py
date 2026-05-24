"""Config for Isaac-Tractor-Nav-v0.

Direct-style RL env (subclass of DirectRLEnv). Single-tensor observation
of shape (H, W, 3 + P): RGB camera channels + per-env proprio values
broadcast as constant spatial planes. 4-wheel velocity action
(skid-steer). Exploration reward — forward vel minus collisions, jerk,
tilt, lateral drift.

Scene includes scattered box obstacles per env so the policy has to
learn to avoid them. Domain randomization (physics + visual) is applied
at reset / observation time inside `tractor_env.py`.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    RigidObjectCfg,
    RigidObjectCollectionCfg,
)
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


# Path to the converted USD. Run convert_urdf.py once, then set this env var
# or drop the file at the default location.
TRACTOR_USD = os.environ.get(
    "TRACTOR_USD",
    os.path.expanduser("~/ros2-rover/tractor_isaac/tractor_isaac.usd"),
)


# Image dims — matched to the deployed Arducam stream after letterbox.
IMAGE_HEIGHT = 72
IMAGE_WIDTH = 128

# Proprio = [lin_vel_x_b, lin_vel_y_b, ang_vel_z_b, last_action[0..3]] = 7 floats.
PROPRIO_DIM = 7

# Per-env obstacle count. Each is a free rigid cube re-placed on reset.
NUM_OBSTACLES_PER_ENV = 12


TRACTOR_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Tractor",
    spawn=sim_utils.UsdFileCfg(
        usd_path=TRACTOR_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        activate_contact_sensors=True,
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
            velocity_limit_sim=20.0,   # rad/s — matches velocity_limit on the rover
            effort_limit_sim=4.0,      # N·m per wheel
            stiffness=0.0,             # velocity control: no position stiffness
            damping=15.0,
        ),
    },
)


def _build_obstacle_collection() -> RigidObjectCollectionCfg:
    """N free cubes per env, initial pose far below ground.

    Real positions get assigned in `_reset_idx`; we just need each prim
    to exist with a unique path. Stashing them at z=-100 keeps them out
    of the way before the first reset.
    """
    cube_props = sim_utils.RigidBodyPropertiesCfg(
        max_linear_velocity=10.0,
        max_depenetration_velocity=2.0,
        disable_gravity=False,
        kinematic_enabled=False,
    )
    mass_props = sim_utils.MassPropertiesCfg(mass=2.0)
    collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=True)

    objs: dict[str, RigidObjectCfg] = {}
    for i in range(NUM_OBSTACLES_PER_ENV):
        objs[f"box_{i}"] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Obstacle_box_{i}",
            spawn=sim_utils.CuboidCfg(
                size=(0.30, 0.30, 0.30),
                rigid_props=cube_props,
                mass_props=mass_props,
                collision_props=collision_props,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.6, 0.4, 0.2), roughness=0.7
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, -100.0 - i),  # stash; _reset_idx repositions
            ),
        )
    return RigidObjectCollectionCfg(rigid_objects=objs)


@configclass
class TractorSceneCfg(InteractiveSceneCfg):
    """Flat ground + tractor + scattered cube obstacles + tiled camera + contact sensor."""

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

    # Cube obstacles, one collection per env. Re-randomized on reset.
    obstacles: RigidObjectCollectionCfg = _build_obstacle_collection()

    # TiledCamera renders all envs into a single big tile and slices it —
    # much faster than a regular Camera for large num_envs.
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Tractor/base_link/front_camera",
        update_period=0.0,         # render every render step
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.93,
            focus_distance=400.0,
            horizontal_aperture=2.65,
            clipping_range=(0.05, 30.0),
        ),
        # Position relative to base_link, matches the URDF: platform offset
        # (z=0.08025) + camera_joint (x=0.12085, z=0.25225).
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.12085, 0.0, 0.3325),
            rot=(0.5, -0.5, 0.5, -0.5),  # ROS optical: x=right, y=down, z=forward
            convention="ros",
        ),
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Tractor/base_link",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )

    # Dome light so the camera has something to see. Bare spawn configs
    # are rejected by InteractiveScene; wrap in AssetBaseCfg.
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=2500.0,
            color=(0.9, 0.95, 1.0),
        ),
    )


@configclass
class TractorNavEnvCfg(DirectRLEnvCfg):
    # Episode
    episode_length_s: float = 20.0
    decimation: int = 5             # 100 Hz physics / 5 = 20 Hz policy
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 100.0, render_interval=5)

    # Scene
    scene: TractorSceneCfg = TractorSceneCfg(num_envs=4096, env_spacing=6.0)

    # Spaces — single-tensor obs (H, W, 3 + P), see _get_observations.
    action_space: int = 4           # [FL, FR, RL, RR] wheel velocity in [-1, 1]
    observation_space: list = [IMAGE_HEIGHT, IMAGE_WIDTH, 3 + PROPRIO_DIM]
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

    # Obstacle placement (uniform square around env origin, robot at center)
    obstacle_area_half: float = 2.5     # m — area where boxes can spawn
    obstacle_min_robot_dist: float = 0.8  # m — keep clear around the robot

    # Domain randomization — physics
    randomize_friction_range: tuple[float, float] = (0.6, 1.1)
    randomize_mass_range_kg: tuple[float, float] = (-0.5, 1.5)  # additive
    randomize_wheel_damping_range: tuple[float, float] = (10.0, 22.0)

    # Domain randomization — visual (applied per-env per-step on GPU)
    aug_brightness: float = 0.10        # ± U
    aug_contrast: float = 0.15          # 1 ± U
    aug_noise_std: float = 0.02         # additive Gaussian per pixel
