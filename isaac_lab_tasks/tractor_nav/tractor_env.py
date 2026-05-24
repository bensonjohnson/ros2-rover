"""DirectRLEnv implementation for Isaac-Tractor-Nav-v0.

Pure-exploration policy: drive forward, avoid the scattered cubes,
don't flip, don't oscillate.

Observation: single tensor (H, W, 3 + P) — RGB image (NHWC) concatenated
along the channel axis with proprio values broadcast as constant spatial
planes. This lets a stock CNN consume both signals without dict obs.
Visual augmentation (brightness / contrast / Gaussian noise) is applied
per-env on the GPU at observation time.

Action: 4D wheel velocity command in [-1, 1] → wheel velocity targets.

Domain randomization: friction, mass, and wheel actuator damping are
sampled per-env on reset. Box obstacle positions are re-randomized on
reset.
"""

from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, TiledCamera

from .tractor_env_cfg import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_OBSTACLES_PER_ENV, PROPRIO_DIM, TractorNavEnvCfg


class TractorNavEnv(DirectRLEnv):
    cfg: TractorNavEnvCfg

    def __init__(self, cfg: TractorNavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint indices for the 4 wheels. Names come from tractor_isaac.urdf.
        self._wheel_joint_ids, _ = self._robot.find_joints(
            [
                "front_left_wheel_joint",
                "front_right_wheel_joint",
                "rear_left_wheel_joint",
                "rear_right_wheel_joint",
            ]
        )

        self._last_action = torch.zeros(self.num_envs, 4, device=self.device)
        self._action_buf = torch.zeros(self.num_envs, 4, device=self.device)

        # Cache nominal masses so additive-mass randomization is reproducible.
        self._nominal_masses = self._robot.root_physx_view.get_masses().clone()

    # ---------- scene setup ----------
    def _setup_scene(self):
        # InteractiveScene already spawned everything from cfg.scene; we just
        # grab typed references.
        self._robot: Articulation = self.scene["robot"]
        self._camera: TiledCamera = self.scene["tiled_camera"]
        self._contact: ContactSensor = self.scene["contact_sensor"]
        self._obstacles: RigidObjectCollection = self.scene["obstacles"]

    # ---------- action ----------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._action_buf = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        vel_targets = self._action_buf * self.cfg.max_wheel_vel
        self._robot.set_joint_velocity_target(vel_targets, joint_ids=self._wheel_joint_ids)

    # ---------- observation ----------
    def _get_observations(self) -> dict:
        # RGB image: TiledCamera returns NHWC uint8 in output["rgb"]. Normalize.
        rgb = self._camera.data.output["rgb"].float() / 255.0  # (N, H, W, 3)

        # Per-env visual augmentation. Random brightness shift, random
        # contrast factor, additive Gaussian noise. Each per-env scalar
        # is broadcast over the spatial dims.
        n = rgb.shape[0]
        brightness = (torch.rand(n, 1, 1, 1, device=self.device) * 2 - 1) * self.cfg.aug_brightness
        contrast = 1.0 + (torch.rand(n, 1, 1, 1, device=self.device) * 2 - 1) * self.cfg.aug_contrast
        noise = torch.randn_like(rgb) * self.cfg.aug_noise_std
        rgb = ((rgb - 0.5) * contrast + 0.5 + brightness + noise).clamp(0.0, 1.0)

        # Proprio: linear vel (xy body), yaw rate, last action. Shape (N, P).
        root_lin_vel_b = self._robot.data.root_lin_vel_b[:, :2]    # (N, 2)
        root_ang_vel_b = self._robot.data.root_ang_vel_b[:, 2:3]   # (N, 1) yaw
        proprio = torch.cat([root_lin_vel_b, root_ang_vel_b, self._last_action], dim=-1)
        assert proprio.shape[1] == PROPRIO_DIM

        # Broadcast proprio onto spatial planes → (N, H, W, P), concat as channels.
        proprio_planes = proprio.view(n, 1, 1, PROPRIO_DIM).expand(n, IMAGE_HEIGHT, IMAGE_WIDTH, PROPRIO_DIM)
        obs = torch.cat([rgb, proprio_planes], dim=-1)  # (N, H, W, 3 + P)

        return {"policy": obs}

    # ---------- reward ----------
    def _get_rewards(self) -> torch.Tensor:
        fwd_vel = self._robot.data.root_lin_vel_b[:, 0]
        lat_vel = self._robot.data.root_lin_vel_b[:, 1].abs()

        # Tilt from upright via body-z dotted with world-z.
        quat = self._robot.data.root_quat_w  # (w, x, y, z)
        bz_z = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        tilt = torch.acos(bz_z.clamp(-1.0, 1.0))

        action_rate = torch.sum((self._action_buf - self._last_action) ** 2, dim=-1)

        contact_forces = self._contact.data.net_forces_w_history  # (N, T, B, 3)
        max_force = contact_forces.norm(dim=-1).amax(dim=(1, 2))
        in_collision = (max_force > self.cfg.collision_force_threshold).float()

        rew = (
            self.cfg.rew_forward_vel * fwd_vel
            + self.cfg.rew_alive
            + self.cfg.pen_action_rate * action_rate
            + self.cfg.pen_lateral_vel * lat_vel
            + self.cfg.pen_tilt * tilt
            + self.cfg.pen_collision * in_collision
        )

        self._last_action = self._action_buf.clone()
        return rew

    # ---------- termination ----------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        quat = self._robot.data.root_quat_w
        bz_z = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        tilt = torch.acos(bz_z.clamp(-1.0, 1.0))
        flipped = tilt > self.cfg.tilt_limit_rad

        return flipped, time_out

    # ---------- reset ----------
    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        n_reset = len(env_ids)
        env_origins = self.scene.env_origins[env_ids]  # (n, 3)

        # --- robot pose: origin of each env, random yaw ---
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += env_origins

        yaw = (torch.rand(n_reset, device=self.device) * 2 - 1) * math.pi
        default_root_state[:, 3] = torch.cos(yaw / 2)  # qw
        default_root_state[:, 4] = 0.0
        default_root_state[:, 5] = 0.0
        default_root_state[:, 6] = torch.sin(yaw / 2)  # qz

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # --- obstacle re-randomization ---
        self._reset_obstacles(env_ids, env_origins)

        # --- physics domain randomization ---
        self._randomize_physics(env_ids)

        self._last_action[env_ids] = 0.0
        self._action_buf[env_ids] = 0.0

    # ---------- helpers ----------
    def _reset_obstacles(self, env_ids: torch.Tensor, env_origins: torch.Tensor) -> None:
        """Scatter cubes uniformly inside a square around each env origin,
        rejecting positions inside a keep-out radius around the robot."""
        n = len(env_ids)
        k = NUM_OBSTACLES_PER_ENV
        a = self.cfg.obstacle_area_half
        keep_out = self.cfg.obstacle_min_robot_dist

        # Sample positions; resample any that land too close to (0,0) in env-local frame.
        local_xy = (torch.rand(n, k, 2, device=self.device) * 2 - 1) * a
        too_close = local_xy.norm(dim=-1) < keep_out
        # Push too-close boxes radially outward to the keep-out boundary.
        if too_close.any():
            dirs = local_xy[too_close]
            norms = dirs.norm(dim=-1, keepdim=True).clamp(min=1e-3)
            local_xy[too_close] = dirs / norms * keep_out

        world_xy = local_xy + env_origins[:, None, :2]            # (n, k, 2)
        z = torch.full((n, k, 1), 0.15, device=self.device)        # half-cube above ground
        pos = torch.cat([world_xy, z], dim=-1)                     # (n, k, 3)

        # Random yaw per box.
        yaw = (torch.rand(n, k, device=self.device) * 2 - 1) * math.pi
        quat = torch.zeros(n, k, 4, device=self.device)
        quat[..., 0] = torch.cos(yaw / 2)
        quat[..., 3] = torch.sin(yaw / 2)

        # RigidObjectCollection.write_object_state expects (n_env, n_obj, 13):
        # [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        state = torch.zeros(n, k, 13, device=self.device)
        state[..., 0:3] = pos
        state[..., 3:7] = quat
        # vel left at zero

        # The collection writes by env index, not by object index.
        self._obstacles.write_object_link_pose_to_sim(state[..., 0:7], env_ids=env_ids)
        self._obstacles.write_object_com_velocity_to_sim(state[..., 7:13], env_ids=env_ids)

    def _randomize_physics(self, env_ids: torch.Tensor) -> None:
        """Per-env friction, mass, wheel damping."""
        n = len(env_ids)

        # Friction: dynamic & static drawn from the same range.
        f_lo, f_hi = self.cfg.randomize_friction_range
        materials = self._robot.root_physx_view.get_material_properties()  # (num_envs, num_shapes, 3)
        sampled = torch.rand(n, materials.shape[1], 2, device=materials.device) * (f_hi - f_lo) + f_lo
        materials[env_ids, :, 0] = sampled[..., 0]   # static
        materials[env_ids, :, 1] = sampled[..., 1]   # dynamic
        # restitution col 2 — leave alone
        self._robot.root_physx_view.set_material_properties(materials, env_ids.cpu())

        # Mass: additive perturbation on base_link (index 0 is the root link).
        m_lo, m_hi = self.cfg.randomize_mass_range_kg
        masses = self._nominal_masses.clone()
        delta = torch.rand(n, device=masses.device) * (m_hi - m_lo) + m_lo
        masses[env_ids, 0] += delta
        self._robot.root_physx_view.set_masses(masses, env_ids.cpu())

        # Wheel actuator damping. Direct env's actuators live in self._robot.actuators.
        d_lo, d_hi = self.cfg.randomize_wheel_damping_range
        for actuator in self._robot.actuators.values():
            new_damp = torch.rand(n, len(actuator.joint_indices), device=self.device) * (d_hi - d_lo) + d_lo
            actuator.damping[env_ids] = new_damp
