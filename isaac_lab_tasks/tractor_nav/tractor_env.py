"""DirectRLEnv implementation for Isaac-Tractor-Nav-v0.

Pure-exploration policy: drive forward, don't crash, don't flip, don't oscillate.
No goal conditioning. Observation = 128x72 RGB + proprio (lin vel, ang vel, prev action).
Action = 4D wheel velocity command in [-1, 1].
"""

from __future__ import annotations

import torch
import gymnasium as gym

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, ContactSensor

from .tractor_env_cfg import TractorNavEnvCfg


class TractorNavEnv(DirectRLEnv):
    cfg: TractorNavEnvCfg

    def __init__(self, cfg: TractorNavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint indices for the 4 wheels. Names come from tractor_isaac.urdf.
        self._wheel_joint_ids, _ = self._robot.find_joints(
            ["front_left_wheel_joint", "front_right_wheel_joint",
             "rear_left_wheel_joint", "rear_right_wheel_joint"]
        )
        self._last_action = torch.zeros(self.num_envs, 4, device=self.device)
        self._action_buf = torch.zeros(self.num_envs, 4, device=self.device)

    # ---------- scene setup ----------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.scene.robot)
        self._camera = Camera(self.cfg.scene.camera)
        self._contact = ContactSensor(self.cfg.scene.contact_sensor)
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["camera"] = self._camera
        self.scene.sensors["contact"] = self._contact
        self.cfg.scene.terrain.func(self.cfg.scene.terrain.prim_path, self.cfg.scene.terrain)
        self.cfg.scene.dome_light.func("/World/Light", self.cfg.scene.dome_light)
        self.scene.clone_environments(copy_from_source=False)

    # ---------- action ----------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # actions: (num_envs, 4) in [-1, 1] → wheel velocity targets
        self._action_buf = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        vel_targets = self._action_buf * self.cfg.max_wheel_vel
        self._robot.set_joint_velocity_target(vel_targets, joint_ids=self._wheel_joint_ids)

    # ---------- observation ----------
    def _get_observations(self) -> dict:
        # RGB image: (num_envs, H, W, 3) uint8 → float [0, 1], NCHW
        rgb = self._camera.data.output["rgb"].float() / 255.0
        rgb = rgb.permute(0, 3, 1, 2)  # → (N, 3, 72, 128)

        # Proprio: linear vel (xy, body frame), yaw rate, last action
        root_lin_vel_b = self._robot.data.root_lin_vel_b[:, :2]   # (N, 2)
        root_ang_vel_b = self._robot.data.root_ang_vel_b[:, 2:3]  # (N, 1) yaw
        proprio = torch.cat([root_lin_vel_b, root_ang_vel_b, self._last_action], dim=-1)

        return {"policy": {"image": rgb, "proprio": proprio}}

    # ---------- reward ----------
    def _get_rewards(self) -> torch.Tensor:
        # Forward velocity in body frame (x = forward)
        fwd_vel = self._robot.data.root_lin_vel_b[:, 0]
        lat_vel = self._robot.data.root_lin_vel_b[:, 1].abs()

        # Tilt: angle between body z-axis and world z-axis
        # root_quat_w is (w, x, y, z) in Isaac Lab
        quat = self._robot.data.root_quat_w
        # body z-axis in world frame, z-component:
        bz_z = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        tilt = torch.acos(bz_z.clamp(-1.0, 1.0))  # rad from upright

        # Action smoothness
        action_rate = torch.sum((self._action_buf - self._last_action) ** 2, dim=-1)

        # Collision: any contact force above threshold
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
        # Time-out
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Flipped
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

        # Reset robot to origin of each env with small random yaw
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Random yaw in [-pi, pi]
        yaw = (torch.rand(len(env_ids), device=self.device) * 2 - 1) * 3.14159
        qw = torch.cos(yaw / 2)
        qz = torch.sin(yaw / 2)
        default_root_state[:, 3] = qw
        default_root_state[:, 4] = 0.0
        default_root_state[:, 5] = 0.0
        default_root_state[:, 6] = qz

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset wheel velocities
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        self._last_action[env_ids] = 0.0
        self._action_buf[env_ids] = 0.0
