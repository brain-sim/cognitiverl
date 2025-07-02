from __future__ import annotations

import torch
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg

from .leatherback_nav_env_cfg import LeatherbackNavEnvCfg
from .nav_env import NavEnv


class LeatherbackNavEnv(NavEnv):
    cfg: LeatherbackNavEnvCfg

    def _setup_config(self):
        self.position_tolerance = self.cfg.position_tolerance
        self.goal_reached_bonus = self.cfg.goal_reached_bonus
        self.wall_penalty_weight = self.cfg.wall_penalty_weight
        self.linear_speed_weight = self.cfg.linear_speed_weight
        self.laziness_penalty_weight = self.cfg.laziness_penalty_weight
        self.throttle_scale = self.cfg.throttle_scale
        self.throttle_max = self.cfg.throttle_max
        self.steering_scale = self.cfg.steering_scale
        self.steering_max = self.cfg.steering_max
        self.laziness_decay = (
            self.cfg.laziness_decay
        )  # How much previous laziness carries over
        self.laziness_threshold = (
            self.cfg.laziness_threshold
        )  # Speed threshold for considering "lazy"
        self.max_laziness = (
            self.cfg.max_laziness
        )  # Cap on accumulated laziness to prevent extreme penalties

        self._throttle_state = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=torch.float32
        )
        self._steering_state = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=torch.float32
        )
        self._previous_actions = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=torch.float32
        )
        self._smoothing_factor = 0.5

    def _setup_robot_dof_idx(self):
        self._throttle_dof_idx, _ = self.robot.find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.robot.find_joints(self.cfg.steering_dof_name)

    def _setup_camera(self):
        camera_cfg = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/Rigid_Bodies/Chassis/Camera_Left",
            update_period=0.05,
            height=32,
            width=32,
            data_types=["rgb"],
            spawn=None,
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0), rot=(1, 0, 0, 0), convention="ros"
            ),
        )
        self.camera = TiledCamera(camera_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Add velocity smoothing
        actions = (
            self._smoothing_factor * actions
            + (1 - self._smoothing_factor) * self._previous_actions
        )
        self._previous_actions = actions
        self._throttle_action = (
            actions[:, 0].repeat_interleave(4).reshape((-1, 4)) * self.throttle_scale
        )
        self._throttle_action = torch.clamp(
            self._throttle_action, -1, self.throttle_max
        )
        self.throttle_action = self._throttle_action.clone()
        self._throttle_state = self._throttle_action

        self._steering_action = (
            actions[:, 1].repeat_interleave(2).reshape((-1, 2)) * self.steering_scale
        )
        self._steering_action = torch.clamp(
            self._steering_action, -self.steering_max, self.steering_max
        )
        self._steering_state = self._steering_action

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(
            self._throttle_action, joint_ids=self._throttle_dof_idx
        )
        self.robot.set_joint_position_target(
            self._steering_state, joint_ids=self._steering_dof_idx
        )

    def _get_image_obs(self) -> torch.Tensor:
        image_obs = self.camera.data.output["rgb"].float().permute(0, 3, 1, 2) / 255.0
        # image_obs = F.interpolate(image_obs, size=(32, 32), mode='bilinear', align_corners=False)
        image_obs = image_obs.reshape(self.num_envs, -1)
        return image_obs

    def _get_state_obs(self, image_obs) -> torch.Tensor:
        return torch.cat(
            (
                image_obs,
                self._throttle_state[:, 0].unsqueeze(dim=1),
                self._steering_state[:, 0].unsqueeze(dim=1),
                self._get_distance_to_walls().unsqueeze(dim=1),  # Add wall distance
            ),
            dim=-1,
        )

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        goal_reached = self._position_error < self.position_tolerance
        goal_reached_reward = self.goal_reached_bonus * torch.nan_to_num(
            torch.where(
                goal_reached,
                1.0,
                torch.zeros_like(self._position_error),
            ),
            posinf=0.0,
            neginf=0.0,
        )

        self._target_index = self._target_index + goal_reached
        self._episode_waypoints_passed += goal_reached.int()
        self.task_completed = self._target_index > (self._num_goals - 1)
        self._target_index = self._target_index % self._num_goals

        # # Calculate current laziness based on speed
        linear_speed = torch.norm(
            self.robot.data.root_lin_vel_b[:, :2], dim=-1
        )  # XY plane velocity
        current_laziness = torch.where(
            linear_speed < self.laziness_threshold,
            torch.ones_like(linear_speed),  # Count as lazy
            torch.zeros_like(linear_speed),  # Not lazy
        )

        # Update accumulated laziness with decay
        self._accumulated_laziness = (
            self._accumulated_laziness * self.laziness_decay
            + current_laziness * (1 - self.laziness_decay)
        )
        # Clamp to prevent extreme values
        self._accumulated_laziness = torch.clamp(
            self._accumulated_laziness, 0.0, self.max_laziness
        )

        # Reset accumulated laziness when reaching waypoint
        self._accumulated_laziness = torch.where(
            goal_reached,
            torch.zeros_like(self._accumulated_laziness),
            self._accumulated_laziness,
        )

        # Calculate laziness penalty using log
        laziness_penalty = self.laziness_penalty_weight * torch.nan_to_num(
            torch.log1p(self._accumulated_laziness),
            posinf=0.0,
            neginf=0.0,
        )  # log1p(x) = log(1 + x)
        # Add wall distance penalty
        min_wall_dist = self._get_distance_to_walls()
        danger_distance = (
            self.wall_thickness / 2 + 5.0
        )  # Distance at which to start penalizing
        wall_penalty = torch.where(
            min_wall_dist > danger_distance,
            torch.zeros_like(min_wall_dist),
            self.wall_penalty_weight
            * torch.exp(1.0 - min_wall_dist / danger_distance),  # Exponential penalty
        )
        linear_speed_reward = self.linear_speed_weight * torch.nan_to_num(
            linear_speed,  #  / (self.target_heading_error + 1e-8),
            posinf=0.0,
            neginf=0.0,
        )
        # Create a tensor of 0s (future), 1s (current), and 2s (completed)
        marker_indices = torch.zeros(
            (self.num_envs, self._num_goals), device=self.device, dtype=torch.long
        )

        # Set current targets to 1 (green)
        marker_indices[
            torch.arange(self.num_envs, device=self.device), self._target_index
        ] = 1

        # Set completed targets to 2 (invisible)
        for env_idx in range(self.num_envs):
            target_idx = self._target_index[env_idx].item()
            if target_idx > 0:  # If we've passed at least one waypoint
                marker_indices[env_idx, :target_idx] = 2

        # Flatten and convert to list
        marker_indices = marker_indices.view(-1).tolist()

        # Update visualizations
        self.waypoints.visualize(marker_indices=marker_indices)

        return {
            "Episode_Reward/goal_reached_reward": goal_reached_reward,
            "Episode_Reward/linear_speed_reward": linear_speed_reward,
            "Episode_Reward/laziness_penalty": laziness_penalty,
            "Episode_Reward/wall_penalty": wall_penalty,
        }
