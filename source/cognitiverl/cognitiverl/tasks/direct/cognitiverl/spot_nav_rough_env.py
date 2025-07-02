from __future__ import annotations

import os

import torch
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from termcolor import colored

from .nav_env import NavEnv
from .spot_nav_rough_env_cfg import SpotNavRoughEnvCfg
from .spot_policy_controller import SpotRoughPolicyController


class SpotNavRoughEnv(NavEnv):
    cfg: SpotNavRoughEnvCfg
    ACTION_SCALE = 0.2  # Scale for policy output (delta from default pose)

    def __init__(
        self,
        cfg: SpotNavRoughEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        # Add room size as a class attribute
        self._goal_reached = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.int32
        )
        self.task_completed = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.bool
        )
        self._target_positions = torch.zeros(
            (self.num_envs, self._num_goals, 2),
            device=self.device,
            dtype=torch.float32,
        )
        self._markers_pos = torch.zeros(
            (self.num_envs, self._num_goals, 3),
            device=self.device,
            dtype=torch.float32,
        )
        self.env_spacing = self.cfg.env_spacing
        self._target_index = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.int32
        )

        # Add accumulated laziness tracker
        self._accumulated_laziness = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float32
        )

    def _setup_robot_dof_idx(self):
        self._dof_idx, _ = self.robot.find_joints(self.cfg.dof_name)

    # def _setup_plane(self):
    #     # Create robust rough terrain configuration with safe parameters
    #     terrain_cfg = TerrainImporterCfg(
    #         prim_path="/World/ground",
    #         terrain_type="generator",
    #         terrain_generator=terrain_gen.TerrainGeneratorCfg(
    #             size=(1000.0, 1000.0),  # Size of terrain
    #             border_width=1.0,  # Safe border width (>0 to avoid division issues)
    #             num_rows=1,  # Single terrain patch
    #             num_cols=1,  # Single terrain patch
    #             horizontal_scale=1.0,  # Safe horizontal resolution (>=1.0)
    #             vertical_scale=0.01,  # Safe vertical resolution (>=0.1)
    #             slope_threshold=1.0,  # Safe slope threshold (well above 0)
    #             use_cache=True,  # Enable caching for performance
    #             sub_terrains={
    #                 "rough_terrain": terrain_gen.HfRandomUniformTerrainCfg(
    #                     proportion=1.0,
    #                     noise_range=(0.05, 0.15),  # Safe noise range (min >= 0.05)
    #                     noise_step=0.05,  # Safe step size (>= 0.05)
    #                     border_width=1.0,  # Match outer border width
    #                 ),
    #             },
    #         ),
    #         max_init_terrain_level=0,  # Single level to avoid complexity
    #         debug_vis=False,
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             friction_combine_mode="multiply",
    #             restitution_combine_mode="multiply",
    #             static_friction=max(self.cfg.static_friction, 0.1),  # Ensure >= 0.1
    #             dynamic_friction=max(self.cfg.dynamic_friction, 0.1),  # Ensure >= 0.1
    #             restitution=0.0,
    #         ),
    #     )

    #     # Create the terrain importer
    #     terrain_importer = TerrainImporter(terrain_cfg)
    #     self._terrain_importer = terrain_importer

    def _setup_config(self):
        # --- Low-level Spot policy integration ---
        policy_file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "custom_assets",
            self.cfg.policy_file_path,
        )
        print(colored(f"[INFO] Loading policy from {policy_file_path}", "green"))
        self.policy = SpotRoughPolicyController(policy_file_path)
        # Buffers for previous action and default joint positions
        self._low_level_previous_action = torch.zeros(
            (self.num_envs, 12), device=self.device, dtype=torch.float32
        )
        self._previous_action = torch.zeros(
            (self.num_envs, self.cfg.action_space),
            device=self.device,
            dtype=torch.float32,
        )
        self._previous_waypoint_reached_step = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.int32
        )
        self.position_tolerance = self.cfg.position_tolerance
        self.goal_reached_bonus = self.cfg.goal_reached_bonus
        self.laziness_penalty_weight = self.cfg.laziness_penalty_weight
        self.laziness_decay = (
            self.cfg.laziness_decay
        )  # How much previous laziness carries over
        self.laziness_threshold = (
            self.cfg.laziness_threshold
        )  # Speed threshold for considering "lazy"
        self.max_laziness = (
            self.cfg.max_laziness
        )  # Cap on accumulated laziness to prevent extreme penalties
        self.wall_penalty_weight = self.cfg.wall_penalty_weight
        self.linear_speed_weight = self.cfg.linear_speed_weight
        self.throttle_scale = self.cfg.throttle_scale
        self._actions = torch.zeros(
            (self.num_envs, self.cfg.action_space),
            device=self.device,
            dtype=torch.float32,
        )
        self.steering_scale = self.cfg.steering_scale
        self.throttle_max = self.cfg.throttle_max
        self.steering_max = self.cfg.steering_max
        self._default_pos = self.robot.data.default_joint_pos.clone()
        self._smoothing_factor = torch.tensor([0.7, 0.5, 0.5], device=self.device)
        self.max_episode_length_buf = torch.full(
            (self.num_envs,), self.max_episode_length, device=self.device
        )

    def _setup_camera(self):
        camera_prim_path = "/World/envs/env_.*/Robot/body/Camera"
        pinhole_cfg = PinholeCameraCfg(
            focal_length=16.0,
            horizontal_aperture=32.0,
            vertical_aperture=32.0,
            focus_distance=1.0,
            clipping_range=(0.01, 1000.0),
            lock_camera=True,
        )
        camera_cfg = TiledCameraCfg(
            prim_path=camera_prim_path,
            update_period=self.step_dt,
            height=32,
            width=32,
            data_types=["rgb"],
            spawn=pinhole_cfg,
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.25, 0.0, 0.25),  # At the head, adjust as needed
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
        )
        self.camera = TiledCamera(camera_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = (
            self._smoothing_factor * actions
            + (1 - self._smoothing_factor) * self._previous_action
        )
        self._previous_action = actions.clone()
        self._actions = actions.clone()
        self._actions = torch.nan_to_num(self._actions, 0.0)
        self._actions[:, 0] = self._actions[:, 0] * self.throttle_scale
        self._actions[:, 1:] = self._actions[:, 1:] * self.steering_scale
        self._actions[:, 0] = torch.clamp(
            self._actions[:, 0], min=0, max=self.throttle_max
        )
        self._actions[:, 1:] = torch.clamp(
            self._actions[:, 1:], min=-self.steering_max, max=self.steering_max
        )

    def _apply_action(self) -> None:
        # --- Vectorized low-level Spot policy call for all environments ---
        # Gather all required robot state as torch tensors
        # TODO: Replace the following with actual command logic per environment
        default_pos = self._default_pos.clone()  # [num_envs, 12]
        # The following assumes your robot exposes these as torch tensors of shape [num_envs, ...]
        base_lin_vel = self.robot.data.root_lin_vel_b  # [num_envs, 3]
        base_ang_vel = self.robot.data.root_ang_vel_b  # [num_envs, 3]
        projected_gravity = self.robot.data.projected_gravity_b  # [num_envs, 3]
        joint_pos = self.robot.data.joint_pos  # [num_envs, 12]
        joint_vel = self.robot.data.joint_vel  # [num_envs, 12]
        # Compute actions for all environments
        actions = self.policy.get_action(
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            self._actions,
            self._low_level_previous_action,
            default_pos,
            joint_pos,
            joint_vel,
        )
        # Update previous action buffer
        self._low_level_previous_action = actions.detach()
        # Scale and offset actions as in Spot reference policy
        joint_positions = self._default_pos + actions * self.ACTION_SCALE
        # Apply joint position targets directly
        self.robot.set_joint_position_target(joint_positions)

    def _get_image_obs(self) -> torch.Tensor:
        image_obs = self.camera.data.output["rgb"].float().permute(0, 3, 1, 2) / 255.0
        # image_obs = F.interpolate(image_obs, size=(32, 32), mode='bilinear', align_corners=False)
        image_obs = image_obs.reshape(self.num_envs, -1)
        return image_obs

    def _get_state_obs(self, image_obs) -> torch.Tensor:
        return torch.cat(
            (
                image_obs,
                self._actions[:, 0].unsqueeze(dim=1),
                self._actions[:, 1].unsqueeze(dim=1),
                self._actions[:, 2].unsqueeze(dim=1),
                self._get_distance_to_walls().unsqueeze(dim=1),  # Add wall distance
            ),
            dim=-1,
        )

    # def _check_stuck_termination(self, max_steps: int = 200) -> torch.Tensor:
    #     """Early termination if robot is stuck/wandering without progress"""
    #     # If no goal reached in last 200 steps and barely moving, terminate
    #     steps_since_goal = (
    #         self.episode_length_buf - self._previous_waypoint_reached_step
    #     )
    #     stuck_too_long = steps_since_goal > max_steps
    #     return stuck_too_long

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
        assert (
            self._previous_waypoint_reached_step[goal_reached]
            < self.episode_length_buf[goal_reached]
        ).all(), "Previous waypoint reached step is greater than episode length"
        # max_reward = 125.0
        # epsilon = 1.0  # reward at max episode length
        # max_steps = self.max_episode_length  # should be a scalar

        # Compute k using torch.log
        k = torch.log(torch.tensor(125.0, device=self.device)) / (
            self.max_episode_length - 1
        )
        steps_taken = self.episode_length_buf - self._previous_waypoint_reached_step

        fast_goal_reached_reward = torch.where(
            goal_reached,
            125.0 * torch.exp(-k * (steps_taken - 1)),
            torch.zeros_like(self._previous_waypoint_reached_step),
        )
        fast_goal_reached_reward = torch.clamp(
            fast_goal_reached_reward, min=0.0, max=125.0
        )
        self._previous_waypoint_reached_step = torch.where(
            goal_reached,
            self.episode_length_buf,
            self._previous_waypoint_reached_step,
        )

        # Calculate current laziness based on speed
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
        laziness_penalty = torch.nan_to_num(
            self.laziness_penalty_weight * torch.log1p(self._accumulated_laziness),
            posinf=0.0,
            neginf=0.0,
        )  # log1p(x) = log(1 + x)
        # Add wall distance penalty
        min_wall_dist = self._get_distance_to_walls()
        danger_distance = (
            self.wall_thickness / 2 + 5.0
        )  # Distance at which to start penalizing
        wall_penalty = torch.nan_to_num(
            torch.where(
                min_wall_dist > danger_distance,
                torch.zeros_like(min_wall_dist),
                self.wall_penalty_weight
                * torch.exp(
                    1.0 - min_wall_dist / danger_distance
                ),  # Exponential penalty
            ),
            posinf=0.0,
            neginf=0.0,
        )
        linear_speed_reward = self.linear_speed_weight * torch.nan_to_num(
            linear_speed,
            posinf=0.0,
            neginf=0.0,
        )

        # Create a tensor of 0s (future), 1s (current), and 2s (completed)
        marker_indices = torch.zeros(
            (self.num_envs, self._num_goals),
            device=self.device,
            dtype=torch.long,
        )

        # Set current targets to 1 (green)
        marker_indices[
            torch.arange(self.num_envs, device=self.device), self._target_index
        ] = 1

        # Set completed targets to 2 (invisible)
        # Create a mask for completed targets
        target_mask = (self._target_index.unsqueeze(1) > 0) & (
            torch.arange(self._num_goals, device=self.device)
            < self._target_index.unsqueeze(1)
        )
        marker_indices[target_mask] = 2

        # Original implementation:
        # for env_idx in range(self.num_envs):
        #     target_idx = self._target_index[env_idx].item()
        #     if target_idx > 0:  # If we've passed at least one waypoint
        #         marker_indices[env_idx, :target_idx] = 2

        # Flatten and convert to list
        marker_indices = marker_indices.view(-1).tolist()

        # Update visualizations
        self.waypoints.visualize(marker_indices=marker_indices)

        return {
            "Episode_Reward/goal_reached_reward": goal_reached_reward,
            "Episode_Reward/linear_speed_reward": linear_speed_reward,
            "Episode_Reward/laziness_penalty": laziness_penalty,
            "Episode_Reward/wall_penalty": wall_penalty,
            "Episode_Reward/fast_goal_reached_reward": fast_goal_reached_reward,
        }
