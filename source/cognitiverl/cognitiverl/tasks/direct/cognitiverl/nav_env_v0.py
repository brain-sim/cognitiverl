from __future__ import annotations

from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .nav_env_cfg import NavEnvCfg


class NavEnv(DirectRLEnv):
    cfg: NavEnvCfg

    def __init__(self, cfg: NavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._throttle_dof_idx, _ = self.robot.find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.robot.find_joints(self.cfg.steering_dof_name)
        self._throttle_state = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=torch.float32
        )
        self._steering_state = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=torch.float32
        )
        self._goal_reached = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.int32
        )
        self.task_completed = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.bool
        )
        self._num_goals = cfg.num_goals
        self._target_positions = torch.zeros(
            (self.num_envs, self._num_goals, 2), device=self.device, dtype=torch.float32
        )
        self._markers_pos = torch.zeros(
            (self.num_envs, self._num_goals, 3), device=self.device, dtype=torch.float32
        )
        self.env_spacing = self.cfg.env_spacing
        self.course_length_coefficient = self.cfg.course_length_coefficient
        self.course_width_coefficient = self.cfg.course_width_coefficient
        self.position_tolerance = self.cfg.position_tolerance
        self.goal_reached_bonus = self.cfg.goal_reached_bonus
        self.position_progress_weight = self.cfg.position_progress_weight
        self.heading_coefficient = self.cfg.heading_coefficient
        self.heading_progress_weight = self.cfg.heading_progress_weight
        self.wall_penalty_weight = self.cfg.wall_penalty_weight
        self.linear_speed_weight = self.cfg.linear_speed_weight
        self.laziness_penalty_weight = self.cfg.laziness_penalty_weight
        self._target_index = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.int32
        )
        self._accumulated_laziness = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float32
        )
        self.laziness_decay = self.cfg.laziness_decay
        self.laziness_threshold = self.cfg.laziness_threshold
        self.max_laziness = self.cfg.max_laziness

    def _setup_scene(self):
        # Create a large ground plane without grid
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                size=(500.0, 500.0),  # Much larger ground plane (500m x 500m)
                color=(0.2, 0.2, 0.2),  # Dark gray color
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )

        # Setup rest of the scene
        self.robot = Articulation(self.cfg.robot_cfg)
        self.waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)
        self.object_state = []

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot

        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        throttle_scale = self.cfg.throttle_scale
        throttle_max = self.cfg.throttle_max
        steering_scale = self.cfg.steering_scale
        steering_max = self.cfg.steering_max

        self._throttle_action = (
            actions[:, 0].repeat_interleave(4).reshape((-1, 4)) * throttle_scale
        )
        self.throttle_action = torch.clamp(
            self._throttle_action, -throttle_max, throttle_max
        )
        self._throttle_state = self._throttle_action

        self._steering_action = (
            actions[:, 1].repeat_interleave(2).reshape((-1, 2)) * steering_scale
        )
        self._steering_action = torch.clamp(
            self._steering_action, -steering_max, steering_max
        )
        self._steering_state = self._steering_action

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(
            self._throttle_action, joint_ids=self._throttle_dof_idx
        )
        self.robot.set_joint_position_target(
            self._steering_state, joint_ids=self._steering_dof_idx
        )

    def _get_observations(self) -> dict:
        current_target_positions = self._target_positions[
            self.robot._ALL_INDICES, self._target_index
        ]
        self._position_error_vector = (
            current_target_positions - self.robot.data.root_pos_w[:, :2]
        )
        self._previous_position_error = self._position_error.clone()
        self._position_error = torch.norm(self._position_error_vector, dim=-1)

        heading = self.robot.data.heading_w
        target_heading_w = torch.atan2(
            self._target_positions[self.robot._ALL_INDICES, self._target_index, 1]
            - self.robot.data.root_link_pos_w[:, 1],
            self._target_positions[self.robot._ALL_INDICES, self._target_index, 0]
            - self.robot.data.root_link_pos_w[:, 0],
        )
        self.target_heading_error = torch.atan2(
            torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading)
        )

        obs = torch.cat(
            (
                self._position_error.unsqueeze(dim=1),
                torch.cos(self.target_heading_error).unsqueeze(dim=1),
                torch.sin(self.target_heading_error).unsqueeze(dim=1),
                self.robot.data.root_lin_vel_b[:, 0].unsqueeze(dim=1),
                self.robot.data.root_lin_vel_b[:, 1].unsqueeze(dim=1),
                self.robot.data.root_ang_vel_w[:, 2].unsqueeze(dim=1),
                self._throttle_state[:, 0].unsqueeze(dim=1),
                self._steering_state[:, 0].unsqueeze(dim=1),
                self._get_distance_to_walls().unsqueeze(dim=1),
            ),
            dim=-1,
        )

        if torch.any(obs.isnan()):
            raise ValueError("Observations cannot be NAN")

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        position_progress_reward = torch.nan_to_num(
            self.position_progress_weight
            * (self._previous_position_error - self._position_error),
            posinf=0.0,
            neginf=0.0,
        )
        target_heading_reward = torch.nan_to_num(
            self.heading_progress_weight
            * torch.exp(
                -torch.abs(self.target_heading_error) / self.heading_coefficient
            ),
            posinf=0.0,
            neginf=0.0,
        )
        goal_reached_reward = self.goal_reached_bonus * torch.nan_to_num(
            torch.where(
                self._position_error < self.position_tolerance,
                1.0,
                torch.zeros_like(self._position_error),
            ),
            posinf=0.0,
            neginf=0.0,
        )

        goal_reached = self._position_error < self.position_tolerance
        self._target_index = self._target_index + goal_reached
        self.task_completed = self._target_index > (self._num_goals - 1)
        self._target_index = self._target_index % self._num_goals

        linear_speed = torch.norm(self.robot.data.root_lin_vel_b[:, :2], dim=-1)
        current_laziness = torch.where(
            linear_speed < self.laziness_threshold,
            torch.ones_like(linear_speed),
            torch.zeros_like(linear_speed),
        )

        self._accumulated_laziness = (
            self._accumulated_laziness * self.laziness_decay + current_laziness
        )
        self._accumulated_laziness = torch.clamp(
            self._accumulated_laziness, 0.0, self.max_laziness
        )

        laziness_penalty = torch.nan_to_num(
            -self.laziness_penalty_weight * torch.log1p(self._accumulated_laziness),
            posinf=0.0,
            neginf=0.0,
        )

        self._accumulated_laziness = torch.where(
            goal_reached,
            torch.zeros_like(self._accumulated_laziness),
            self._accumulated_laziness,
        )

        min_wall_dist = self._get_distance_to_walls()
        danger_distance = self.cfg.wall_thickness / 2 + 2.0
        wall_penalty = torch.where(
            min_wall_dist > danger_distance,
            torch.zeros_like(min_wall_dist),
            -self.wall_penalty_weight
            * torch.exp(1.0 - min_wall_dist / danger_distance),
        )
        linear_speed_reward = self.linear_speed_weight * torch.nan_to_num(
            linear_speed / (self.target_heading_error + 1e-8),
            posinf=0.0,
            neginf=0.0,
        )

        composite_reward = (
            position_progress_reward
            + target_heading_reward
            + goal_reached_reward
            + linear_speed_reward
            + laziness_penalty
            + wall_penalty
        )

        one_hot_encoded = torch.nn.functional.one_hot(
            self._target_index.long(), num_classes=self._num_goals
        )
        marker_indices = one_hot_encoded.view(-1).tolist()
        self.waypoints.visualize(marker_indices=marker_indices)

        if torch.any(composite_reward.isnan()):
            raise ValueError("Rewards cannot be NAN")

        return composite_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        task_failed = self.episode_length_buf > self.max_episode_length
        return task_failed, self.task_completed

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)
        default_state = self.robot.data.default_root_state[env_ids]
        robot_pose = default_state[:, :7]
        robot_velocities = default_state[:, 7:]
        joint_positions = self.robot.data.default_joint_pos[env_ids]
        joint_velocities = self.robot.data.default_joint_vel[env_ids]

        robot_pose[:, :3] += self.scene.env_origins[env_ids]
        robot_pose[:, 0] -= self.env_spacing / 2
        robot_pose[:, 1] += (
            2.0
            * torch.rand((num_reset), dtype=torch.float32, device=self.device)
            * self.course_width_coefficient
        )

        angles = (
            torch.pi
            / 6.0
            * torch.rand((num_reset), dtype=torch.float32, device=self.device)
        )
        robot_pose[:, 3] = torch.cos(angles * 0.5)
        robot_pose[:, 6] = torch.sin(angles * 0.5)

        self.robot.write_root_pose_to_sim(robot_pose, env_ids)
        self.robot.write_root_velocity_to_sim(robot_velocities, env_ids)
        self.robot.write_joint_state_to_sim(
            joint_positions, joint_velocities, None, env_ids
        )

        self._target_positions[env_ids, :, :] = 0.0
        self._markers_pos[env_ids, :, :] = 0.0

        spacing = 2 / self._num_goals
        target_positions = (
            torch.arange(-0.8, 1.1, spacing, device=self.device)
            * self.env_spacing
            / self.course_length_coefficient
        )
        self._target_positions[env_ids, : len(target_positions), 0] = target_positions
        self._target_positions[env_ids, :, 1] = (
            torch.rand(
                (num_reset, self._num_goals), dtype=torch.float32, device=self.device
            )
            + self.course_length_coefficient
        )
        self._target_positions[env_ids, :] += self.scene.env_origins[
            env_ids, :2
        ].unsqueeze(1)

        self._target_index[env_ids] = 0
        self._markers_pos[env_ids, :, :2] = self._target_positions[env_ids]
        visualize_pos = self._markers_pos.view(-1, 3)
        self.waypoints.visualize(translations=visualize_pos)

        current_target_positions = self._target_positions[
            self.robot._ALL_INDICES, self._target_index
        ]
        self._position_error_vector = (
            current_target_positions[:, :2] - self.robot.data.root_pos_w[:, :2]
        )
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        self._previous_position_error = self._position_error.clone()

        heading = self.robot.data.heading_w[:]
        target_heading_w = torch.atan2(
            self._target_positions[:, 0, 1] - self.robot.data.root_pos_w[:, 1],
            self._target_positions[:, 0, 0] - self.robot.data.root_pos_w[:, 0],
        )
        self._heading_error = torch.atan2(
            torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading)
        )
        self._previous_heading_error = self._heading_error.clone()

    def _get_distance_to_walls(self):
        robot_positions = self.robot.data.root_pos_w[:, :2]
        env_origins = self.scene.env_origins[:, :2]

        relative_positions = robot_positions - env_origins

        wall_position = self.cfg.env_spacing / 2

        north_dist = wall_position - relative_positions[:, 1]
        south_dist = wall_position + relative_positions[:, 1]
        east_dist = wall_position - relative_positions[:, 0]
        west_dist = wall_position + relative_positions[:, 0]

        wall_distances = torch.stack(
            [north_dist, south_dist, east_dist, west_dist], dim=1
        )
        min_wall_distance = torch.min(wall_distances, dim=1)[0]

        return min_wall_distance
