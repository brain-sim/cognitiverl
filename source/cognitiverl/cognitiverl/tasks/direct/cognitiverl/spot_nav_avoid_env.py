from __future__ import annotations

import os
from collections.abc import Sequence

import torch
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from termcolor import colored

from .nav_env import NavEnv
from .spot_nav_avoid_env_cfg import SpotNavAvoidEnvCfg
from .spot_policy_controller import SpotPolicyController


class SpotNavAvoidEnv(NavEnv):
    cfg: SpotNavAvoidEnvCfg
    ACTION_SCALE = 0.2  # Scale for policy output (delta from default pose)

    def __init__(
        self,
        cfg: SpotNavAvoidEnvCfg,
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

        # Add avoid goal collision tracking
        self._episode_avoid_collisions = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.int32
        )
        self._avoid_goal_hit_this_step = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.bool
        )

    def _setup_robot_dof_idx(self):
        self._dof_idx, _ = self.robot.find_joints(self.cfg.dof_name)

    def _setup_config(self):
        # --- Low-level Spot policy integration ---
        # TODO: Set the correct path to your TorchScript policy file
        policy_file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "custom_assets",
            self.cfg.policy_file_path,
        )
        print(
            colored(
                f"[INFO] Loading policy from {policy_file_path}",
                "magenta",
                attrs=["bold"],
            )
        )
        self.policy = SpotPolicyController(policy_file_path)
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
            (self.num_envs,), device=self.device, dtype=torch.int32
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
        self._smoothing_factor = torch.tensor([0.75, 0.3, 0.3], device=self.device)
        self.max_episode_length_buf = torch.full(
            (self.num_envs,), self.max_episode_length, device=self.device
        )

        # Add avoid penalty weight
        self.avoid_penalty_weight = self.cfg.avoid_penalty_weight
        self.fast_goal_reached_bonus = self.cfg.fast_goal_reached_weight

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
            self._actions[:, 0], min=0.0, max=self.throttle_max
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
        lin_vel_I = self.robot.data.root_lin_vel_w  # [num_envs, 3]
        ang_vel_I = self.robot.data.root_ang_vel_w  # [num_envs, 3]
        q_IB = self.robot.data.root_quat_w  # [num_envs, 4]
        joint_pos = self.robot.data.joint_pos  # [num_envs, 12]
        joint_vel = self.robot.data.joint_vel  # [num_envs, 12]
        # Compute actions for all environments
        actions = self.policy.get_action(
            lin_vel_I,
            ang_vel_I,
            q_IB,
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

    # def _check_stuck_termination(self, max_steps: int = 300) -> torch.Tensor:
    #     """Early termination if robot is stuck/wandering without progress"""
    #     # If no goal reached in last max_steps and barely moving, terminate
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

        # Check for avoid goal collisions (future waypoints) - VECTORIZED VERSION
        robot_positions = self.robot.data.root_pos_w[:, :2]  # (num_envs, 2)
        avoid_penalty = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float32
        )

        # Reset step-level collision tracker
        self._avoid_goal_hit_this_step.fill_(False)

        # Create mask for future waypoints (goals with index > current target index)
        goal_indices = torch.arange(self._num_goals, device=self.device).unsqueeze(
            0
        )  # (1, num_goals)
        target_indices = self._target_index.unsqueeze(1)  # (num_envs, 1)
        future_waypoint_mask = goal_indices > target_indices  # (num_envs, num_goals)

        # Calculate distances from each robot to all waypoints
        # robot_positions: (num_envs, 2) -> (num_envs, 1, 2)
        # _target_positions: (num_envs, num_goals, 2)
        robot_pos_expanded = robot_positions.unsqueeze(1)  # (num_envs, 1, 2)
        distances = torch.norm(
            robot_pos_expanded - self._target_positions, dim=2
        )  # (num_envs, num_goals)

        # Apply future waypoint mask and check for collisions
        future_distances = (
            distances * future_waypoint_mask.float()
        )  # Zero out non-future waypoints
        future_distances = torch.where(
            future_waypoint_mask,
            future_distances,
            torch.full_like(future_distances, float("inf")),  # Set non-future to inf
        )

        # Check which environments have collisions with future waypoints
        collision_mask = (
            future_distances < self.position_tolerance
        )  # (num_envs, num_goals)
        env_has_collision = collision_mask.any(dim=1)  # (num_envs,)

        # Apply penalties for environments with collisions
        avoid_penalty[env_has_collision] = self.avoid_penalty_weight
        self._avoid_goal_hit_this_step[env_has_collision] = True

        # Count collisions per environment (sum of collision_mask per env)
        collisions_per_env = collision_mask.sum(dim=1)  # (num_envs,)
        self._episode_avoid_collisions += collisions_per_env

        self._target_index = self._target_index + goal_reached
        self._episode_waypoints_passed += goal_reached.int()
        self.task_completed = self._target_index > (self._num_goals - 1)
        self._target_index = self._target_index % self._num_goals
        assert (
            self._previous_waypoint_reached_step[goal_reached]
            < self.episode_length_buf[goal_reached]
        ).all(), "Previous waypoint reached step is greater than episode length"
        # Compute k using torch.log
        k = torch.log(torch.tensor(self.fast_goal_reached_bonus, device=self.device)) / (
            self.max_episode_length - 1
        )
        steps_taken = self.episode_length_buf - self._previous_waypoint_reached_step
        fast_goal_reached_reward = torch.where(
            goal_reached,
            self.fast_goal_reached_bonus * torch.exp(-k * (steps_taken - 1)),
            torch.zeros_like(self._previous_waypoint_reached_step),
        )
        fast_goal_reached_reward = torch.clamp(
            fast_goal_reached_reward, min=0.0, max=self.fast_goal_reached_bonus
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
            -self.laziness_penalty_weight * torch.log1p(self._accumulated_laziness),
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
                -self.wall_penalty_weight
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
            "Episode_Reward/avoid_penalty": avoid_penalty,
        }

    def _generate_waypoints_with_spacing(self, env_ids, robot_poses, min_spacing=3.0):
        """Generate waypoints ensuring minimum spacing between them and from robot spawn - VECTORIZED."""
        num_reset = len(env_ids)
        env_origins = self.scene.env_origins[env_ids, :2]  # (num_reset, 2)
        robot_xy = robot_poses[:, :2]  # (num_reset, 2)

        # Initialize waypoint positions
        waypoint_positions = torch.zeros(
            (num_reset, self._num_goals, 2), device=self.device
        )

        # Constrain waypoints to 60% of distance from center to wall
        max_distance_from_center = 0.6 * (self.wall_position - self.wall_thickness / 2)

        # Generate waypoints sequentially (to maintain spacing constraints)
        for goal_idx in range(self._num_goals):
            max_attempts = 100
            placed = torch.zeros(num_reset, dtype=torch.bool, device=self.device)

            for attempt in range(max_attempts):
                # Generate candidates for all unplaced environments
                unplaced_mask = ~placed
                num_unplaced = unplaced_mask.sum().item()

                if num_unplaced == 0:
                    break

                # Generate random positions for unplaced environments
                tx = (
                    torch.rand(num_unplaced, device=self.device)
                    * 2
                    * max_distance_from_center
                    - max_distance_from_center
                )
                ty = (
                    torch.rand(num_unplaced, device=self.device)
                    * 2
                    * max_distance_from_center
                    - max_distance_from_center
                )

                # Convert to world coordinates
                unplaced_origins = env_origins[unplaced_mask]  # (num_unplaced, 2)
                candidate_positions = (
                    torch.stack([tx, ty], dim=1) + unplaced_origins
                )  # (num_unplaced, 2)

                # Check distance from robot (minimum 2.5m)
                unplaced_robot_pos = robot_xy[unplaced_mask]  # (num_unplaced, 2)
                robot_distances = torch.norm(
                    candidate_positions - unplaced_robot_pos, dim=1
                )  # (num_unplaced,)
                robot_valid = robot_distances >= 2.5

                # Check distance from previously placed waypoints in same environment
                waypoint_valid = torch.ones(
                    num_unplaced, dtype=torch.bool, device=self.device
                )
                if goal_idx > 0:
                    # Get previously placed waypoints for unplaced environments
                    prev_waypoints = waypoint_positions[
                        unplaced_mask, :goal_idx, :
                    ]  # (num_unplaced, goal_idx, 2)

                    # Calculate distances to all previous waypoints
                    candidate_expanded = candidate_positions.unsqueeze(
                        1
                    )  # (num_unplaced, 1, 2)
                    distances_to_prev = torch.norm(
                        candidate_expanded - prev_waypoints, dim=2
                    )  # (num_unplaced, goal_idx)

                    # Check if any distance is too small
                    min_distances = distances_to_prev.min(dim=1)[0]  # (num_unplaced,)
                    waypoint_valid = min_distances >= min_spacing

                # Combine all validity checks
                valid = robot_valid & waypoint_valid

                # Update positions for valid placements
                valid_indices = torch.where(unplaced_mask)[0][valid]
                if len(valid_indices) > 0:
                    waypoint_positions[valid_indices, goal_idx, :] = (
                        candidate_positions[valid]
                    )
                    placed[valid_indices] = True

            # Fallback for any remaining unplaced waypoints
            if not placed.all():
                unplaced_mask = ~placed
                num_unplaced = unplaced_mask.sum().item()

                # Use relaxed constraints - smaller area but guaranteed placement
                tx = (
                    torch.rand(num_unplaced, device=self.device)
                    * max_distance_from_center
                    - max_distance_from_center / 2
                )
                ty = (
                    torch.rand(num_unplaced, device=self.device)
                    * max_distance_from_center
                    - max_distance_from_center / 2
                )

                unplaced_origins = env_origins[unplaced_mask]
                fallback_positions = torch.stack([tx, ty], dim=1) + unplaced_origins

                unplaced_indices = torch.where(unplaced_mask)[0]
                waypoint_positions[unplaced_indices, goal_idx, :] = fallback_positions

        return waypoint_positions

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self.camera.reset(env_ids)
        if self.play_mode:
            self.max_episode_length_buf[env_ids] = self.max_episode_length
        else:
            min_episode_length = min(
                200 + self.common_step_counter, int(0.7 * self.max_episode_length)
            )
            self.max_episode_length_buf[env_ids] = torch.randint(
                min_episode_length,
                self.max_episode_length + 1,
                (len(env_ids),),
                device=self.device,
            )

        self._episode_reward_buf[env_ids] = 0.0
        self._episode_waypoints_passed[env_ids] = 0
        # Reset avoid collision counter
        self._episode_avoid_collisions[env_ids] = 0
        if hasattr(self, "_previous_waypoint_reached_step"):
            self._previous_waypoint_reached_step[env_ids] = 0

        num_reset = len(env_ids)
        default_state = self.robot.data.default_root_state[env_ids]
        robot_pose = default_state[:, :7]
        robot_velocities = default_state[:, 7:]
        joint_positions = self.robot.data.default_joint_pos[env_ids]
        joint_velocities = self.robot.data.default_joint_vel[env_ids]

        robot_pose[:, :3] += self.scene.env_origins[env_ids]

        # CHANGE: Set car position to be randomly inside the room rather than outside of it
        # Use smaller margins to keep car away from walls
        safe_room_size = self.room_size // 2

        # Random position inside the room with margin
        robot_pose[:, 0] += (
            torch.rand(num_reset, dtype=torch.float32, device=self.device)
            * safe_room_size
            - safe_room_size / 2
        )
        robot_pose[:, 1] += (
            torch.rand(num_reset, dtype=torch.float32, device=self.device)
            * safe_room_size
            - safe_room_size / 2
        )

        # Keep random rotation for variety
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

        # Generate waypoints with proper spacing
        waypoint_positions = self._generate_waypoints_with_spacing(
            env_ids, robot_pose, min_spacing=3.0
        )
        self._target_positions[env_ids] = waypoint_positions

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
