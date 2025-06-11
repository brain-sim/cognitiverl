from __future__ import annotations

import torch
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg

from .nav_env import NavEnv
from .spot_nav_env_cfg import SpotNavEnvCfg
from .spot_policy_controller import SpotPolicyController


class SpotNavEnv(NavEnv):
    cfg: SpotNavEnvCfg
    ACTION_SCALE = 0.2  # Scale for policy output (delta from default pose)

    def __init__(
        self,
        cfg: SpotNavEnvCfg,
        render_mode: str | None = None,
        debug: bool = False,
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

        self._debug = debug

    def _setup_robot_dof_idx(self):
        self._dof_idx, _ = self.robot.find_joints(self.cfg.dof_name)

    def _setup_config(self):
        # --- Low-level Spot policy integration ---
        # TODO: Set the correct path to your TorchScript policy file
        policy_file_path = "/home/user/cognitiverl/source/cognitiverl/cognitiverl/tasks/direct/custom_assets/spot_policy.pt"  # <-- Set this to your actual policy file
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
        self.previous_linear_speed = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float32
        )
        self.position_tolerance = self.cfg.position_tolerance
        self.goal_reached_bonus = self.cfg.goal_reached_bonus
        self.heading_progress_weight = self.cfg.heading_progress_weight
        self.heading_coefficient = self.cfg.heading_coefficient
        self.laziness_penalty_weight = self.cfg.laziness_penalty_weight
        self.position_progress_weight = self.cfg.position_progress_weight
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
        self.flip_penalty_weight = self.cfg.flip_penalty_weight
        self.action_scale = self.cfg.action_scale
        self.action_max = self.cfg.action_max
        self._action_state = torch.zeros(
            (self.num_envs, self.cfg.action_space),
            device=self.device,
            dtype=torch.float32,
        )
        self._default_pos = self.robot.data.default_joint_pos.clone()
        self._smoothing_factor = 0.5

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
            update_period=0.0025,
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
        self._actions = self._actions * self.action_scale
        self._actions = torch.nan_to_num(self._actions, 0.0)
        self._actions = torch.clamp(
            self._actions, min=-self.action_max, max=self.action_max
        )
        self._action_state = self._actions.clone()

    def _apply_action(self) -> None:
        # --- Vectorized low-level Spot policy call for all environments ---
        # Gather all required robot state as torch tensors
        # TODO: Replace the following with actual command logic per environment
        previous_action = self._low_level_previous_action  # [num_envs, 12]
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
            previous_action,
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
                # self._position_error.unsqueeze(dim=1),
                # torch.cos(self.target_heading_error).unsqueeze(dim=1),
                # torch.sin(self.target_heading_error).unsqueeze(dim=1),
                self._action_state[:, 0].unsqueeze(dim=1),
                self._action_state[:, 1].unsqueeze(dim=1),
                self._action_state[:, 2].unsqueeze(dim=1),
                self._get_distance_to_walls().unsqueeze(dim=1),  # Add wall distance
            ),
            dim=-1,
        )

    def _get_rewards(self) -> torch.Tensor:
        # position_progress_reward = (
        #     torch.nan_to_num(
        #         self._previous_position_error - self._position_error,
        #         posinf=0.0,
        #         neginf=0.0,
        #     )
        #     * self.position_progress_weight
        # )
        # target_heading_reward = self.heading_progress_weight * torch.nan_to_num(
        #     torch.exp(-torch.abs(self.target_heading_error) / self.heading_coefficient),
        #     posinf=0.0,
        #     neginf=0.0,
        # )
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
        self.task_completed = self._target_index > (self._num_goals - 1)
        self._target_index = self._target_index % self._num_goals

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
            self._accumulated_laziness * self.laziness_decay + current_laziness
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

        # Debug print
        if not hasattr(self, "_debug_counter"):
            self._debug_counter = 0
        self._debug_counter += 1

        if self._debug and self._debug_counter % 100 == 0:
            with torch.no_grad():
                debug_size = 5
                print("\nLaziness Debug (Step {}):".format(self._debug_counter))
                for i in range(min(debug_size, self.num_envs)):
                    print(f"Env {i}:")
                    print(f"  Current speed: {linear_speed[i]:.3f}")
                    print(
                        f"  Accumulated laziness: {self._accumulated_laziness[i]:.3f}"
                    )
                    print(f"  Laziness penalty: {laziness_penalty[i]:.3f}")

        # Add wall distance penalty
        min_wall_dist = self._get_distance_to_walls()
        danger_distance = (
            self.wall_thickness / 2 + 2.0
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
            linear_speed * 0.9 + self.previous_linear_speed * 0.1,
            posinf=0.0,
            neginf=0.0,
        )
        self.previous_linear_speed = linear_speed.clone()
        time_penalty = -torch.ones_like(laziness_penalty)
        flip_penalty = -self.flip_penalty_weight * self._vehicle_flipped
        composite_reward = (
            goal_reached_reward
            + linear_speed_reward
            + laziness_penalty
            + wall_penalty
            + time_penalty
            + flip_penalty
            # + position_progress_reward
            # + target_heading_reward
        )

        if self._debug and self._debug_counter % 100 == 0:
            print("=" * 100)
            print(f"Goal reached: {goal_reached[0].item()}")
            print(f"Goal Reward: {goal_reached_reward[0].item()}")
            # print(f"Position progress reward: {position_progress_reward[0].item()}")
            # print(f"Target heading reward: {target_heading_reward[0].item()}")
            print(f"Linear speed reward: {linear_speed[0].item()}")
            print(f"Laziness penalty: {laziness_penalty[0].item()}")
            print(f"Wall penalty: {wall_penalty[0].item()}")
            print("=" * 100)

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

        if torch.any(composite_reward.isnan()):
            raise ValueError("Rewards cannot be NAN")

        return composite_reward
