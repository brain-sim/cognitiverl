from __future__ import annotations

from collections.abc import Sequence

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import FixedCuboid

from .nav_env_cfg import NavEnvCfg


class NavEnv(DirectRLEnv):
    cfg: NavEnvCfg

    def __init__(self, cfg: NavEnvCfg, render_mode: str | None = None, **kwargs):
        # Add room size as a class attribute
        self.room_size = 40.0  # Adjust as needed

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
        self._num_goals = 10
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
        self._target_index = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.int32
        )

        # Add accumulated laziness tracker
        self._accumulated_laziness = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float32
        )
        self.laziness_decay = (
            self.cfg.laziness_decay
        )  # How much previous laziness carries over
        self.laziness_threshold = (
            self.cfg.laziness_threshold
        )  # Speed threshold for considering "lazy"
        self.max_laziness = (
            self.cfg.max_laziness
        )  # Cap on accumulated laziness to prevent extreme penalties

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

        # FIRST: Clone environments to initialize env_origins
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot

        # Import the necessary classes and NumPy

        # Define wall properties
        wall_thickness = self.cfg.wall_thickness
        wall_height = self.cfg.wall_height
        wall_position = self.room_size / 2
        self.wall_thickness = wall_thickness

        # Create physics material for walls
        PhysicsMaterial(
            prim_path="/World/physics_material/wall_material",
            dynamic_friction=1.0,
            static_friction=1.5,
            restitution=0.1,
        )

        # Print the actual environment names to debug
        for env_idx, env_origin in enumerate(self.scene.env_origins):
            # This might need to be adjusted based on your environment naming scheme
            env_name = f"env_{env_idx}"

            # print(f"Setting up walls for environment: {env_name}")

            # Convert CUDA tensor to CPU before using in NumPy
            origin_cpu = env_origin.cpu().numpy()

            # North wall (top)
            FixedCuboid(
                prim_path=f"/World/envs/{env_name}/walls/north_wall",
                position=np.array(
                    [origin_cpu[0], origin_cpu[1] + wall_position, wall_height / 2]
                ),
                scale=np.array(
                    [self.room_size + wall_thickness, wall_thickness, wall_height]
                ),
                color=np.array([0.2, 0.3, 0.8]),
            )
            # north_wall.set_collision_enabled(True)
            # north_wall.apply_physics_material(wall_material)

            # South wall (bottom)
            FixedCuboid(
                prim_path=f"/World/envs/{env_name}/walls/south_wall",
                position=np.array(
                    [origin_cpu[0], origin_cpu[1] - wall_position, wall_height / 2]
                ),
                scale=np.array(
                    [self.room_size + wall_thickness, wall_thickness, wall_height]
                ),
                color=np.array([0.2, 0.3, 0.8]),
            )
            # south_wall.set_collision_enabled(True)
            # south_wall.apply_physics_material(wall_material)

            # East wall (right)
            FixedCuboid(
                prim_path=f"/World/envs/{env_name}/walls/east_wall",
                position=np.array(
                    [origin_cpu[0] + wall_position, origin_cpu[1], wall_height / 2]
                ),
                scale=np.array(
                    [wall_thickness, self.room_size + wall_thickness, wall_height]
                ),
                color=np.array([0.2, 0.3, 0.8]),
            )
            # east_wall.set_collision_enabled(True)
            # east_wall.apply_physics_material(wall_material)

            # West wall (left)
            FixedCuboid(
                prim_path=f"/World/envs/{env_name}/walls/west_wall",
                position=np.array(
                    [origin_cpu[0] - wall_position, origin_cpu[1], wall_height / 2]
                ),
                scale=np.array(
                    [wall_thickness, self.room_size + wall_thickness, wall_height]
                ),
                color=np.array([0.2, 0.3, 0.8]),
            )
            # west_wall.set_collision_enabled(True)
            # west_wall.apply_physics_material(wall_material)

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
        self._throttle_action = torch.clamp(self._throttle_action, -1, throttle_max)
        self.throttle_action = self._throttle_action.clone()
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
                self._get_distance_to_walls().unsqueeze(dim=1),  # Add wall distance
            ),
            dim=-1,
        )

        if torch.any(obs.isnan()):
            raise ValueError("Observations cannot be NAN")

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        position_progress_rew = self._previous_position_error - self._position_error
        target_heading_rew = torch.exp(
            -torch.abs(self.target_heading_error) / self.heading_coefficient
        )
        goal_reached = self._position_error < self.position_tolerance
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

        # Calculate laziness penalty using log
        laziness_penalty = -0.3 * torch.log1p(
            self._accumulated_laziness
        )  # log1p(x) = log(1 + x)

        # Reset accumulated laziness when reaching waypoint
        self._accumulated_laziness = torch.where(
            goal_reached,
            torch.zeros_like(self._accumulated_laziness),
            self._accumulated_laziness,
        )

        # Debug print
        if not hasattr(self, "_debug_counter"):
            self._debug_counter = 0
        self._debug_counter += 1

        if self._debug_counter % 100 == 0:
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
            self.wall_thickness / 2 + 5.0
        )  # Distance at which to start penalizing
        wall_penalty = torch.where(
            min_wall_dist > danger_distance,
            torch.zeros_like(min_wall_dist),
            -0.2
            * torch.exp(1.0 - min_wall_dist / danger_distance),  # Exponential penalty
        )

        composite_reward = (
            # position_progress_rew * self.position_progress_weight +
            position_progress_rew * 3
            + target_heading_rew * 0.5
            + goal_reached * self.goal_reached_bonus
            + linear_speed / (self.target_heading_error + 1e-8) * 0.05
            + laziness_penalty  # Updated laziness penalty
            + wall_penalty
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

        # CHANGE: Set car position to be randomly inside the room rather than outside of it
        # Use smaller margins to keep car away from walls
        room_margin = 10.0  # keep the larger car away from walls
        safe_room_size = self.room_size - room_margin * 2

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

        # Define square room size
        room_size = 20.0  # Size of the square room (20x20 units)

        # Generate random positions within the square room
        for i in range(self._num_goals):
            # Random positions within the square
            self._target_positions[env_ids, i, 0] = (
                torch.rand(num_reset, device=self.device) * room_size - room_size / 2
            )
            self._target_positions[env_ids, i, 1] = (
                torch.rand(num_reset, device=self.device) * room_size - room_size / 2
            )

        # Offset by environment origins
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
        """Calculate the minimum distance from the robot to the nearest wall.
        Returns:
            torch.Tensor: Minimum distance to nearest wall for each environment (num_envs,)
        """
        # Get robot positions and environment origins
        robot_positions = self.robot.data.root_pos_w[
            :, :2
        ]  # Shape: (num_envs, 2) - XY positions
        env_origins = self.scene.env_origins[
            :, :2
        ]  # Get XY origins for each environment

        # Calculate relative positions within each environment
        relative_positions = (
            robot_positions - env_origins
        )  # Subtract environment origin

        # Calculate distances to each wall within local environment coordinates
        wall_position = self.room_size / 2

        # Distance to walls (positive means inside the room)
        north_dist = (
            wall_position - relative_positions[:, 1]
        )  # Distance to north wall (y+)
        south_dist = (
            wall_position + relative_positions[:, 1]
        )  # Distance to south wall (y-)
        east_dist = (
            wall_position - relative_positions[:, 0]
        )  # Distance to east wall (x+)
        west_dist = (
            wall_position + relative_positions[:, 0]
        )  # Distance to west wall (x-)

        # Stack all distances and get the minimum
        wall_distances = torch.stack(
            [north_dist, south_dist, east_dist, west_dist], dim=1
        )
        min_wall_distance = torch.min(wall_distances, dim=1)[
            0
        ]  # Get minimum distance for each environment

        # Add debug printing for the first few environments (every 100 steps)
        if hasattr(self, "_debug_counter") and self._debug_counter % 100 == 0:
            with torch.no_grad():
                debug_size = 5
                print("\nDistance Calculation Debug:")
                for i in range(min(debug_size, self.num_envs)):
                    print(f"Env {i}:")
                    print(f"  Robot position: {robot_positions[i]}")
                    print(f"  Env origin: {env_origins[i]}")
                    print(f"  Relative position: {relative_positions[i]}")
                    print(f"  Wall distances [N,S,E,W]: {wall_distances[i]}")

        return min_wall_distance
