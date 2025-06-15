from __future__ import annotations

from collections.abc import Sequence

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import math
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import FixedCuboid

from .nav_env_cfg import NavEnvCfg


class NavEnv(DirectRLEnv):
    cfg: NavEnvCfg

    def __init__(
        self,
        cfg: NavEnvCfg,
        render_mode: str | None = None,
        debug: bool = False,
        max_total_steps: int | None = None,
        **kwargs,
    ):
        # Add room size as a class attribut
        self.room_size = getattr(cfg, "room_size", 40.0)
        self._num_goals = getattr(cfg, "num_goals", 1)
        self.env_spacing = getattr(cfg, "env_spacing", 40.0)

        super().__init__(cfg, render_mode, **kwargs)
        self._setup_robot_dof_idx()
        self._goal_reached = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.int32
        )
        self.task_completed = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.bool
        )
        self._target_positions = torch.zeros(
            (self.num_envs, self._num_goals, 2), device=self.device, dtype=torch.float32
        )
        self._markers_pos = torch.zeros(
            (self.num_envs, self._num_goals, 3), device=self.device, dtype=torch.float32
        )
        self._target_index = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.int32
        )

        # Add accumulated laziness tracker
        self._accumulated_laziness = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float32
        )
        self._episode_waypoints_passed = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.int32
        )
        self._episode_reward_buf = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float32
        )

        self._debug = debug
        self._setup_config()
        self.max_total_steps = max_total_steps

    def _setup_config(self):
        raise NotImplementedError("Subclass must implement this method")

    def _setup_robot_dof_idx(self):
        raise NotImplementedError("Subclass must implement this method")

    def _setup_camera(self):
        raise NotImplementedError("Subclass must implement this method")

    def _setup_scene(self):
        # Create a large ground plane without grid
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                size=(
                    4096 * 40.0,
                    4096 * 40.0,
                ),  # Much larger ground plane (500m x 500m)
                color=(0.2, 0.2, 0.2),  # Dark gray color
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=self.cfg.static_friction,
                    dynamic_friction=self.cfg.dynamic_friction,
                    restitution=0.0,
                ),
            ),
        )

        # Setup rest of the scene
        self.robot = Articulation(self.cfg.robot_cfg)
        self._setup_camera()
        self.waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)
        self.object_state = []

        # FIRST: Clone environments to initialize env_origins
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot

        # Import the necessary classes and NumPy
        import numpy as np

        # Define wall properties
        self.wall_thickness = self.cfg.wall_thickness
        self.wall_height = self.cfg.wall_height
        self.wall_position = (self.room_size - self.wall_thickness) / 2

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
                    [
                        origin_cpu[0],
                        origin_cpu[1] + self.wall_position,
                        self.wall_height / 2,
                    ]
                ),
                scale=np.array(
                    [
                        self.room_size,
                        self.wall_thickness,
                        self.wall_height,
                    ]
                ),
                color=np.array([0.2, 0.3, 0.8]),
            )

            # South wall (bottom)
            FixedCuboid(
                prim_path=f"/World/envs/{env_name}/walls/south_wall",
                position=np.array(
                    [
                        origin_cpu[0],
                        origin_cpu[1] - self.wall_position,
                        self.wall_height / 2,
                    ]
                ),
                scale=np.array(
                    [
                        self.room_size,
                        self.wall_thickness,
                        self.wall_height,
                    ]
                ),
                color=np.array([0.2, 0.3, 0.8]),
            )

            # East wall (right)
            FixedCuboid(
                prim_path=f"/World/envs/{env_name}/walls/east_wall",
                position=np.array(
                    [
                        origin_cpu[0] + self.wall_position,
                        origin_cpu[1],
                        self.wall_height / 2,
                    ]
                ),
                scale=np.array(
                    [
                        self.wall_thickness,
                        self.room_size,
                        self.wall_height,
                    ]
                ),
                color=np.array([0.2, 0.3, 0.8]),
            )

            # West wall (left)
            FixedCuboid(
                prim_path=f"/World/envs/{env_name}/walls/west_wall",
                position=np.array(
                    [
                        origin_cpu[0] - self.wall_position,
                        origin_cpu[1],
                        self.wall_height / 2,
                    ]
                ),
                scale=np.array(
                    [
                        self.wall_thickness,
                        self.room_size,
                        self.wall_height,
                    ]
                ),
                color=np.array([0.2, 0.3, 0.8]),
            )
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        raise NotImplementedError("Subclass must implement this method")

    def _apply_action(self) -> None:
        raise NotImplementedError("Subclass must implement this method")

    def _get_image_obs(self) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement this method")

    def _get_state_obs(self) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement this method")

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
        image_obs = self._get_image_obs()
        state_obs = self._get_state_obs(image_obs)
        state_obs = torch.nan_to_num(state_obs, posinf=0.0, neginf=0.0)
        if torch.any(state_obs.isnan()):
            raise ValueError("Observations cannot be NAN")
        return {"policy": state_obs}

    def _log_episode_info(self, env_ids: torch.Tensor):
        """Logs episode information for the given environment IDs.
        Args:
            env_ids: A tensor of environment IDs that have been reset.
        """
        if len(env_ids) > 0:
            # log episode length
            self.extras["episode_length"] = torch.mean(
                self.episode_length_buf[env_ids].float()
            ).item()
            # calculate and log completion percentage
            completion_frac = (
                self._episode_waypoints_passed[env_ids].float() / self._num_goals
            )
            self.extras["success_rate"] = torch.mean(completion_frac).item()
            # log episode reward
            self.extras["episode_reward"] = torch.mean(
                self._episode_reward_buf[env_ids].float()
            ).item()
            self.extras["goals_reached"] = torch.mean(
                self._goal_reached[env_ids].float()
            ).item()
            self.extras["waypoints_passed"] = torch.mean(
                self._episode_waypoints_passed[env_ids].float()
            ).item()
            self.extras["max_episode_length"] = torch.mean(
                self.max_episode_length_buf[env_ids].float()
            )

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model.apply(action)

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if (
                self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
            ):
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
        self.camera.update(dt=self.step_dt)
        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._log_episode_info(reset_env_ids)
            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(
                self.obs_buf["policy"]
            )
        for k, v in self.obs_buf.items():
            if k != "policy":
                del self.obs_buf[k]
        # return observations, rewards, resets and extras
        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """Run rendering without stepping through the physics.

        By convention, if mode is:

        - **human**: Render to the current display and return nothing. Usually for human consumption.
        - **rgb_array**: Return an numpy.ndarray with shape (x, y, 3), representing RGB values for an
          x-by-y pixel image, suitable for turning into a video.

        Args:
            recompute: Whether to force a render even if the simulator has already rendered the scene.
                Defaults to False.

        Returns:
            The rendered image as a numpy array if mode is "rgb_array". Otherwise, returns None.

        Raises:
            RuntimeError: If mode is set to "rgb_data" and simulation render mode does not support it.
                In this case, the simulation render mode must be set to ``RenderMode.PARTIAL_RENDERING``
                or ``RenderMode.FULL_RENDERING``.
            NotImplementedError: If an unsupported rendering mode is specified.
        """
        # run a rendering step of the simulator
        # if we have rtx sensors, we do not need to render again sin
        if not self.sim.has_rtx_sensors() and not recompute:
            self.sim.render()
        # decide the rendering mode
        if self.render_mode == "human" or self.render_mode is None:
            return None
        elif self.render_mode == "rgb_array":
            # check that if any render could have happened
            if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
                raise RuntimeError(
                    f"Cannot render '{self.render_mode}' when the simulation render mode is"
                    f" '{self.sim.render_mode.name}'. Please set the simulation render mode to:"
                    f"'{self.sim.RenderMode.PARTIAL_RENDERING.name}' or '{self.sim.RenderMode.FULL_RENDERING.name}'."
                    " If running headless, make sure --enable_cameras is set."
                )
            # create the annotator if it does not exist
            if not hasattr(self, "_rgb_annotator"):
                import omni.replicator.core as rep

                # create render product
                self._render_product = rep.create.render_product(
                    self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
                )
                # create rgb annotator -- used to read data from the render product
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator(
                    "rgb", device="cpu"
                )
                self._rgb_annotator.attach([self._render_product])
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            # note: initially the renerer is warming up and returns empty data
            if rgb_data.size == 0:
                return np.zeros(
                    (self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3),
                    dtype=np.uint8,
                )
            else:
                # image_obs = self.camera.data.output["rgb"]
                # rgb_data = image_obs[0, :, :, :3].detach().cpu().numpy()
                # return cv2.resize(rgb_data, (self.cfg.viewer.resolution[0], self.cfg.viewer.resolution[1]))
                return rgb_data[:, :, :3]

        else:
            raise NotImplementedError(
                f"Render mode '{self.render_mode}' is not supported. Please use: {self.metadata['render_modes']}."
            )

    def _get_rewards(self) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement this method")

    def _check_vehicle_flipped(self) -> torch.Tensor:
        # Get robot's orientation
        robot_quat = self.robot.data.root_quat_w  # (num_envs, 4) in (w,x,y,z)
        local_up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        world_up_vector = math.quat_rotate(
            robot_quat, local_up.repeat(self.num_envs, 1)
        )
        world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        up_dot_product = torch.sum(world_up_vector * world_up, dim=-1)  # (num_envs,)
        up_angle = torch.abs(
            torch.acos(torch.clamp(up_dot_product, -1.0, 1.0))
        )  # radians

        # Consider flipped if angle > 60 degrees (pi/3 radians)
        flipped = up_angle > (torch.pi / 3)
        return flipped

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        task_failed = self.episode_length_buf > self.max_episode_length_buf
        self._vehicle_flipped = self._check_vehicle_flipped()
        task_failed |= self._vehicle_flipped
        debug_size = 5
        if self._debug:
            if torch.any(self._vehicle_flipped[:debug_size]):
                print(f"Vehicle flipped : {self._vehicle_flipped[:debug_size]}")
            if torch.any(task_failed[:debug_size]):
                print(f"Task failed : {task_failed[:debug_size]}")
            if torch.any(self.task_completed[:debug_size]):
                print(f"Task completed : {self.task_completed[:debug_size]}")
        return task_failed, self.task_completed

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self.camera.reset(env_ids)
        if self.max_total_steps is None:
            self.max_episode_length_buf[env_ids] = self.max_episode_length
        else:
            min_episode_length = max(
                min(
                    100
                    + int(
                        0.8
                        * self.max_episode_length
                        * self.common_step_counter
                        * self.num_envs
                        / self.max_total_steps
                    ),
                    int(0.8 * self.max_episode_length),
                ),
                100,
            )
            self.max_episode_length_buf[env_ids] = torch.randint(
                min_episode_length,
                self.max_episode_length + 1,
                (len(env_ids),),
                device=self.device,
            )

        self._episode_waypoints_passed[env_ids] = 0
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

        # Generate random positions within the square room (vectorized)
        margin = self.cfg.position_tolerance + self.cfg.position_margin_epsilon
        for i in range(self._num_goals):
            robot_xy = robot_pose[:, :2]  # shape: (num_reset, 2)
            env_origins = self.scene.env_origins[env_ids, :2]  # shape: (num_reset, 2)
            num_reset = len(env_ids)
            if i == 0:
                # First goal: must satisfy margin from robot
                N = 20  # number of candidates per env
                N1 = N // 2
                N2 = N - N1
                # First half: smaller range
                tx_cand1 = (
                    torch.rand(num_reset, N1, device=self.device) * self.room_size / 2
                    - self.room_size / 4
                )
                ty_cand1 = (
                    torch.rand(num_reset, N1, device=self.device) * self.room_size / 2
                    - self.room_size / 4
                )
                # Second half: larger range
                tx_cand2 = (
                    torch.rand(num_reset, N2, device=self.device) * self.room_size
                    - self.room_size / 2
                )
                ty_cand2 = (
                    torch.rand(num_reset, N2, device=self.device) * self.room_size
                    - self.room_size / 2
                )
                # Concatenate
                tx_cand = torch.cat([tx_cand1, tx_cand2], dim=1)
                ty_cand = torch.cat([ty_cand1, ty_cand2], dim=1)
                # Clip to wall margin
                tx_cand = tx_cand.clip(
                    min=-self.wall_position
                    + self.wall_thickness / 2
                    + self.cfg.position_tolerance,
                    max=self.wall_position
                    - self.wall_thickness / 2
                    - self.cfg.position_tolerance,
                )
                ty_cand = ty_cand.clip(
                    min=-self.wall_position
                    + self.wall_thickness / 2
                    + self.cfg.position_tolerance,
                    max=self.wall_position
                    - self.wall_thickness / 2
                    - self.cfg.position_tolerance,
                )
                # Offset by environment origin
                tx_cand = tx_cand + env_origins[:, 0:1]
                ty_cand = ty_cand + env_origins[:, 1:2]
                # Compute distances to robot
                dx = tx_cand - robot_xy[:, 0:1]
                dy = ty_cand - robot_xy[:, 1:2]
                dist = (dx**2 + dy**2).sqrt()  # shape: (num_reset, N)
                valid = dist >= margin
                # For each env, pick the first valid candidate, else last
                first_valid_idx = valid.float().argmax(dim=1)
                has_valid = valid.any(dim=1)
                # If no valid, use last candidate
                first_valid_idx = torch.where(
                    has_valid, first_valid_idx, torch.full_like(first_valid_idx, N - 1)
                )
                tx = tx_cand[torch.arange(num_reset), first_valid_idx]
                ty = ty_cand[torch.arange(num_reset), first_valid_idx]
                self._target_positions[env_ids, i, 0] = tx
                self._target_positions[env_ids, i, 1] = ty
            else:
                # Other goals: no margin validation, just sample and assign
                tx = (
                    torch.rand(num_reset, device=self.device) * self.room_size / 2
                    - self.room_size / 4
                )
                ty = (
                    torch.rand(num_reset, device=self.device) * self.room_size / 2
                    - self.room_size / 4
                )
                tx = tx.clip(
                    min=-self.wall_position
                    + self.wall_thickness / 2
                    + self.cfg.position_tolerance,
                    max=self.wall_position
                    - self.wall_thickness / 2
                    - self.cfg.position_tolerance,
                )
                ty = ty.clip(
                    min=-self.wall_position
                    + self.wall_thickness / 2
                    + self.cfg.position_tolerance,
                    max=self.wall_position
                    - self.wall_thickness / 2
                    - self.cfg.position_tolerance,
                )
                self._target_positions[env_ids, i, 0] = tx + env_origins[:, 0]
                self._target_positions[env_ids, i, 1] = ty + env_origins[:, 1]

        # Verify only the first goal maintains minimum distance from robot
        robot_xy = robot_pose[:, :2]  # Get robot XY positions
        tx = self._target_positions[env_ids, 0, 0]  # First goal X positions
        ty = self._target_positions[env_ids, 0, 1]  # First goal Y positions
        dist = ((tx - robot_xy[:, 0]) ** 2 + (ty - robot_xy[:, 1]) ** 2).sqrt()
        min_dist = dist.min().item()
        if not torch.all(dist >= margin):
            raise ValueError(
                f"Invalid target placement: Some first-goal targets are too close to robot. "
                f"Minimum distance {min_dist:.2f} is less than required margin {margin:.2f}"
            )

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
        # Distance to walls (positive means inside the room)
        north_dist = (
            self.wall_position - relative_positions[:, 1]
        )  # Distance to north wall (y+)
        south_dist = (
            self.wall_position + relative_positions[:, 1]
        )  # Distance to south wall (y-)
        east_dist = (
            self.wall_position - relative_positions[:, 0]
        )  # Distance to east wall (x+)
        west_dist = (
            self.wall_position + relative_positions[:, 0]
        )  # Distance to west wall (x-)

        # Stack all distances and get the minimum
        wall_distances = torch.stack(
            [north_dist, south_dist, east_dist, west_dist], dim=1
        )
        min_wall_distance = torch.min(wall_distances, dim=1)[
            0
        ]  # Get minimum distance for each environment

        # Add debug printing for the first few environments (every 100 steps)
        if (
            self._debug
            and hasattr(self, "_debug_counter")
            and self._debug_counter % 100 == 0
        ):
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
