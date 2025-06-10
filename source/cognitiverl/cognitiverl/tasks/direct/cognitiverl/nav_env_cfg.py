# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from .nav import navigation_CFG
from .waypoint import WAYPOINT_CFG


@configclass
class BaseNavEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 1
    observation_space = 4
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    # custom parameters/scales
    # - controllable joint
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    # - action scale
    action_scale = 100.0  # [N]
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    # - reset states/conditions
    initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    max_cart_pos = 3.0  # reset if cart exceeds this position [m]


@configclass
class NavEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 30.0
    """
    observation_space = {
        "state": 6,
        "image": (32, 32, 3),
    }
    """
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)
    robot_cfg: ArticulationCfg = navigation_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=navigation_CFG.spawn.replace(
            scale=(0.03, 0.03, 0.03)
        ),  # 3D vector for scaling
    )
    waypoint_cfg = WAYPOINT_CFG

    throttle_dof_name = [
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right",
        "Wheel__Upright__Rear_Right",
        "Wheel__Upright__Rear_Left",
    ]
    steering_dof_name = [
        "Knuckle__Upright__Front_Right",
        "Knuckle__Upright__Front_Left",
    ]

    # Action and observation space
    action_space = 2
    observation_space = 3075  # Changed from 8 to 9 to include minimum wall distance

    env_spacing = 40.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=env_spacing, replicate_physics=True
    )

    # Wall parameters
    room_size = 40.0
    num_goals = 10
    wall_thickness = 5.0
    wall_height = 3.0
    position_tolerance = waypoint_cfg.markers["marker1"].radius

    # Reward Coefficients (updated to navigation robot)
    goal_reached_bonus = 125.0
    position_progress_weight = 3.0
    heading_progress_weight = 0.5
    wall_penalty_weight = 1.0
    linear_speed_weight = 0.5
    laziness_penalty_weight = 1.0
    heading_coefficient = 0.25
    flip_penalty_weight = 100.0
    # Laziness
    laziness_decay = 0.99
    laziness_threshold = 8.0
    max_laziness = 10.0

    throttle_scale = 10
    throttle_max = 50
    steering_scale = 2.0  # Old value: 0.1
    steering_max = 10.0  # Old value: 0.75
