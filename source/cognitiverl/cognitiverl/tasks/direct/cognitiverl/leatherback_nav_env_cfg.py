# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from .nav import navigation_CFG
from .nav_env_cfg import NavEnvCfg


@configclass
class LeatherbackNavEnvCfg(NavEnvCfg):
    robot_cfg: ArticulationCfg = navigation_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=navigation_CFG.spawn.replace(
            scale=(0.03, 0.03, 0.03)
        ),  # 3D vector for scaling
    )

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

    # Reward Coefficients (updated to navigation robot)
    goal_reached_bonus = 125.0
    position_progress_weight = 3.0
    heading_progress_weight = 0.5
    wall_penalty_weight = 0.2
    linear_speed_weight = 0.05
    laziness_penalty_weight = 0.3
    # flip_penalty_weight = 100.0

    # Laziness
    laziness_decay = 0.99
    laziness_threshold = 8.0
    max_laziness = 10.0

    throttle_scale = 20.0
    throttle_max = 500.0
    steering_scale = 0.5  # Old value: 0.1
    steering_max = 3.0  # Old value: 0.75
