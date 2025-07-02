# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from .nav_env_cfg import NavEnvCfg

# Get absolute path to workspace root
WORKSPACE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../../")
)

# USD path with proper resolution for cross-platform compatibility
USD_PATH = os.path.join(
    WORKSPACE_ROOT,
    "source",
    "cognitiverl",
    "cognitiverl",
    "tasks",
    "direct",
    "custom_assets",
    "cognitiverl_simple_better.usd",
)

navigation_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),
        joint_pos={
            "Wheel__Knuckle__Front_Left": 0.0,
            "Wheel__Knuckle__Front_Right": 0.0,
            "Wheel__Upright__Rear_Right": 0.0,
            "Wheel__Upright__Rear_Left": 0.0,
            "Knuckle__Upright__Front_Right": 0.0,
            "Knuckle__Upright__Front_Left": 0.0,
        },
    ),
    actuators={
        "throttle": ImplicitActuatorCfg(
            joint_names_expr=["Wheel.*"],
            effort_limit_sim=40000.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=100000.0,
        ),
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["Knuckle__Upright__Front.*"],
            effort_limit_sim=40000.0,
            velocity_limit_sim=100.0,
            stiffness=1000.0,
            damping=0.0,
        ),
    },
)


@configclass
class LeatherbackNavEnvCfg(NavEnvCfg):
    img_size = [3, 32, 32]
    observation_space = (
        img_size[0] * img_size[1] * img_size[2] + 4
    )  # Changed from 8 to 9 to include minimum wall distance
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
    wall_penalty_weight = -0.2
    linear_speed_weight = 0.05
    laziness_penalty_weight = -0.3
    # flip_penalty_weight = 100.0

    # Laziness
    laziness_decay = 0.99
    laziness_threshold = 8.0
    max_laziness = 10.0

    throttle_scale = 20.0
    throttle_max = 500.0
    steering_scale = 0.5  # Old value: 0.1
    steering_max = 3.0  # Old value: 0.75
