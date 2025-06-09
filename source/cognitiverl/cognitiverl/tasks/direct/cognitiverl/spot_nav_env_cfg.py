# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.spot import SPOT_CFG

from .waypoint import WAYPOINT_CFG


@configclass
class SpotNavEnvCfg(DirectRLEnvCfg):
    decimation = 3  # 2
    episode_length_s = 20.0
    action_space = 3
    observation_space = 3076  # Changed from 8 to 9 to include minimum wall distance

    """
    observation_space = {
        "state": 6,
        "image": (32, 32, 3),
    }
    """
    state_space = 0
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 250, render_interval=decimation
    )  # dt=1/250
    robot_cfg: ArticulationCfg = SPOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    waypoint_cfg = WAYPOINT_CFG

    dof_name = [
        "fl_hx",
        "fr_hx",
        "hl_hx",
        "hr_hx",
        "fl_hy",
        "fr_hy",
        "hl_hy",
        "hr_hy",
        "fl_kn",
        "fr_kn",
        "hl_kn",
        "hr_kn",
    ]

    # Scene
    env_spacing = 40.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=env_spacing, replicate_physics=True
    )

    # Scene Properties
    room_size = 20.0
    num_goals = 10
    wall_thickness = 5.0
    wall_height = 3.0
    course_length_coefficient = 2.5
    course_width_coefficient = 2.0
    position_tolerance = waypoint_cfg.markers["marker1"].radius

    # Reward Coefficients
    goal_reached_bonus = 10.0
    position_progress_weight = 3.0
    heading_progress_weight = 0.5
    wall_penalty_weight = 1.0
    linear_speed_weight = 0.05
    laziness_penalty_weight = 1.0
    heading_coefficient = 0.25
    # Laziness
    laziness_decay = 0.95
    laziness_threshold = 1.0
    max_laziness = 10.0

    # Action Scaling
    action_scale = 3.0
    action_max = 3.0
