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
    episode_length_s = 30.0
    action_space = 2
    observation_space = 3078  # Changed from 8 to 9 to include minimum wall distance
    """
    observation_space = {
        "state": 6,
        "image": (32, 32, 3),
    }
    """
    state_space = 0
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 400, render_interval=decimation
    )  # dt=1/250
    robot_cfg: ArticulationCfg = SPOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        # spawn=SPOT_CFG.spawn.replace(scale=(0.03, 0.03, 0.03)),  # 3D vector for scaling
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

    env_spacing = 40.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=env_spacing, replicate_physics=True
    )

    num_goals = 10

    course_length_coefficient = 2.5
    course_width_coefficient = 2.0
    position_tolerance = 0.15
    goal_reached_bonus = 10.0
    position_progress_weight = 1.0
    heading_coefficient = 0.25
    heading_progress_weight = 0.05

    action_scale = 3.0
    action_max = 3.0

    laziness_decay = 0.95
    laziness_threshold = 0.5
    max_laziness = 10.0

    wall_thickness = 2.0
    wall_height = 3.0
