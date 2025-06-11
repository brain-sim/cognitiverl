# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .waypoint import WAYPOINT_CFG


@configclass
class NavEnvCfg(DirectRLEnvCfg):
    """
    observation_space = {
        "state": 0,
        "image": (32, 32, 3),
    }
    """

    # env
    decimation = 4
    episode_length_s = 30.0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)

    # scene
    env_spacing = 40.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=env_spacing, replicate_physics=True
    )
    waypoint_cfg = WAYPOINT_CFG

    # Wall parameters
    room_size = 40.0
    num_goals = 10
    wall_thickness = 2.0
    wall_height = 3.0
    position_tolerance = waypoint_cfg.markers["marker1"].radius - 0.5
