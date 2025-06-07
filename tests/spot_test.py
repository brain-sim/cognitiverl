# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates different legged robots.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/quadrupeds.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse
import sys

import gymnasium as gym
import torch
from isaaclab.app import AppLauncher

"""Rest everything follows."""


def launch_app():
    # add argparse arguments
    parser = argparse.ArgumentParser(
        description="This script demonstrates different legged robots."
    )
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    # parse the arguments
    args_cli = parser.parse_args()

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    return simulation_app


try:
    from isaaclab.app import AppLauncher

    simulation_app = launch_app()
except ImportError:
    raise ImportError("Isaac Lab is not installed. Please install it first.")


def make_isaaclab_env(
    task,
    device,
    num_envs,
    capture_video,
    disable_fabric,
    log_dir=None,
    video_length=200,
    *args,
    **kwargs,
):
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    import cognitiverl.tasks  # noqa: F401

    def thunk():
        cfg = parse_env_cfg(
            task, device, num_envs=num_envs, use_fabric=not disable_fabric
        )
        env = gym.make(
            task,
            cfg=cfg,
            render_mode="rgb_array"
            if (capture_video and log_dir is not None)
            else None,
        )
        return env

    return thunk


try:
    import msvcrt  # For Windows

    def get_key():
        return msvcrt.getch().decode("utf-8").lower()
except ImportError:
    import termios
    import tty

    def get_key():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch.lower()


# Keyboard mapping: (forward, left, yaw)
KEYBOARD_MAPPING = {
    "w": (10.0, 0.0, 0.0),  # forward
    "s": (-5.0, 0.0, 0.0),  # backward
    "a": (0.0, 3.0, 0.0),  # left
    "d": (0.0, -3.0, 0.0),  # right
    "q": (0.0, 0.0, 3.0),  # yaw left
    "e": (0.0, 0.0, -3.0),  # yaw right
}


def main():
    """Main function."""
    print("[INFO]: Creating environment Spot Nav...")
    env = make_isaaclab_env(
        "Spot-Nav-v0",
        "cuda:0",
        1,
        False,
        False,
    )()
    print("[INFO]: Environment Spot Nav has been created.")
    obs, _ = env.reset()
    action = torch.zeros(env.action_space.shape, dtype=torch.float32)
    print(
        "[INFO]: Use WASD to move, Q/E to yaw. Press 'x' to exit, 'r' to reset, space to zero action."
    )

    i = 0
    while True:
        env.step(torch.zeros(env.action_space.shape, dtype=torch.float32))
        key = get_key()
        if key == "x":
            print("[INFO]: Exiting...")
            break
        if key in KEYBOARD_MAPPING:
            delta = torch.tensor(KEYBOARD_MAPPING[key], dtype=torch.float32)
            action[:3] = delta
            action[:3].clip_(min=-10.0, max=10.0)
            print(f"[DEBUG]: Action updated: {action.tolist()}")
            for _ in range(3):
                _, reward, _, _, _ = env.step(action.clone())
                i += 1
        elif key == " ":
            action.zero_()
            _, reward, _, _, _ = env.step(action.clone())
            print("[DEBUG]: Action reset to zero.")
        elif key == "r":
            obs, _ = env.reset()
            action.zero_()
            print("[DEBUG]: Environment reset.")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Exception:", e)
        raise e
    finally:
        simulation_app.close()
