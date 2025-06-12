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
from pynput import keyboard

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
    "w": (1.0, 0.0),
    "s": (-1.0, 0.0),
    "a": (1.0, 1.0),
    "d": (1.0, -1.0),
    "z": (-1.0, 2.0),  # backward left
    "c": (-1.0, -2.0),  # backward right
}

# Global action variable
action = None
exit_flag = False

# Add this global variable
key_pressed = False


def on_press(key):
    global action, exit_flag, key_pressed
    try:
        k = key.char.lower()
        if k == "x":
            print("[INFO]: Exiting...")
            exit_flag = True
            return False  # Stop listener
        elif k in KEYBOARD_MAPPING:
            delta = torch.tensor(KEYBOARD_MAPPING[k], dtype=torch.float32)
            action[:2] = delta
            key_pressed = True
            print(f"[DEBUG]: Action updated: {action.tolist()}")
        elif k == " ":
            action.zero_()
            key_pressed = True
            print("[DEBUG]: Action reset to zero.")
        elif k == "r":
            action.zero_()
            key_pressed = True
            print("[DEBUG]: Environment reset.")
    except AttributeError:
        pass


def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press)
    listener.start()


def main():
    """Main function."""
    print("[INFO]: Creating environment Spot Nav...")
    env = make_isaaclab_env(
        "Leatherback-Nav-v0",
        "cuda:0",
        1,
        False,
        False,
    )()
    print("[INFO]: Environment Spot Nav has been created.")
    global action, key_pressed
    action = torch.zeros(env.action_space.shape, dtype=torch.float32)
    print(
        "[INFO]: Use WASD to move, Q/E to yaw. Press 'x' to exit, 'r' to reset, space to zero action."
    )

    start_keyboard_listener()
    obs, _ = env.reset()
    while not exit_flag:
        if key_pressed:
            step_action = action.clone()
            key_pressed = False
            action.zero_()
        else:
            step_action = torch.zeros_like(action)
        _, reward, _, _, _ = env.step(step_action)
        # Optionally, add a small sleep to avoid maxing out CPU
        # import time; time.sleep(0.02)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Exception:", e)
        raise e
    finally:
        simulation_app.close()
