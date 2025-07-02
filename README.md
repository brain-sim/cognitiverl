# CognitiveRL - Robotic Navigation with Isaac Lab

## Overview

CognitiveRL is a specialized robotic navigation framework built on Isaac Lab, focusing on autonomous navigation tasks for legged robots in complex environments. The project implements multiple challenging navigation scenarios with various terrain types, obstacles, and environmental conditions for training robust navigation policies.

**Key Features:**

- `Multi-Robot Support` - Specialized environments for Spot and Leatherback robots
- `Diverse Navigation Tasks` - Multiple navigation scenarios including obstacle avoidance, rough terrain, and grid-based navigation
- `Multi-Framework Support` - Compatible with RSL-RL, SKRL, Stable Baselines3, TorchRL, JAX-RL, and RL-Games
- `Comprehensive Training Recipes` - Detailed PPO training configurations and best practices
- `Advanced Terrain Generation` - Procedural terrain generation with height maps and obstacles
- `Isolation & Flexibility` - Developed as an extension outside the core Isaac Lab repository

**Keywords:** robotics, navigation, reinforcement learning, isaaclab, legged robots, terrain traversal

## Installation

- Before installing Isaac Lab, make sure to install the following dependencies:
  - [NVIDIA Driver](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
  - [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
  - [UV](README/environment_README.md) Install packages after cloning the repository as mentioned in Step 3 below.
  - [TurboVNC](README/server_DISPLAY_README.md)
  - [TurboVNC Viewer](README/client_DISPLAY_README.md)
    Note: Don't install VirtualGL for now (haven't tested it yet).
  - Create a Weights & Biases account at https://wandb.ai/login and obtain your API key. Set the `WANDB_API_KEY` environment variable with your key. (Weights & Biases is required for training process logging and visualization.)


- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) using the repo [IsaacLab](https://github.com/chamorajg/IsaacLab).
  We recommend using the uv installation as it simplifies calling `python` scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory) and install the dependencies using the `requirements.txt` file or `requirements_machine.txt` file if you are running on the server machine (RTX 6000 PRO Blackwell).

- (Optional) Try installing [quadruped](https://github.com/chamorajg/quadruped) to train low level policies for quadruped bots.

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```
        Preferred `<RL_LIBRARY>` is `torchrl` as it is a single file and easy to run.

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

## Quick Start

### List Available Environments
```bash
python scripts/list_envs.py
```

### Train with TorchRL (Recommended)
```bash
python scripts/torchrl/train.py --task=Spot-Nav-v0
```

### Train with RSL-RL (Optional)
```bash
python scripts/rsl_rl/train.py --task=Spot-Nav-v0
```

### Play with the trained policy
```bash
python scripts/torchrl/play.py --task=Spot-Nav-v0 --checkpoint_path=<PATH_TO_CHECKPOINT>
```

### Cleaning up logs and initial checkpoints
```bash
chmod +x cleanup_wandb.sh
./cleanup_wandb.sh
```
Use `DRYRUN` in the script to check if the cleanup script is working correctly.


### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

- For cursor users, you can use the `pyrightconfig.json` file to setup the cursor IDE and link the python interpreter to the vscode.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/cognitiverl/cognitiverl/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/cognitiverl"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```