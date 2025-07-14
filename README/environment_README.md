Environment Setup - IsaacSim 5.0.0

# Install and setup python packet manager (uv) using the following steps:

1. Install and setup the uv environment on the server machine.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Setup a virtual environment using the following command:
```bash
uv venv $HOME/isaaclab_env --python 3.11 # Can change the name of the virtual environment to anything you want.
```

3. Activate the virtual environment using the following command:
```bash
source $HOME/isaaclab_env/bin/activate # Can change the name of the virtual environment to anything you want.
```

4. Install the packages using the requirements.txt file.
```bash
uv pip install -r requirements_machine.txt
```

5. Install IsaacSim 5.0.0
```bash
git clone https://github.com/isaac-sim/IsaacSim.git isaacsim
cd isaacsim
git lfs install
git lfs pull
```

6. Check GCC and G++ version
```bash
gcc --version
g++ --version
```
Make sure the GCC and G++ version 11 is being used.

```bash
./build.sh
```

7. Install IsaacLab for IsaacSim 5.0.0
```bash
git clone -b feature/isaacsim_5_0 https://github.com/isaac-sim/IsaacLab.git
ln -s ../IsaacSim/_build/linux-x86_64/release _isaac_sim # This is a symbolic link to the IsaacSim build directory.
```

8. Now go to ISAAC_INSTALLATIONS and setup the IsaacSim 5.0.0 in your UV environment.
```bash
./isaacsim.sh $HOME/IsaacSim
```

9. Now go to ISAAC_INSTALLATIONS and setup the IsaacLab in your UV environment.
```bash
./isaaclab.sh $HOME/IsaacLab
```


Environment Setup - IsaacSim 4.5.0

# Install and setup python packet manager (uv) using the following steps:

1. Install and setup the uv environment on the server machine.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Setup a virtual environment using the following command:
```bash
uv venv $HOME/isaaclab_env --python 3.10
```

3. Activate the virtual environment using the following command:
```bash
source $HOME/isaaclab_env/bin/activate
```

4. Install the packages using the requirements.txt file.
```bash
uv pip install -r requirements.txt
```
or If you are running on the server machine, you can use the requirements_machine.txt file to install the packages.
```bash
uv pip install -r requirements_machine.txt
```

5. Test if both jax and pytorch are installed correctly using the following command:
Check for gpu availability in both jax and pytorch:

```bash
```python
python -c "import jax; import torch; print(f'JAX devices: {jax.devices()}'); print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'PyTorch device count: {torch.cuda.device_count()}'); print(f'Pytorch and JAX have been properly installed and GPU devices are properly configured')"
```
```
