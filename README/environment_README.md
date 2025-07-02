Environment Setup

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
uv install -r requirements.txt
```
or If you are running on the server machine, you can use the requirements_machine.txt file to install the packages.
```bash
uv install -r requirements_machine.txt
```

5. Test if both jax and pytorch are installed correctly using the following command:
Check for gpu availability in both jax and pytorch:

```bash
```python
python -c "import jax; import torch; print(f'JAX devices: {jax.devices()}'); print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'PyTorch device count: {torch.cuda.device_count()}'); print(f'Pytorch and JAX have been properly installed and GPU devices are properly configured')"
```
```
