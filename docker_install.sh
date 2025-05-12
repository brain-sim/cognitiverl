#!/bin/bash

set -e

echo "🔧 Updating package lists..."
sudo apt-get update

echo "🐳 Installing Docker..."
curl -fsSL https://get.docker.com | sh

echo "🔧 Adding user '$USER' to the 'docker' group..."
sudo usermod -aG docker $USER

echo "🔄 Applying new group membership..."
newgrp docker <<EONG
echo "✅ User '$USER' is now in the 'docker' group."
EONG

echo "🔍 Verifying Docker installation..."
docker run --rm hello-world

echo "🎮 Installing NVIDIA Container Toolkit..."

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

echo "⚙️ Configuring Docker to use the NVIDIA runtime..."
sudo nvidia-ctk runtime configure --runtime=docker

echo "🔄 Restarting Docker daemon..."
sudo systemctl restart docker

echo "🧪 Testing GPU access within Docker..."
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

echo "✅ Installation and configuration complete!"


# docker system prune -a --volumes