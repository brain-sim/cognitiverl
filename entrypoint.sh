#!/bin/bash
set -e

# Start VNC server
vncserver :1 -geometry 1920x1080 -depth 24

# Run the training script with default arguments
python scripts/skrl/train.py isaaclab "$@"
