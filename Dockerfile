# =============================
# Build the Docker image:
#  docker build --build-arg REPO_URL=https://github.com/youruser/yourrepo.git --build-arg REPO_BRANCH=main --build-arg GITHUB_UNAME=your_username --build-arg GITHUB_PAT=your_token_here -t my-vnc-image .
#
# Run the Docker container (with NVIDIA GPU and port forwarding):
#  docker run --gpus all -p 5901:5901 -e USER=vncuser -e PASSWORD=vncpassword my-vnc-image
#
# Add the necessary *.usd* files into the docker image before running.
# COPY source/cognitiverl/cognitiverl/tasks/direct/custom_assets/ /workspace/repo/source/cognitiverl/cognitiverl/tasks/direct/custom_assets/
# =============================


FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Build arguments for custom repo
ARG REPO_URL
ARG REPO_BRANCH=main

# System dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    python3.10 \
    python3-pip \
    curl \
    wget \
    ca-certificates \
    libglu1-mesa \
    libxi6 \
    libxrender1 \
    libxtst6 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Support cloning private repos with GITHUB_PAT
ARG GITHUB_UNAME
ARG GITHUB_PAT
RUN if [ -n "$GITHUB_PAT" ] && [ -n "$GITHUB_UNAME" ]; then \
    git clone --branch $REPO_BRANCH https://$GITHUB_UNAME:$GITHUB_PAT@${REPO_URL#https://} /workspace/repo; \
    else \
    git clone --branch $REPO_BRANCH $REPO_URL /workspace/repo; \
    fi

WORKDIR /workspace/repo

# Install Python dependencies with uv
RUN mkdir -p /root/.venv
RUN uv venv /root/.venv/isaaclab_env --python 3.10
# Use the virtual environment automatically
ENV VIRTUAL_ENV=/root/.venv/isaaclab_env
# Place entry points in the environment at the front of the path
ENV PATH="/root/.venv/isaaclab_env/bin:$PATH"

COPY requirements_docker.txt .
RUN uv pip install -r requirements_docker.txt

ARG ISAACLAB_BRANCH=main
RUN if [ -n "$GITHUB_PAT" ] && [ -n "$GITHUB_UNAME" ]; then \
    git clone --branch $ISAACLAB_BRANCH https://$GITHUB_UNAME:$GITHUB_PAT@github.com/$GITHUB_UNAME/IsaacLab.git /workspace/IsaacLab; \
    else \
    git clone --branch $ISAACLAB_BRANCH https://github.com/$GITHUB_UNAME/IsaacLab.git /workspace/IsaacLab; \
    fi
# Install IsaacLab (adjust as needed: pip install -e . or pip install .)
WORKDIR /workspace/IsaacLab
RUN ./isaaclab.sh --install
WORKDIR /workspace/repo

RUN uv pip install -U "jax[cuda12]"

# --- TurboVNC setup (official repo) ---
RUN wget -q -O- https://packagecloud.io/dcommander/turbovnc/gpgkey | gpg --dearmor > /etc/apt/trusted.gpg.d/TurboVNC.gpg && \
    wget https://raw.githubusercontent.com/TurboVNC/repo/main/TurboVNC.list -O /etc/apt/sources.list.d/TurboVNC.list

# --- VirtualGL setup (official repo) ---
RUN wget -q -O- https://packagecloud.io/dcommander/virtualgl/gpgkey | gpg --dearmor > /etc/apt/trusted.gpg.d/VirtualGL.gpg && \
    wget https://raw.githubusercontent.com/VirtualGL/repo/main/VirtualGL.list -O /etc/apt/sources.list.d/VirtualGL.list

# --- Install TurboVNC and VirtualGL ---
RUN apt-get update && \
    apt-get install -y turbovnc virtualgl

ENV PATH="/opt/TurboVNC/bin:/opt/VirtualGL/bin:${PATH}"

# Now vncpasswd is available!
# --- Set up TurboVNC password (default: vncpassword) ---
RUN mkdir -p /root/.vnc && \
    echo "vncpassword" | vncpasswd -f > /root/.vnc/passwd && \
    chmod 600 /root/.vnc/passwd

# --- Configure VirtualGL (non-interactive defaults) ---
RUN /opt/VirtualGL/bin/vglserver_config -silent

# --- Expose default VNC port ---
EXPOSE 5901

# --- Copy entrypoint script and set as default CMD ---
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENV OMNI_KIT_ACCEPT_EULA="Y"
CMD ["/entrypoint.sh", "--headless", "--task=CognitiveRL-Nav-v2", "--num_envs=8"]

# docker build --build-arg REPO_URL=https://github.com/chamorajg/cognitiverl.git --build-arg REPO_BRANCH=docker --build-arg GITHUB_UNAME=chamorajg --build-arg GITHUB_PAT=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 -t my-vnc-image .
# docker run --gpus all -p 5902:5901 -e USER=vncuser -e PASSWORD=vncpassword my-vnc-image