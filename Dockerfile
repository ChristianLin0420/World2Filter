# World2Filter Docker Image
# DreamerV3 with SAM3 segmentation for visual RL
#
# Build: docker build -t world2filter .
# Run:   docker run --gpus all -it world2filter

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set up locale
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python and build tools
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    # MuJoCo dependencies
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libosmesa6-dev \
    libglew-dev \
    libglfw3-dev \
    patchelf \
    # EGL rendering (for headless GPU rendering)
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    # FFmpeg for video encoding
    ffmpeg \
    # Other utilities
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set MuJoCo rendering backend for headless GPU rendering
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# Create working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install dm_control with dependencies (skip labmaze)
RUN pip install --no-cache-dir dm_control --no-deps && \
    pip install --no-cache-dir dm-env dm-tree lxml scipy

# Install moviepy for video logging
RUN pip install --no-cache-dir moviepy

# Copy source code
COPY . .
RUN pip install -e .
RUN pip install gym

ENV WANDB_DIR=/workspace/World2Filter/logs

# Default command
CMD ["/bin/bash"]

