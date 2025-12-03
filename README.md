# World2Filter

A PyTorch implementation of DreamerV3 world model with SAM3-based foreground/background segmentation for improved visual representation learning.

## Overview

World2Filter extends the DreamerV3 architecture by incorporating Meta's SAM3 (Segment Anything Model 3) to disentangle foreground (agent) and background representations. This enables the world model to learn separate latent representations for the controllable agent and the environment background, improving generalization in visually complex environments.

## Features

- **Original DreamerV3**: Full PyTorch implementation of DreamerV3 with discrete latent states
- **SAM3 Integration**: Preprocessing pipeline using SAM3 for foreground/background segmentation
- **Dual-Decoder Architecture**: Separate reconstruction heads for foreground and background
- **DistractingCS Support**: Training on Distracting Control Suite for robust visual learning
- **WandB Logging**: Comprehensive logging with scalars, images, videos, and histograms

## Installation

### Option 1: Docker (Recommended)

The easiest way to get started is using Docker with GPU support:

```bash
# Clone the repository
git clone https://github.com/yourusername/World2Filter.git
cd World2Filter

# Build the Docker image
docker build -t world2filter:latest .

# (Optional) Tag the image for your registry (e.g., Docker Hub)
docker tag world2filter:latest yourusername/world2filter:latest

# Start interactive container (mounts entire project)
docker run --gpus all -it --rm \
    --name w2f-container \
    -v $(pwd):/workspace/World2Filter \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    world2filter:latest

# Inside container, run training:
python scripts/train_dreamer.py --config configs/default.yaml
```

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/World2Filter.git
cd World2Filter

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install base dependencies
pip install -r requirements.txt

# Install dm_control (skip labmaze which requires bazel)
pip install dm_control --no-deps
pip install dm-env dm-tree lxml scipy

# Install moviepy for video logging
pip install moviepy

# (Optional) Install SAM3 for World2Filter with segmentation
pip install git+https://github.com/facebookresearch/sam3.git
```

### Verifying Installation

```bash
# Test that dm_control works
python -c "from dm_control import suite; env = suite.load('walker', 'walk'); print('DMControl OK!')"

# Test World2Filter modules
python -c "from src.models.dreamer_v3.world_model import WorldModel; print('World2Filter OK!')"
```

**Notes**:
- The `labmaze` package is skipped as it requires `bazel` to build. Standard DMControl tasks (walker, cheetah, cartpole, etc.) work without it.
- If SAM3 installation fails, the code will use a fallback heuristic segmentation method.
- For headless servers, you may see GLFW warnings about missing display - this is normal and rendering still works via offscreen buffers.

## Project Structure

```
World2Filter/
├── configs/
│   ├── default.yaml              # Base configuration (training, logging, replay)
│   ├── dreamer_v3.yaml           # DreamerV3 hyperparameters (RSSM, encoder, decoder, actor-critic)
│   └── distracting_cs.yaml       # Environment and segmentation settings
├── scripts/
│   ├── train_dreamer.py          # Train original DreamerV3
│   ├── train_world2filter.py     # Train World2Filter with FG/BG segmentation
│   └── preprocess_masks.py       # Pre-generate SAM3 masks for dataset
├── src/
│   ├── models/
│   │   ├── dreamer_v3/
│   │   │   ├── rssm.py           # Recurrent State Space Model (discrete latents)
│   │   │   ├── encoder.py        # CNN image encoder
│   │   │   ├── decoder.py        # CNN image decoder
│   │   │   ├── actor_critic.py   # Actor-Critic networks with EMA target
│   │   │   ├── networks.py       # MLP, GRU, distributions, symlog
│   │   │   └── world_model.py    # Complete DreamerV3 world model
│   │   ├── segmentation/
│   │   │   ├── sam3_wrapper.py   # SAM3 inference wrapper
│   │   │   └── mask_processor.py # Mask caching and preprocessing
│   │   └── world2filter/
│   │       ├── fg_bg_decoder.py      # Dual foreground/background decoders
│   │       └── fg_bg_world_model.py  # World2Filter world model
│   ├── agents/
│   │   ├── dreamer_agent.py      # DreamerV3 agent for environment interaction
│   │   └── world2filter_agent.py # World2Filter agent with mask processing
│   ├── envs/
│   │   ├── wrappers.py           # DMControl wrappers (action repeat, frame stack)
│   │   └── distracting_cs.py     # Distracting Control Suite setup
│   ├── data/
│   │   ├── replay_buffer.py      # Episode-based experience replay
│   │   └── episode_dataset.py    # PyTorch dataset for episodes
│   ├── training/
│   │   ├── trainer.py            # Training loop (world model + actor-critic)
│   │   └── losses.py             # All loss functions (reconstruction, KL, FG/BG)
│   └── utils/
│       ├── config.py             # Configuration loading (Hydra/OmegaConf)
│       ├── logging.py            # WandB logger with images/videos/histograms
│       └── visualization.py      # Reconstruction grids, video generation
├── requirements.txt
└── README.md
```

## Usage

### Training Original DreamerV3

```bash
# Basic training (without WandB)
python scripts/train_dreamer.py \
    --config configs/default.yaml \
    --no-wandb

# Training with specific environment
python scripts/train_dreamer.py \
    --config configs/default.yaml \
    environment.domain=cheetah \
    environment.task=run
```

### Training World2Filter

```bash
# Train World2Filter with SAM3 segmentation
python scripts/train_world2filter.py \
    --config configs/default.yaml

# Without SAM3 (uses fallback heuristic segmentation)
python scripts/train_world2filter.py \
    --config configs/default.yaml \
    --no-sam3
```

### Parallel Training

The training pipeline supports two types of parallelism that can be combined:

1. **Parallel Environments** - Multiple CPU processes for faster data collection (single GPU)
2. **Multi-GPU (DDP)** - Distributed training across multiple GPUs

#### Parallel Environments (Single GPU)

Speed up data collection by running multiple environments in parallel on CPU:

```bash
# Train with 8 parallel environments on 1 GPU
python scripts/train_dreamer.py \
    --config configs/default.yaml \
    environment.num_envs=8

# World2Filter with parallel envs
python scripts/train_world2filter.py \
    --config configs/default.yaml \
    environment.num_envs=8
```

#### Multi-GPU Training (DDP)

Use `torchrun` to distribute training across multiple GPUs:

```bash
# Train on 8 GPUs (each GPU has 1 environment)
torchrun --nproc_per_node=8 scripts/train_dreamer.py \
    --config configs/default.yaml

# Train on 4 GPUs with 4 parallel envs per GPU (16 total envs)
torchrun --nproc_per_node=4 scripts/train_dreamer.py \
    --config configs/default.yaml \
    environment.num_envs=4

# World2Filter on multiple GPUs
torchrun --nproc_per_node=8 scripts/train_world2filter.py \
    --config configs/default.yaml
```

#### Multi-Node Training

For training across multiple machines:

```bash
# On node 0 (master)
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=<MASTER_IP> --master_port=29500 \
    scripts/train_dreamer.py --config configs/default.yaml

# On node 1
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
    --master_addr=<MASTER_IP> --master_port=29500 \
    scripts/train_dreamer.py --config configs/default.yaml
```

#### Key Features

| Feature | Description |
|---------|-------------|
| **Single WandB logging** | Only rank 0 logs to WandB (clean dashboard) |
| **Single checkpoint** | Only rank 0 saves checkpoints |
| **Independent replay buffers** | Each GPU has its own buffer for sample diversity |
| **Automatic gradient sync** | DDP handles gradient averaging |
| **Portable checkpoints** | Saved without DDP wrapper, loadable on any setup |
| **Flexible design** | Works with any algorithm (DreamerV3, World2Filter, etc.) |

#### Effective Batch Size

With multi-GPU training, the effective batch size scales:

| Config `batch_size` | GPUs | Effective Batch |
|---------------------|------|-----------------|
| 64 | 1 | 64 |
| 64 | 4 | 256 |
| 64 | 8 | 512 |

### Resuming Training

Training automatically saves checkpoints. If training is interrupted, you can resume:

```bash
# Auto-resume from latest checkpoint
python scripts/train_dreamer.py --config configs/default.yaml --resume

# Resume from specific checkpoint
python scripts/train_dreamer.py --config configs/default.yaml \
    --checkpoint checkpoints/checkpoint_50000.pt

# Using Docker
docker-compose up resume
```

**Resume Features**:
- Automatically resumes from the latest `checkpoint_latest.pt`
- Restores WandB run to continue logging to the same experiment
- Keeps last 3 checkpoints to save disk space
- Saves emergency checkpoint on Ctrl+C or errors

## WandB Logging

The project uses [Weights & Biases](https://wandb.ai) for experiment tracking with comprehensive logging.

### Setup

```bash
# Login to WandB (first time only)
wandb login

# Or set API key via environment variable
export WANDB_API_KEY=your_api_key_here
```

### Configuration

WandB settings can be configured in `configs/default.yaml`:

```yaml
logging:
  use_wandb: True
  wandb_project: world2filter    # Project name on WandB
  wandb_entity: null             # Team/username (null = default account)
  log_scalars: True
  log_images: True
  log_videos: True
  log_histograms: True
  image_log_freq: 5000           # Log images every N steps
  video_log_freq: 10000          # Log videos every N steps
```

### Command Line Options

```bash
# Enable WandB logging
python scripts/train_dreamer.py --config configs/default.yaml --wandb

# Disable WandB logging
python scripts/train_dreamer.py --config configs/default.yaml --no-wandb

# Override project and run name via config
python scripts/train_dreamer.py \
    --config configs/default.yaml \
    logging.wandb_project=my_project \
    logging.wandb_entity=my_team

# Set custom seed (affects run name)
python scripts/train_dreamer.py --config configs/default.yaml --seed 123
```

### Run Naming

Runs are automatically named based on the experiment configuration:
- **DreamerV3**: `dreamer_{domain}_{task}_{seed}` (e.g., `dreamer_walker_walk_42`)
- **World2Filter**: `world2filter_{domain}_{task}_{seed}` (e.g., `world2filter_cheetah_run_123`)

### Logged Metrics

| Category | Metrics |
|----------|---------|
| **World Model** | `wm/total_loss`, `wm/image_loss`, `wm/kl_loss`, `wm/kl_value`, `wm/reward_loss`, `wm/continue_loss`, `wm/grad_norm` |
| **Actor-Critic** | `ac/actor_loss`, `ac/critic_loss`, `ac/entropy`, `ac/value_mean`, `ac/advantage_mean` |
| **Episode** | `episode/return`, `episode/length` |
| **Evaluation** | `eval/return_mean`, `eval/return_std`, `eval/length_mean` |
| **World2Filter** | `wm/fg_loss`, `wm/bg_loss` (foreground/background reconstruction) |

### Visual Logging

| Media Type | WandB Keys | Description |
|------------|-----------|-------------|
| **Original Image** | `train/original` | Ground truth observation from replay buffer |
| **Reconstructed Image** | `train/reconstructed` | World model reconstruction from posterior |
| **Observation Video** | `train/observation_video` | Actual environment frames |
| **Reconstruction Video** | `train/reconstruction_video` | Decoded frames from posterior states |
| **Imagined Video** | `train/imagination/imagined` | Future frames predicted by world model + policy |
| **Imagination Comparison** | `train/imagination/comparison` | Side-by-side: ground truth (left) vs imagined (right) |
| **Segmentation Masks** | `segmentation/*` | FG/BG mask overlays (World2Filter only) |

#### Imagination Video (Key DreamerV3 Feature)

The imagination video shows the model's ability to predict future frames:
1. **Context**: First 5 frames are used to establish the world model's internal state
2. **Imagination**: The model then imagines the next 15 frames using:
   - The learned world model dynamics (RSSM)
   - The current policy to select actions
3. **Comparison**: Imagined frames are compared with actual ground truth futures

This visualization is crucial for debugging world model quality - a well-trained model should produce plausible future predictions that match the ground truth structure.

## Architecture

### DreamerV3 World Model

The world model consists of:
- **RSSM**: Recurrent State Space Model with 32x32 discrete latent states
- **Encoder**: CNN-based image encoder
- **Decoder**: CNN-based image decoder with symlog predictions
- **Reward/Continue Heads**: MLP predictors for reward and episode continuation

### World2Filter Extension

World2Filter adds:
- **SAM3 Preprocessing**: Generate foreground/background masks
- **Dual Decoders**: Separate reconstruction for foreground and background
- **Mask-Guided Loss**: Weighted reconstruction loss based on segmentation masks

## Configuration

Key hyperparameters can be modified in the YAML config files:

```yaml
# configs/dreamer_v3.yaml
world_model:
  rssm:
    deter_size: 4096      # Deterministic state size
    stoch_size: 32        # Stochastic dimensions
    classes: 32           # Classes per stochastic dimension
    
actor:
  layers: 5
  units: 1024
  entropy_scale: 3e-4     # Entropy bonus for exploration

imagination:
  horizon: 15             # Imagination rollout length
  discount: 0.997         # Discount factor
```

## Docker Usage

### Building

```bash
docker build -t world2filter .
```

### Running Container

The Docker setup mounts the entire project into the container, so any changes to code, logs, or checkpoints are persisted on your host machine.

```bash
# Start interactive container with GPU support
docker-compose run --rm world2filter bash

# Or run a command directly
docker-compose run --rm world2filter python scripts/train_dreamer.py --config configs/default.yaml
```

### Inside Container

Once inside the container, use the same commands as local installation:

```bash
# Train DreamerV3
python scripts/train_dreamer.py --config configs/default.yaml

# Train specific environment
python scripts/train_dreamer.py --config configs/default.yaml \
    environment.domain=walker environment.task=walk

# Resume from checkpoint
python scripts/train_dreamer.py --config configs/default.yaml --resume

# Train World2Filter
python scripts/train_world2filter.py --config configs/default.yaml
```

### Environment Variables

Set before running docker-compose:

```bash
export WANDB_API_KEY=your_wandb_api_key
docker-compose run --rm world2filter bash
```

| Variable | Description |
|----------|-------------|
| `WANDB_API_KEY` | WandB API key for logging |

## Citation

If you use this code, please cite:

```bibtex
@article{hafner2023dreamerv3,
  title={Mastering Diverse Domains through World Models},
  author={Hafner, Danijar and others},
  journal={arXiv preprint arXiv:2301.04104},
  year={2023}
}

@misc{carion2025sam3,
  title={SAM 3: Segment Anything with Concepts},
  author={Nicolas Carion and others},
  year={2025},
  eprint={2511.16719},
  archivePrefix={arXiv}
}
```

## License

MIT License
