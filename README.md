# World2Filter

A PyTorch implementation of DreamerV3 with MDP-correlated visual distractions (ColorGrid) for robust visual representation learning.

## Overview

World2Filter implements DreamerV3 with advanced visual distraction mechanisms based on psp_camera_ready's proven ColorGrid system. The ColorGrid background correlates with MDP variables (actions, rewards, timesteps), providing a much stronger test of foreground/background separation than standard visual distractions.

## Key Features

- **DreamerV3 Implementation**: Full PyTorch implementation with discrete latent states (RSSM)
- **ColorGrid Distractions**: MDP-correlated background changes that test foreground/background separation
  - `max`: Background correlates with both action AND reward (hardest)
  - `action`: Background correlates with action
  - `reward`: Background correlates with reward
  - `sequence`: Background correlates with timestep
  - `minimum`: Random background each step
- **Multi-GPU Training**: Distributed Data Parallel (DDP) support for 8 GPU training
- **Embodied Interface**: Compatible with DreamerV3's embodied framework
- **WandB Logging**: Comprehensive experiment tracking with 20+ metrics

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

### Training DreamerV3

```bash
# Single GPU training (for testing on headless server)
python scripts/train_dreamer.py --config configs/default.yaml

# 8 GPU training (production)
torchrun --nproc_per_node=8 scripts/train_dreamer.py \
    --config configs/default.yaml

# Train with specific environment
python scripts/train_dreamer.py \
    --config configs/default.yaml \
    environment.domain=cheetah \
    environment.task=run

# Train with different ColorGrid evil levels
python scripts/train_dreamer.py \
    --config configs/default.yaml \
    environment.evil_level=action  # Options: max, action, reward, sequence, minimum, none
```

### ColorGrid Configuration

The ColorGrid system creates MDP-correlated visual distractions:

```yaml
# configs/distracting_cs.yaml
environment:
  use_color_grid: True
  evil_level: max              # Hardest: correlates with action + reward
  num_cells_per_dim: 16        # 16×16 grid of color cells
  num_colors_per_cell: 11664   # Total color patterns (3^6 * reward_bins)
  action_dims_to_split: null   # null = use all action dimensions
  action_power: 3              # Discretization per action dim
```

**Evil Levels**:
- `max`: Background = f(action, reward) - Hardest test
- `action`: Background = f(action) - Tests action invariance
- `reward`: Background = f(reward) - Tests reward invariance
- `sequence`: Background = f(timestep) - Tests temporal invariance
- `minimum`: Random background - Minimal distraction
- `none`: No distraction - Baseline

### Training World2Filter

```bash
# Train World2Filter with SAM3 segmentation (future work)
python scripts/train_world2filter.py \
    --config configs/default.yaml

# Without SAM3 (uses fallback heuristic segmentation)
python scripts/train_world2filter.py \
    --config configs/default.yaml \
    --no-sam3
```

### Parallel Training

The training pipeline supports multi-GPU distributed training using PyTorch DDP.

#### Single GPU (Headless Server)

For headless servers with limited EGL contexts (can't run multiple parallel environments):

```bash
# Adjusted config for single env
python scripts/train_dreamer.py \
    --config configs/default.yaml \
    environment.num_envs=1 \
    training.batch_size=2 \
    training.train_ratio=64.0
```

**Note**: Single environment training uses:
- Smaller `batch_size=2` (faster buffer fill)
- Higher `train_ratio=64.0` (compensates for slower data collection)
- Accumulation-based training (waits for enough sequences, not just episodes)

#### 8 GPU Training (Production)

For production training with 8 GPUs:

```bash
# 8 GPUs, 1 environment per GPU (default config)
torchrun --nproc_per_node=8 scripts/train_dreamer.py \
    --config configs/default.yaml

# 8 GPUs, 2 environments per GPU (16 total)
torchrun --nproc_per_node=8 scripts/train_dreamer.py \
    --config configs/default.yaml \
    environment.num_envs=2
```

**Effective Configuration**:
- `batch_size: 16` per GPU → Effective batch size: **128** (16 × 8)
- `num_envs: 8` (1 per GPU) → Data collection: **8× faster**
- `train_ratio: 32.0` → Trains 32× relative to data collection
- Gradient synchronization: Automatic via DDP

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
| **ColorGrid Distractions** | MDP-correlated backgrounds for robust testing |
| **Single WandB logging** | Only rank 0 logs to WandB (clean dashboard) |
| **Single checkpoint** | Only rank 0 saves checkpoints |
| **Independent replay buffers** | Each GPU has its own buffer for sample diversity |
| **Automatic gradient sync** | DDP handles gradient averaging |
| **Sequence-based sampling** | Can sample multiple sequences from single episode |
| **Portable checkpoints** | Saved without DDP wrapper, loadable on any setup |

#### Effective Batch Size & Training Speed

With multi-GPU training, the effective batch size scales:

| Config `batch_size` | GPUs | Effective Batch | Training Speedup |
|---------------------|------|-----------------|------------------|
| 16 | 1 | 16 | 1× |
| 16 | 4 | 64 | ~3.5× |
| 16 | 8 | 128 | ~7× |

**Training Speed Formula**:
- Data collection: `num_gpus × num_envs_per_gpu` environments
- Gradient steps: `train_ratio × num_envs / batch_length` per env step
- Example (8 GPUs): 8 envs collect data → 32 gradient steps per env step

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

All metrics are logged to WandB every 300 steps:

| Category | Metrics | Description |
|----------|---------|-------------|
| **World Model** | `wm/total_loss`, `wm/image_loss`, `wm/kl_loss`, `wm/kl_value` | Reconstruction and KL divergence |
| | `wm/reward_loss`, `wm/continue_loss` | Reward and continuation prediction |
| | `wm/prior_entropy`, `wm/posterior_entropy` | Latent state entropy |
| | `wm/wm_grad_norm` | Gradient norm for stability monitoring |
| **Actor-Critic** | `ac/actor_loss`, `ac/pg_loss` | Policy gradient loss |
| | `ac/critic_loss` | Value function loss |
| | `ac/entropy`, `ac/log_prob` | Policy entropy and log probability |
| | `ac/advantage_mean`, `ac/advantage_std` | Advantage statistics |
| | `ac/value_mean`, `ac/return_mean` | Value predictions |
| | `ac/actor_grad_norm`, `ac/critic_grad_norm` | Gradient norms |
| **Episode** | `episode/return`, `episode/length` | Episode statistics during training |
| **Evaluation** | `eval/return_mean`, `eval/return_std`, `eval/length_mean` | Evaluation performance (every 100k steps) |

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
- **RSSM**: Recurrent State Space Model with 4096-dim deterministic and 32×32 discrete latent states
- **Encoder**: ResNet-style CNN encoder (96→192→384→768 channels)
- **Decoder**: ResNet-style CNN decoder (768→384→192→96 channels)
- **Reward Head**: 5-layer MLP with symlog discrete distribution (255 bins)
- **Continue Head**: 5-layer MLP for episode continuation prediction

### ColorGrid Background System

From psp_camera_ready's proven implementation:
- **Grid-based**: 16×16 grid of colored cells
- **MDP-correlated**: Each cell color is determined by:
  - `action`: Discretized action dimensions (e.g., 3^6 = 729 combinations)
  - `reward`: Discretized reward bins
  - `timestep`: Sequence position
- **Evil levels**: Different correlation patterns to test different aspects

### Training Paradigm

Following DreamerV3's design:
- **train_ratio**: 32 gradient updates per environment step
- **Sequence sampling**: Samples 64-step sequences from episode replay buffer
- **Can sample from partial episodes**: Uses sequence-based sampling, not episode count
- **Accumulation**: With single env, accumulates fractional training steps across iterations

### World2Filter Extension (Future Work)

Planned additions:
- **SAM3 Preprocessing**: Generate foreground/background masks
- **Dual Decoders**: Separate reconstruction for foreground and background  
- **Mask-Guided Loss**: Weighted reconstruction loss based on segmentation masks

## Configuration

Production configuration for 8 GPU training:

```yaml
# configs/default.yaml
training:
  total_steps: 1_000_000      # 1M steps (standard for DMC)
  batch_size: 16              # Per GPU (effective: 128 with 8 GPUs)
  batch_length: 64            # Sequence length for BPTT
  train_ratio: 32.0           # 32 gradient updates per env step
  log_every: 300              # Log every 300 steps
  
# configs/distracting_cs.yaml
environment:
  num_envs: 8                 # 1 environment per GPU
  use_color_grid: True
  evil_level: max             # Hardest distraction level
  
world_model:
  rssm:
    deter_size: 4096          # Deterministic state size
    stoch_size: 32            # Stochastic dimensions
    classes: 32               # Classes per stochastic dimension
    
actor:
  layers: 5
  units: 1024
  entropy_scale: 3e-4         # Entropy bonus for exploration

imagination:
  horizon: 15                 # Imagination rollout length
  discount: 0.997             # Discount factor
```

### Headless Server Configuration

For single-GPU headless servers with EGL limitations:

```yaml
# Single environment setup
environment:
  num_envs: 1
  
training:
  batch_size: 2               # Reduced for faster buffer fill
  train_ratio: 64.0           # Compensate for slower data collection
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
