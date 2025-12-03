# World2Filter

PyTorch implementation of DreamerV3 with comprehensive visual distraction benchmarks for robust representation learning.

## Features

- **DreamerV3**: Full PyTorch implementation (RSSM, discrete latents, actor-critic)
- **8 Benchmark Types**: Standard DMC + 7 ColorGrid distraction levels + Natural videos
- **30+ Tasks**: DMC (27 tasks) + RLBench (reach_target)
- **Multi-GPU**: DDP support for 8-GPU training
- **ColorGrid System**: MDP-correlated backgrounds from psp_camera_ready

## Installation

### Docker (Recommended)

```bash
# Build image
docker build -t world2filter:latest .

# Run with GPU
docker run --gpus all -it --rm \
    -v $(pwd):/workspace/World2Filter \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    world2filter:latest

# Inside container
python scripts/train_dreamer.py --config configs/default.yaml
```

### Local Installation

```bash
# Clone repository
git clone https://github.com/yourusername/World2Filter.git
cd World2Filter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Install dm_control (skip labmaze which requires bazel)
pip install dm_control --no-deps
pip install dm-env dm-tree lxml scipy

# Install moviepy for video logging
pip install moviepy
```

### Optional Components

```bash
# Natural video backgrounds
pip install scikit-video opencv-python

# RLBench robotic tasks
pip install git+https://github.com/stepjam/RLBench.git
pip install git+https://github.com/stepjam/PyRep.git

# SAM3 segmentation (for World2Filter variant)
pip install git+https://github.com/facebookresearch/sam3.git
```

### Verify Installation

```bash
# Test DMControl
python -c "from dm_control import suite; env = suite.load('walker', 'walk'); print('✓ DMControl OK')"

# Test World2Filter
python -c "from src.models.dreamer_v3.world_model import WorldModel; print('✓ World2Filter OK')"
```

## Quick Start

### Debug Mode (5 minutes, single GPU)

```bash
# Quick test to verify everything works
python scripts/train_dreamer.py --config configs/debug.yaml

# Expected output: Training completes in ~5 minutes with metrics logged every 10 steps
```

### Standard Training

```bash
# Baseline (no distractions)
python scripts/train_dreamer.py --config configs/default.yaml \
    environment.use_color_grid=False

# Hardest benchmark (action × reward correlation)
python scripts/train_dreamer.py --config configs/default.yaml \
    environment.evil_level=max

# 8 GPU training
torchrun --nproc_per_node=8 scripts/train_dreamer.py \
    --config configs/default.yaml
```

## Benchmarks

### Benchmark Types

| Type | Distraction | Difficulty | Description | Command |
|------|-------------|------------|-------------|---------|
| **Standard** | None | Baseline | Clean observations | `use_color_grid=False` |
| **ColorGrid Max** | Action × Reward | Hardest | Background = f(action, reward) | `evil_level=max` |
| **ColorGrid Action** | Action | Hard | Background = f(action) | `evil_level=action` |
| **ColorGrid Reward** | Reward | Hard | Background = f(reward) | `evil_level=reward` |
| **ColorGrid Sequence** | Timestep | Medium | Background = f(timestep) | `evil_level=sequence` |
| **ColorGrid Action×Seq** | Action × Time | Hard | Background = f(action, t) | `evil_level=action_cross_sequence` |
| **ColorGrid Minimum** | Random | Easy | Random each step | `evil_level=minimum` |
| **Natural Video** | Real videos | Medium | Kinetics dataset | `evil_level=natural` |

### Task Suite (28 Tasks)

**Locomotion (11 tasks)**
- `walker`: walk, run, stand
- `cheetah`: run
- `hopper`: stand, hop
- `humanoid`: stand, walk, run
- `quadruped`: walk, run

**Manipulation (6 tasks)**
- `reacher`: easy, hard
- `finger`: spin, turn_easy, turn_hard
- `ball_in_cup`: catch

**Control (10 tasks)**
- `cartpole`: swingup, swingup_sparse, balance, balance_sparse
- `acrobot`: swingup
- `pendulum`: swingup

**RLBench (1 task)**
- `reach_target`: robotic manipulation

## Usage Examples

### Different Tasks

```bash
# Locomotion
python scripts/train_dreamer.py --config configs/default.yaml \
    environment.domain=walker environment.task=walk

python scripts/train_dreamer.py --config configs/default.yaml \
    environment.domain=cheetah environment.task=run

python scripts/train_dreamer.py --config configs/default.yaml \
    environment.domain=humanoid environment.task=walk

# Manipulation
python scripts/train_dreamer.py --config configs/default.yaml \
    environment.domain=reacher environment.task=easy

python scripts/train_dreamer.py --config configs/default.yaml \
    environment.domain=finger environment.task=spin

# Control
python scripts/train_dreamer.py --config configs/default.yaml \
    environment.domain=cartpole environment.task=swingup
```

### Different Distraction Levels

```bash
# No distraction (baseline)
python scripts/train_dreamer.py --config configs/default.yaml \
    environment.use_color_grid=False

# Action-correlated
python scripts/train_dreamer.py --config configs/default.yaml \
    environment.evil_level=action

# Reward-correlated
python scripts/train_dreamer.py --config configs/default.yaml \
    environment.evil_level=reward

# Maximum evil (action × reward)
python scripts/train_dreamer.py --config configs/default.yaml \
    environment.evil_level=max

# Minimum distraction (random)
python scripts/train_dreamer.py --config configs/default.yaml \
    environment.evil_level=minimum
```

### Natural Video Backgrounds

```bash
# Download Kinetics dataset (optional)
git clone https://github.com/Showmax/kinetics-downloader
cd kinetics-downloader
python download.py --categories "driving_car,walking,running" --num-workers 4

# Use natural videos
python scripts/train_dreamer.py --config configs/default.yaml \
    environment.evil_level=natural \
    environment.natural_video_dir="/path/to/kinetics/train/driving_car/*.mp4" \
    environment.total_natural_frames=1000
```

### RLBench Tasks

```bash
# Robotic manipulation with RLBench
python scripts/train_dreamer.py --config configs/default.yaml \
    environment.suite=rlbench \
    environment.task=reach_target \
    environment.obs.image_size=64 \
    environment.obs.action_repeat=2
```

### Training Duration Options

```bash
# Quick debug (500 steps, ~5 minutes)
python scripts/train_dreamer.py --config configs/debug.yaml

# Short training (100k steps, ~9 hours on single GPU)
python scripts/train_dreamer.py --config configs/default.yaml \
    training.total_steps=100_000

# Standard training (1M steps, ~4 days on single GPU, ~12 hours on 8 GPUs)
python scripts/train_dreamer.py --config configs/default.yaml \
    training.total_steps=1_000_000
```

### WandB Logging Options

```bash
# Custom project name
python scripts/train_dreamer.py --config configs/default.yaml \
    logging.wandb_project=my_experiment

# Disable WandB
python scripts/train_dreamer.py --config configs/default.yaml \
    logging.use_wandb=False

# Minimal logging (faster)
python scripts/train_dreamer.py --config configs/default.yaml \
    logging.log_images=False \
    logging.log_videos=False
```

## Evaluation Protocol

### Phase 1: Quick Validation (Easy Tasks, 100k steps)

```bash
# Verify implementation on fast-converging tasks
for task in cartpole_swingup reacher_easy finger_spin; do
  domain=$(echo $task | cut -d_ -f1)
  task_name=$(echo $task | cut -d_ -f2-)
  python scripts/train_dreamer.py --config configs/default.yaml \
      environment.domain=$domain \
      environment.task=$task_name \
      training.total_steps=100_000 \
      logging.wandb_project=phase1_validation
done
```

### Phase 2: Distraction Ablation (All Evil Levels, walker_walk)

```bash
# Test all distraction types on walker_walk
for evil in none action reward sequence action_cross_sequence minimum max; do
  if [ "$evil" = "none" ]; then
    python scripts/train_dreamer.py --config configs/default.yaml \
        environment.use_color_grid=False \
        logging.wandb_project=distraction_ablation
  else
    python scripts/train_dreamer.py --config configs/default.yaml \
        environment.evil_level=$evil \
        logging.wandb_project=distraction_ablation
  fi
done
```

### Phase 3: Full Benchmark (Medium Tasks, Hardest Distraction)

```bash
# Comprehensive evaluation on multiple tasks
for task in walker_walk walker_run cheetah_run hopper_stand; do
  domain=$(echo $task | cut -d_ -f1)
  task_name=$(echo $task | cut -d_ -f2-)
  python scripts/train_dreamer.py --config configs/default.yaml \
      environment.domain=$domain \
      environment.task=$task_name \
      environment.evil_level=max \
      logging.wandb_project=full_benchmark
done
```

### Phase 4: Multi-Task Sweep (All Tasks × Multiple Distractions)

```bash
# Create comprehensive benchmark script
cat > run_benchmark.sh << 'EOF'
#!/bin/bash
TASKS="walker_walk cheetah_run hopper_stand cartpole_swingup"
EVILS="none action max"

for task in $TASKS; do
  domain=$(echo $task | cut -d_ -f1)
  task_name=$(echo $task | cut -d_ -f2-)
  
  for evil in $EVILS; do
    if [ "$evil" = "none" ]; then
      python scripts/train_dreamer.py --config configs/default.yaml \
          environment.domain=$domain \
          environment.task=$task_name \
          environment.use_color_grid=False \
          logging.wandb_project=multi_task_sweep &
    else
      python scripts/train_dreamer.py --config configs/default.yaml \
          environment.domain=$domain \
          environment.task=$task_name \
          environment.evil_level=$evil \
          logging.wandb_project=multi_task_sweep &
    fi
    
    # Limit parallel jobs (adjust based on GPU count)
    if (( $(jobs -r | wc -l) >= 4 )); then
      wait -n
    fi
  done
done
wait
EOF

chmod +x run_benchmark.sh
./run_benchmark.sh
```

## Configuration

### Production (8 GPUs)

```yaml
# configs/default.yaml
training:
  total_steps: 1_000_000      # 1M steps
  batch_size: 16              # Per GPU (128 effective)
  train_ratio: 32.0           # 32 gradient steps per env step
  log_every: 300              # Log every 300 steps
  
environment:
  num_envs: 8                 # 1 per GPU
  use_color_grid: True
  evil_level: max
```

### Headless Server (Single GPU)

```yaml
# Optimized for headless server with limited EGL contexts
environment:
  num_envs: 1                 # Single environment

training:
  batch_size: 2               # Smaller batch
  train_ratio: 64.0           # Higher ratio compensates for slower data
  prefill_steps: 1000         # Standard prefill
```

### Debug Mode (Fast Testing)

```yaml
# configs/debug.yaml
training:
  total_steps: 500            # Very short
  prefill_steps: 200          # Minimal prefill
  batch_size: 2
  train_ratio: 32.0
  log_every: 10               # Frequent logging
  
logging:
  wandb_project: debug_world2filter
  image_log_freq: 100
  video_log_freq: 200
```

## Multi-GPU Training

### Single Node (8 GPUs)

```bash
# Standard 8 GPU training
torchrun --nproc_per_node=8 scripts/train_dreamer.py \
    --config configs/default.yaml

# With custom config overrides
torchrun --nproc_per_node=8 scripts/train_dreamer.py \
    --config configs/default.yaml \
    environment.evil_level=max \
    training.total_steps=1_000_000
```

### Multi-Node (2 Nodes × 8 GPUs = 16 GPUs)

```bash
# Node 0 (master)
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=192.168.1.100 --master_port=29500 \
    scripts/train_dreamer.py --config configs/default.yaml

# Node 1 (worker)
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
    --master_addr=192.168.1.100 --master_port=29500 \
    scripts/train_dreamer.py --config configs/default.yaml
```

### Training Speed Comparison

| Setup | Environments | Effective Batch | Time (1M steps) |
|-------|-------------|----------------|-----------------|
| 1 GPU, 1 env | 1 | 2 | ~4 days |
| 1 GPU, 4 env | 4 | 2 | ~2 days |
| 8 GPU, 8 env | 8 | 128 | ~12 hours |
| 16 GPU, 16 env | 16 | 256 | ~6 hours |

## WandB Logging

### Metrics Tracked (20+)

**World Model**
- `wm/total_loss`, `wm/image_loss`, `wm/kl_loss`, `wm/kl_value`
- `wm/reward_loss`, `wm/continue_loss`
- `wm/prior_entropy`, `wm/posterior_entropy`
- `wm/wm_grad_norm`

**Actor-Critic**
- `ac/actor_loss`, `ac/pg_loss`, `ac/critic_loss`
- `ac/entropy`, `ac/log_prob`
- `ac/advantage_mean`, `ac/advantage_std`
- `ac/value_mean`, `ac/return_mean`
- `ac/actor_grad_norm`, `ac/critic_grad_norm`

**Episodes & Evaluation**
- `episode/return`, `episode/length`
- `eval/return_mean`, `eval/return_std`, `eval/length_mean`

**Visual Outputs**
- Original images, Reconstructed images
- Observation videos, Reconstruction videos
- Imagination videos (with comparison)

### View Results

```bash
# Check WandB dashboard
wandb login
# Visit: https://wandb.ai/your_username/your_project
```

## Troubleshooting

### Headless Server EGL Errors

```bash
# If you see: EGLError: err = EGL_BAD_ALLOC
# Solution: Use single environment
python scripts/train_dreamer.py --config configs/default.yaml \
    environment.num_envs=1 \
    training.batch_size=2 \
    training.train_ratio=64.0
```

### Out of Memory

```bash
# Reduce batch size or model size
python scripts/train_dreamer.py --config configs/default.yaml \
    training.batch_size=1 \
    world_model.rssm.deter_size=2048
```

### Slow Training

```bash
# Use smaller model for faster iteration
python scripts/train_dreamer.py --config configs/debug.yaml \
    training.total_steps=10000
```

## Comparison with psp_camera_ready

| Feature | psp_camera_ready | World2Filter |
|---------|------------------|--------------|
| **Framework** | JAX | PyTorch |
| **Algorithms** | 5 (DV3, DrQv2, TIA, Dreamer Pro, Denoised MDP) | 1 (DreamerV3) |
| **DMC Tasks** | ~10 | 27 |
| **RLBench** | reach_target | reach_target |
| **ColorGrid** | All 7 levels | All 7 levels |
| **Natural Videos** | Kinetics | Kinetics |
| **Multi-GPU** | Limited | Full DDP support |
| **Documentation** | Minimal | Comprehensive |

## Citation

```bibtex
@article{hafner2023dreamerv3,
  title={Mastering Diverse Domains through World Models},
  author={Hafner, Danijar and others},
  journal={arXiv preprint arXiv:2301.04104},
  year={2023}
}
```

## License

MIT License
