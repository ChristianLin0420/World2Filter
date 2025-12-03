#!/usr/bin/env python3
"""
Training script for DreamerV3.

This script trains the original DreamerV3 world model and actor-critic
on the Distracting Control Suite environment.

Usage:
    python scripts/train_dreamer.py --config configs/default.yaml
    python scripts/train_dreamer.py environment.domain=cheetah environment.task=run
"""

import os
# Set rendering backend for headless servers BEFORE importing dm_control/mujoco
# Use EGL for NVIDIA GPUs, OSMesa for software rendering
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import torch.distributed as dist

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import random

from src.utils.config import load_config, save_config, print_config
from src.utils.logging import WandbLogger
from src.envs.distracting_cs import make_distracting_cs_env, get_env_info
from src.envs.embodied_wrapper import wrap_embodied
from src.envs.parallel import ParallelEnv
from src.models.dreamer_v3.world_model import WorldModel
from src.models.dreamer_v3.actor_critic import ActorCritic
from src.training.trainer import Trainer
from src.agents.dreamer_agent import DreamerAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DreamerV3")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (overrides config)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=None,
        help="Enable WandB logging",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-resume from latest checkpoint in log directory",
    )
    
    # Allow config overrides via command line
    args, remaining = parser.parse_known_args()
    
    # Parse remaining args as config overrides
    overrides = {}
    for arg in remaining:
        if "=" in arg:
            key, value = arg.split("=", 1)
            # Try to parse value as int/float/bool
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
            
            # Handle nested keys
            keys = key.split(".")
            d = overrides
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
    
    args.overrides = overrides
    return args


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main training function."""
    args = parse_args()
    
    # Initialize distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        print(f"Initialized process group: rank {rank}/{world_size}, local_rank {local_rank}")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        
    # Load configuration
    config_path = project_root / args.config
    config = load_config(config_path, args.overrides)
    
    # Override from command line
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device
    if args.wandb is True:
        config.logging.use_wandb = True
    if args.no_wandb:
        config.logging.use_wandb = False
    
    # Set device
    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Set seed (different for each rank)
    set_seed(config.seed + rank)
    
    # Print configuration
    print("\nConfiguration:")
    print_config(config)
    
    # Create environment to get info
    print("\nCreating environment...")
    
    def make_env():
        env = make_distracting_cs_env(
            domain=config.environment.domain,
            task=config.environment.task,
            image_size=config.environment.obs.image_size,
            action_repeat=config.environment.obs.action_repeat,
            frame_stack=config.environment.obs.frame_stack,
            time_limit=config.environment.time_limit,
            # ColorGrid distractions (MDP-correlated)
            use_color_grid=config.environment.get('use_color_grid', False),
            evil_level=config.environment.get('evil_level', 'none'),
            num_cells_per_dim=config.environment.get('num_cells_per_dim', 16),
            num_colors_per_cell=config.environment.get('num_colors_per_cell', 11664),
            action_dims_to_split=config.environment.get('action_dims_to_split', None),
            action_power=config.environment.get('action_power', 3),
            # Standard distractions (fallback)
            background=config.environment.distractions.get('background', False),
            camera=config.environment.distractions.get('camera', False),
            color=config.environment.distractions.get('color', False),
            seed=config.seed,
        )
        # Wrap with embodied interface
        env = wrap_embodied(env)
        return env
        
    num_envs = config.environment.get('num_envs', 1)
    if num_envs > 1:
        print(f"Using {num_envs} parallel environments")
        env = ParallelEnv([make_env for _ in range(num_envs)])
    else:
        env = make_env()
    
    # Get environment specs from wrapped env
    # For embodied env, we need to extract action dimension from act_space
    act_space = env.act_space if hasattr(env, 'act_space') else env.envs[0].act_space
    obs_space = env.obs_space if hasattr(env, 'obs_space') else env.envs[0].obs_space
    
    action_dim = act_space['action'].shape[0]
    obs_shape = obs_space['image'].shape
    
    print(f"Environment: {config.environment.domain}/{config.environment.task}")
    print(f"Observation shape: {obs_shape}")
    print(f"Action dimension: {action_dim}")
    
    # Create world model
    print("\nBuilding world model...")
    world_model = WorldModel(
        obs_shape=obs_shape,
        action_dim=action_dim,
        deter_size=config.world_model.rssm.deter_size,
        stoch_size=config.world_model.rssm.stoch_size,
        classes=config.world_model.rssm.classes,
        hidden_size=config.world_model.rssm.hidden_size,
        gru_layers=config.world_model.rssm.gru_layers,
        unimix_ratio=config.world_model.rssm.unimix_ratio,
        encoder_channels=list(config.world_model.encoder.channels),
        encoder_kernels=list(config.world_model.encoder.kernels),
        encoder_strides=list(config.world_model.encoder.strides),
        decoder_channels=list(config.world_model.decoder.channels),
        decoder_kernels=list(config.world_model.decoder.kernels),
        decoder_strides=list(config.world_model.decoder.strides),
        output_dist=config.world_model.decoder.output_dist,
        reward_layers=config.world_model.reward_head.layers,
        reward_units=config.world_model.reward_head.units,
        reward_bins=config.world_model.reward_head.bins,
        continue_layers=config.world_model.continue_head.layers,
        continue_units=config.world_model.continue_head.units,
        activation=config.world_model.encoder.act,
        norm=config.world_model.encoder.norm,
    )
    
    feature_dim = world_model.feature_dim
    print(f"World model feature dimension: {feature_dim}")
    
    # Count parameters
    wm_params = sum(p.numel() for p in world_model.parameters())
    print(f"World model parameters: {wm_params:,}")
    
    # Create actor-critic
    print("\nBuilding actor-critic...")
    actor_config = {
        'hidden_dim': config.actor.units,
        'num_layers': config.actor.layers,
        'activation': config.actor.act,
        'norm': config.actor.norm,
        'dist_type': config.actor.dist,
        'min_std': config.actor.min_std,
        'max_std': config.actor.max_std,
        'unimix_ratio': config.actor.unimix_ratio,
    }
    
    critic_config = {
        'hidden_dim': config.critic.units,
        'num_layers': config.critic.layers,
        'activation': config.critic.act,
        'norm': config.critic.norm,
        'dist_type': config.critic.dist,
        'num_bins': config.critic.bins,
        'slow_target': config.critic.slow_target,
        'slow_target_update': config.critic.slow_target_update,
        'slow_target_fraction': config.critic.slow_target_fraction,
    }
    
    actor_critic = ActorCritic(
        feature_dim=feature_dim,
        action_dim=action_dim,
        actor_config=actor_config,
        critic_config=critic_config,
        discount=config.imagination.discount,
        lambda_=config.imagination.lambda_,
        return_normalization=config.imagination.return_normalization,
        return_ema_decay=config.imagination.return_ema_decay,
    )
    
    ac_params = sum(p.numel() for p in actor_critic.parameters())
    print(f"Actor-critic parameters: {ac_params:,}")
    print(f"Total parameters: {wm_params + ac_params:,}")
    
    # Create experiment name for organized folder structure
    experiment_name = f"dreamer_{config.environment.domain}_{config.environment.task}_{config.seed}"
    
    # Setup hierarchical directories: logs/experiment_name/, checkpoints/experiment_name/
    log_dir = Path(config.log_dir) / experiment_name
    checkpoint_dir = Path(config.checkpoint_dir) / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config with experiment-specific paths
    config.log_dir = str(log_dir)
    config.checkpoint_dir = str(checkpoint_dir)
    
    print(f"\nExperiment: {experiment_name}")
    print(f"Log dir: {log_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    
    # Determine checkpoint to resume from
    resume_checkpoint = None
    wandb_run_id = None
    
    if args.checkpoint:
        resume_checkpoint = Path(args.checkpoint)
    elif args.resume:
        # Auto-find latest checkpoint in experiment folder
        resume_checkpoint = Trainer.find_latest_checkpoint(checkpoint_dir)
        if resume_checkpoint:
            print(f"Found checkpoint to resume: {resume_checkpoint}")
        else:
            print("No checkpoint found, starting fresh training")
    
    # Create trainer first (needed to load checkpoint)
    print("\nInitializing trainer...")
    
    # Temporary logger (will be replaced after loading checkpoint if resuming)
    logger = None
    
    trainer = Trainer(
        world_model=world_model,
        actor_critic=actor_critic,
        env=env,
        config=dict(config),
        logger=logger,
        device=device,
    )
    
    # Load checkpoint if resuming
    if resume_checkpoint and resume_checkpoint.exists():
        wandb_run_id = trainer.load_checkpoint(str(resume_checkpoint))
    
    # Create WandB logger (after loading checkpoint to get run ID)
    # Only on rank 0
    if config.logging.use_wandb and rank == 0:
        print("\nInitializing WandB logger...")
        logger = WandbLogger(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            name=experiment_name,
            config=dict(config),
            log_dir=str(log_dir),
            enabled=True,
            resume_run_id=wandb_run_id,  # Resume existing run if available
        )
        trainer.logger = logger
    
    # Save configuration (rank 0)
    if rank == 0:
        save_config(config, log_dir / "config.yaml")
    
    # Train
    print("\nStarting training...")
    print(f"Total steps: {config.training.total_steps:,}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Sequence length: {config.training.batch_length}")
    print()
    
    try:
        final_metrics = trainer.train()
        
        print("\nTraining complete!")
        print("Final evaluation metrics:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.4f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving emergency checkpoint...")
        trainer._save_checkpoint()
        print(f"Checkpoint saved. Resume with: --resume or --checkpoint {trainer.checkpoint_dir / 'checkpoint_latest.pt'}")
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        print("Saving emergency checkpoint...")
        try:
            trainer._save_checkpoint()
            print(f"Checkpoint saved. Resume with: --resume")
        except:
            print("Failed to save checkpoint")
        raise
    
    finally:
        # Cleanup
        env.close()
        if logger:
            logger.finish()
        if dist.is_initialized():
            dist.destroy_process_group()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

