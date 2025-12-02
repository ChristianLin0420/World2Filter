#!/usr/bin/env python3
"""
Training script for World2Filter.

This script trains the World2Filter model (DreamerV3 with SAM3 segmentation)
on the Distracting Control Suite environment.

Usage:
    python scripts/train_world2filter.py --config configs/default.yaml
    python scripts/train_world2filter.py segmentation.enabled=True
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
from src.envs.distracting_cs import make_distracting_cs_env
from src.envs.parallel import ParallelEnv
from src.models.world2filter.fg_bg_world_model import FGBGWorldModel
from src.models.dreamer_v3.actor_critic import ActorCritic
from src.models.segmentation.mask_processor import MaskProcessor, OnlineMaskProcessor
from src.training.losses import WorldModelLoss, ActorCriticLoss, FGBGReconstructionLoss
from src.data.replay_buffer import ReplayBuffer, Episode, OnlineBuffer
from src.agents.world2filter_agent import World2FilterAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train World2Filter")
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
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use",
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
        "--no-sam3",
        action="store_true",
        help="Disable SAM3 (use fallback segmentation)",
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
    
    args, remaining = parser.parse_known_args()
    
    # Parse remaining args as config overrides
    overrides = {}
    for arg in remaining:
        if "=" in arg:
            key, value = arg.split("=", 1)
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


from src.training.trainer import Trainer

class World2FilterTrainer(Trainer):
    """
    Trainer for World2Filter model.
    
    Extends the base trainer with:
    - Mask generation during data collection (via hooks)
    - FG/BG reconstruction loss
    - Segmentation-aware logging
    """
    
    def __init__(
        self,
        world_model: FGBGWorldModel,
        actor_critic: ActorCritic,
        mask_processor: MaskProcessor,
        env,
        config: dict,
        logger: WandbLogger = None,
        device: str = "cuda",
    ):
        super().__init__(world_model, actor_critic, env, config, logger, device)
        self.mask_processor = mask_processor
        
        # Losses
        self.fg_bg_loss = FGBGReconstructionLoss()
        loss_cfg = config['world_model']['loss']
        self.kl_free = loss_cfg['kl_free']
        self.kl_balance = loss_cfg['kl_balance']
        
        # Online mask processors for each environment
        self.online_mask_processors = [
            OnlineMaskProcessor(mask_processor) for _ in range(self.num_envs)
        ]
    
    def _on_step(self, env_idx, obs, action, reward, done, info):
        """Update mask processor with new observation."""
        # obs is (C, H, W)
        self.online_mask_processors[env_idx].add_observation(obs)
        
    def _on_episode_end(self, env_idx, episode, info):
        """Attach masks to episode."""
        fg_masks, bg_masks = self.online_mask_processors[env_idx].get_episode_masks()
                episode.fg_mask = fg_masks
                episode.bg_mask = bg_masks
        self.online_mask_processors[env_idx].reset()
        
    def _compute_wm_loss(self, output, batch):
        """Compute World2Filter loss with FG/BG reconstruction."""
        # FG/BG reconstruction loss
        if 'fg_mask' in batch and 'bg_mask' in batch:
            recon_loss, fg_loss, bg_loss, _ = self.fg_bg_loss(
                output.fg_pred,
                output.bg_pred,
                batch['obs'],
                batch['fg_mask'],
                batch['bg_mask'],
            )
        else:
            # Fallback to combined loss if masks not available
            combined_pred = (output.fg_pred + output.bg_pred) / 2
            target = batch['obs'].float()
            if target.max() > 1.0:
                target = target / 255.0 - 0.5
            recon_loss = ((combined_pred - target) ** 2).mean()
            fg_loss = recon_loss
            bg_loss = recon_loss
        
        # KL loss
        kl_loss, kl_value = self.world_model.rssm.kl_divergence(
            output.prior, output.posterior,
            free_bits=self.kl_free,
            balance=self.kl_balance,
        )
        kl_loss = kl_loss.mean()
        kl_value = kl_value.mean()
        
        # Reward loss
        from src.models.dreamer_v3.networks import SymlogDist
        reward_dist = SymlogDist(255)
        reward_loss = -reward_dist.log_prob(output.reward_logits, batch['reward']).mean()
        
        # Continue loss
        import torch.nn.functional as F
        continue_loss = F.binary_cross_entropy_with_logits(
            output.continue_logits.squeeze(-1),
            batch['cont'],
        )
        
        # Total loss
        total_loss = recon_loss + kl_loss + reward_loss + continue_loss
        
        metrics = {
            'wm/total_loss': total_loss.item(),
            'wm/fg_loss': fg_loss.item(),
            'wm/bg_loss': bg_loss.item(),
            'wm/kl_loss': kl_loss.item(),
            'wm/kl_value': kl_value.item(),
            'wm/reward_loss': reward_loss.item(),
            'wm/continue_loss': continue_loss.item(),
        }
    
        return total_loss, metrics


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
    
    # Override segmentation to be enabled
    if 'segmentation' not in config:
        config['segmentation'] = {}
    config['segmentation']['enabled'] = True
    
    # Override from command line
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device
    if args.wandb is True:
        config.logging.use_wandb = True
    if args.no_wandb:
        config.logging.use_wandb = False
    
    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    
    # Set seed
    set_seed(config.seed + rank)
    
    print("\nConfiguration:")
    print_config(config)
    
    # Create environment
    print("\nCreating environment...")
    
    def make_env():
        return make_distracting_cs_env(
            domain=config.environment.domain,
            task=config.environment.task,
            image_size=config.environment.obs.image_size,
            action_repeat=config.environment.obs.action_repeat,
            background=config.environment.distractions.background,
            seed=config.seed,
        )
        
    num_envs = config.environment.get('num_envs', 1)
    if num_envs > 1:
        print(f"Using {num_envs} parallel environments")
        env = ParallelEnv([make_env for _ in range(num_envs)])
    else:
        env = make_env()
    
    action_dim = env.action_dim
    obs_shape = env.observation_space['image']
    
    print(f"Environment: {config.environment.domain}/{config.environment.task}")
    print(f"Observation shape: {obs_shape}, Action dim: {action_dim}")
    
    # Create world model
    print("\nBuilding World2Filter model...")
    world_model = FGBGWorldModel(
        obs_shape=obs_shape,
        action_dim=action_dim,
        deter_size=config.world_model.rssm.deter_size,
        stoch_size=config.world_model.rssm.stoch_size,
        classes=config.world_model.rssm.classes,
        hidden_size=config.world_model.rssm.hidden_size,
    )
    
    # Create actor-critic
    feature_dim = world_model.feature_dim
    
    actor_config = {
        'hidden_dim': config.actor.units,
        'num_layers': config.actor.layers,
        'activation': config.actor.act,
        'norm': config.actor.norm,
        'dist_type': config.actor.dist,
        'min_std': config.actor.min_std,
        'max_std': config.actor.max_std,
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
    }
    
    actor_critic = ActorCritic(
        feature_dim=feature_dim,
        action_dim=action_dim,
        actor_config=actor_config,
        critic_config=critic_config,
        discount=config.imagination.discount,
        lambda_=config.imagination.lambda_,
    )
    
    # Create mask processor
    print("\nInitializing SAM3 mask processor...")
    mask_processor = MaskProcessor(
        cache_dir=config.segmentation.get('cache_dir', './mask_cache'),
        prompt=config.segmentation.get('prompt', 'robot'),
        device=device,
        use_sam3=not args.no_sam3,
    )
    
    # Create logger
    logger = None
    if config.logging.use_wandb:
        experiment_name = f"world2filter_{config.environment.domain}_{config.environment.task}_{config.seed}"
        logger = WandbLogger(
            project=config.logging.wandb_project,
            name=experiment_name,
            config=dict(config),
        )
    
    # Save config
    os.makedirs(config.log_dir, exist_ok=True)
    save_config(config, Path(config.log_dir) / "config.yaml")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = World2FilterTrainer(
        world_model=world_model,
        actor_critic=actor_critic,
        mask_processor=mask_processor,
        env=env,
        config=dict(config),
        logger=logger,
        device=device,
    )
    
    # Train
    print("\nStarting World2Filter training...")
    try:
        final_metrics = trainer.train()
        print("\nTraining complete!")
        print(f"Final metrics: {final_metrics}")
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    finally:
        env.close()
        if logger:
            logger.finish()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

