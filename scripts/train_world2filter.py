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


class World2FilterTrainer:
    """
    Trainer for World2Filter model.
    
    Extends the base trainer with:
    - Mask generation during data collection
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
        self.world_model = world_model.to(device)
        self.actor_critic = actor_critic.to(device)
        self.mask_processor = mask_processor
        self.env = env
        self.config = config
        self.logger = logger
        self.device = device
        
        # Training config
        train_cfg = config['training']
        self.total_steps = train_cfg['total_steps']
        self.batch_size = train_cfg['batch_size']
        self.batch_length = train_cfg['batch_length']
        self.prefill_steps = train_cfg['prefill_steps']
        self.train_every = train_cfg['train_every']
        self.train_steps = train_cfg['train_steps']
        self.log_every = train_cfg['log_every']
        self.eval_every = train_cfg['eval_every']
        self.checkpoint_every = train_cfg['checkpoint_every']
        
        # Imagination config
        img_cfg = config['imagination']
        self.imagination_horizon = img_cfg['horizon']
        self.discount = img_cfg['discount']
        self.lambda_ = img_cfg['lambda_']
        
        # Initialize replay buffer
        replay_cfg = config['replay']
        self.replay_buffer = ReplayBuffer(
            capacity=replay_cfg['capacity'],
            min_length=replay_cfg['min_length'],
            prioritize_ends=replay_cfg['prioritize_ends'],
        )
        
        # Optimizers
        wm_cfg = config['world_model']['optimizer']
        self.world_model_optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=wm_cfg['lr'],
            eps=wm_cfg['eps'],
        )
        
        actor_cfg = config['actor']['optimizer']
        self.actor_optimizer = torch.optim.Adam(
            self.actor_critic.actor.parameters(),
            lr=actor_cfg['lr'],
        )
        
        critic_cfg = config['critic']['optimizer']
        self.critic_optimizer = torch.optim.Adam(
            self.actor_critic.critic.parameters(),
            lr=critic_cfg['lr'],
        )
        
        # Losses
        loss_cfg = config['world_model']['loss']
        self.fg_bg_loss = FGBGReconstructionLoss()
        self.kl_free = loss_cfg['kl_free']
        self.kl_balance = loss_cfg['kl_balance']
        
        self.actor_critic_loss = ActorCriticLoss(
            discount=self.discount,
            lambda_=self.lambda_,
            entropy_scale=config['actor']['entropy_scale'],
        )
        
        # Gradient clipping
        self.wm_clip = wm_cfg['clip']
        self.actor_clip = actor_cfg['clip']
        self.critic_clip = critic_cfg['clip']
        
        # Tracking
        self.global_step = 0
        self.episodes_collected = 0
        
        # Online buffer
        self.online_buffer = OnlineBuffer(
            obs_shape=env.observation_space['image'],
            action_dim=env.action_dim,
        )
        
        # Online mask processor
        self.online_mask_processor = OnlineMaskProcessor(mask_processor)
        
        # Agent state
        self._current_state = None
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self):
        """Main training loop."""
        from tqdm import tqdm
        
        # Prefill
        print(f"Prefilling replay buffer with {self.prefill_steps} steps...")
        self._prefill()
        
        # Main loop
        print("Starting World2Filter training...")
        pbar = tqdm(total=self.total_steps, desc="Training")
        
        obs, info = self.env.reset()
        self._current_state = None
        is_first = True
        self.online_mask_processor.reset()
        
        metrics_accumulator = {}
        
        while self.global_step < self.total_steps:
            # Get action
            with torch.no_grad():
                action, state = self._get_action(obs, is_first)
                self._current_state = state
            
            # Get mask for current observation
            fg_mask, bg_mask = self.online_mask_processor.add_observation(obs)
            
            # Environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition with masks
            self.online_buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                is_first=is_first,
                is_terminal=terminated,
            )
            
            obs = next_obs
            is_first = False
            self.global_step += 1
            
            # Handle episode end
            if done:
                episode = self.online_buffer.get_episode()
                
                # Add masks to episode
                fg_masks, bg_masks = self.online_mask_processor.get_episode_masks()
                episode.fg_mask = fg_masks
                episode.bg_mask = bg_masks
                
                self.replay_buffer.add(episode)
                self.episodes_collected += 1
                self.online_buffer.reset()
                self.online_mask_processor.reset()
                
                # Log
                if self.logger:
                    self.logger.log_scalar(
                        "episode/return",
                        info.get('episode_reward', 0),
                        self.global_step
                    )
                
                obs, info = self.env.reset()
                self._current_state = None
                is_first = True
            
            # Training
            if self.global_step % self.train_every == 0:
                for _ in range(self.train_steps):
                    metrics = self._train_step()
                    self._accumulate_metrics(metrics_accumulator, metrics)
            
            # Logging
            if self.global_step % self.log_every == 0 and metrics_accumulator:
                avg_metrics = self._average_metrics(metrics_accumulator)
                if self.logger:
                    self.logger.log_scalars(avg_metrics, self.global_step)
                metrics_accumulator = {}
            
            # Checkpoint
            if self.global_step % self.checkpoint_every == 0:
                self._save_checkpoint()
            
            pbar.update(1)
        
        pbar.close()
        return self._evaluate()
    
    def _prefill(self):
        """Prefill replay buffer with random actions."""
        obs, info = self.env.reset()
        self.online_buffer.reset()
        self.online_mask_processor.reset()
        steps = 0
        is_first = True
        
        while steps < self.prefill_steps:
            action = np.random.uniform(-1, 1, self.env.action_dim)
            
            # Get mask
            self.online_mask_processor.add_observation(obs)
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            self.online_buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                is_first=is_first,
                is_terminal=terminated,
            )
            
            obs = next_obs
            is_first = False
            steps += 1
            
            if done:
                episode = self.online_buffer.get_episode()
                fg_masks, bg_masks = self.online_mask_processor.get_episode_masks()
                episode.fg_mask = fg_masks
                episode.bg_mask = bg_masks
                
                self.replay_buffer.add(episode)
                self.online_buffer.reset()
                self.online_mask_processor.reset()
                
                obs, info = self.env.reset()
                is_first = True
    
    def _get_action(self, obs: np.ndarray, is_first: bool):
        """Get action from policy."""
        obs_tensor = torch.tensor(
            obs[np.newaxis],
            dtype=torch.float32,
            device=self.device
        )
        is_first_tensor = torch.tensor([is_first], dtype=torch.bool, device=self.device)
        action_tensor = torch.zeros(1, self.env.action_dim, device=self.device)
        
        state, features = self.world_model.obs_step(
            obs_tensor, action_tensor, is_first_tensor, self._current_state
        )
        
        action, _ = self.actor_critic.act(features)
        
        return action[0].cpu().numpy(), state
    
    def _train_step(self):
        """Single training step."""
        batch = self.replay_buffer.sample(
            self.batch_size,
            self.batch_length,
            device=self.device,
        )
        
        # Train world model
        wm_metrics = self._train_world_model(batch)
        
        # Train actor-critic
        ac_metrics = self._train_actor_critic(batch)
        
        self.actor_critic.update_slow_target()
        
        return {**wm_metrics, **ac_metrics}
    
    def _train_world_model(self, batch):
        """Train world model with FG/BG loss."""
        self.world_model_optimizer.zero_grad()
        
        output = self.world_model(
            batch['obs'],
            batch['action'],
            batch['is_first'],
        )
        
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
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.wm_clip)
        self.world_model_optimizer.step()
        
        return {
            'wm/total_loss': total_loss.item(),
            'wm/fg_loss': fg_loss.item(),
            'wm/bg_loss': bg_loss.item(),
            'wm/kl_loss': kl_loss.item(),
            'wm/kl_value': kl_value.item(),
            'wm/reward_loss': reward_loss.item(),
            'wm/continue_loss': continue_loss.item(),
        }
    
    def _train_actor_critic(self, batch):
        """Train actor-critic via imagination."""
        with torch.no_grad():
            output = self.world_model(
                batch['obs'], batch['action'], batch['is_first']
            )
            
            batch_size = batch['obs'].shape[0]
            t_idx = torch.randint(0, batch['obs'].shape[1], (batch_size,))
            
            from src.models.dreamer_v3.rssm import RSSMState
            start_state = RSSMState(
                deter=output.posterior.deter[torch.arange(batch_size), t_idx],
                stoch=output.posterior.stoch[torch.arange(batch_size), t_idx],
                logits=output.posterior.logits[torch.arange(batch_size), t_idx],
            )
            start_features = self.world_model.rssm.get_features(start_state)
        
        features, actions, rewards, continues = self.world_model.imagine_with_policy(
            start_features, start_state, self.actor_critic.actor, self.imagination_horizon
        )
        
        with torch.no_grad():
            values = self.actor_critic.value(features)
            bootstrap = self.actor_critic.target_value(features[:, -1])
        
        returns = self.actor_critic_loss.compute_returns(rewards, values, continues, bootstrap)
        returns = self.actor_critic_loss.normalize_returns(returns)
        advantages = returns - values
        
        log_probs = self.actor_critic.actor.log_prob(features, actions)
        entropy = self.actor_critic.actor.entropy(features)
        
        # Actor
        self.actor_optimizer.zero_grad()
        actor_loss, actor_metrics = self.actor_critic_loss.actor_loss(log_probs, entropy, advantages)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), self.actor_clip)
        self.actor_optimizer.step()
        
        # Critic
        features_det = features.detach()
        values_pred = self.actor_critic.value(features_det)
        
        self.critic_optimizer.zero_grad()
        critic_loss, critic_metrics = self.actor_critic_loss.critic_loss(values_pred, returns.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), self.critic_clip)
        self.critic_optimizer.step()
        
        return {f'ac/{k}': v.item() if hasattr(v, 'item') else v for k, v in {**actor_metrics, **critic_metrics}.items()}
    
    def _evaluate(self):
        """Evaluate current policy."""
        returns = []
        for _ in range(10):
            obs, _ = self.env.reset()
            state = None
            is_first = True
            ep_return = 0
            
            while True:
                with torch.no_grad():
                    action, state = self._get_action(obs, is_first)
                obs, reward, term, trunc, _ = self.env.step(action)
                is_first = False
                ep_return += reward
                if term or trunc:
                    break
            
            returns.append(ep_return)
        
        return {'return_mean': np.mean(returns), 'return_std': np.std(returns)}
    
    def _save_checkpoint(self):
        """Save checkpoint."""
        torch.save({
            'global_step': self.global_step,
            'world_model': self.world_model.state_dict(),
            'actor_critic': self.actor_critic.state_dict(),
        }, self.checkpoint_dir / f'checkpoint_{self.global_step}.pt')
    
    def _accumulate_metrics(self, acc, metrics):
        for k, v in metrics.items():
            if k not in acc:
                acc[k] = []
            acc[k].append(v)
    
    def _average_metrics(self, acc):
        return {k: np.mean(v) for k, v in acc.items()}


def main():
    """Main training function."""
    args = parse_args()
    
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
    
    device = config.device if torch.cuda.is_available() else "cpu"
    set_seed(config.seed)
    
    print("\nConfiguration:")
    print_config(config)
    
    # Create environment
    print("\nCreating environment...")
    env = make_distracting_cs_env(
        domain=config.environment.domain,
        task=config.environment.task,
        image_size=config.environment.obs.image_size,
        action_repeat=config.environment.obs.action_repeat,
        background=config.environment.distractions.background,
        seed=config.seed,
    )
    
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

