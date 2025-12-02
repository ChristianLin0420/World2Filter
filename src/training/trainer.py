"""
Trainer for DreamerV3.

Implements the full training loop including:
- World model training from replay buffer
- Actor-critic training via imagination
- Environment interaction and data collection
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from tqdm import tqdm

from src.models.dreamer_v3.world_model import WorldModel
from src.models.dreamer_v3.actor_critic import ActorCritic
from src.models.dreamer_v3.rssm import RSSMState
from src.data.replay_buffer import ReplayBuffer, Episode, OnlineBuffer
from src.training.losses import WorldModelLoss, ActorCriticLoss
from src.utils.logging import WandbLogger


class Trainer:
    """
    DreamerV3 Trainer.
    
    Handles:
    - World model training
    - Actor-critic training via imagination
    - Environment interaction
    - Logging and checkpointing
    """
    
    def __init__(
        self,
        world_model: WorldModel,
        actor_critic: ActorCritic,
        env,
        config: Dict,
        logger: Optional[WandbLogger] = None,
        device: str = "cuda",
    ):
        """
        Args:
            world_model: World model
            actor_critic: Actor-critic module
            env: Training environment
            config: Training configuration
            logger: WandB logger
            device: Device to train on
        """
        self.world_model = world_model.to(device)
        self.actor_critic = actor_critic.to(device)
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
        self.eval_every = train_cfg['eval_every']
        self.eval_episodes = train_cfg['eval_episodes']
        self.checkpoint_every = train_cfg['checkpoint_every']
        self.log_every = train_cfg['log_every']
        
        # Logging config
        log_cfg = config.get('logging', {})
        self.image_log_freq = log_cfg.get('image_log_freq', 5000)
        self.video_log_freq = log_cfg.get('video_log_freq', 10000)
        
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
        
        # Initialize optimizers
        wm_cfg = config['world_model']['optimizer']
        self.world_model_optimizer = Adam(
            self.world_model.parameters(),
            lr=wm_cfg['lr'],
            eps=wm_cfg['eps'],
            weight_decay=wm_cfg['weight_decay'],
        )
        
        actor_cfg = config['actor']['optimizer']
        self.actor_optimizer = Adam(
            self.actor_critic.actor.parameters(),
            lr=actor_cfg['lr'],
            eps=actor_cfg['eps'],
        )
        
        critic_cfg = config['critic']['optimizer']
        self.critic_optimizer = Adam(
            self.actor_critic.critic.parameters(),
            lr=critic_cfg['lr'],
            eps=critic_cfg['eps'],
        )
        
        # Initialize losses
        loss_cfg = config['world_model']['loss']
        self.world_model_loss = WorldModelLoss(
            kl_free=loss_cfg['kl_free'],
            kl_forward=loss_cfg['kl_forward'],
            kl_balance=loss_cfg['kl_balance'],
            kl_weight=loss_cfg['kl_weight'],
            image_weight=loss_cfg['image_weight'],
            reward_weight=loss_cfg['reward_weight'],
            continue_weight=loss_cfg['continue_weight'],
        )
        
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
        self.train_steps_done = 0
        
        # Online buffer for current episode
        self.online_buffer = OnlineBuffer(
            obs_shape=env.observation_space['image'],
            action_dim=env.action_dim,
        )
        
        # Current RSSM state for acting
        self._current_state = None
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self) -> Dict[str, float]:
        """
        Main training loop.
        
        Returns:
            Final metrics
        """
        # Prefill replay buffer if empty (fresh start or resume without buffer)
        if len(self.replay_buffer) == 0 or self.replay_buffer.num_steps < self.prefill_steps:
            print(f"Prefilling replay buffer with {self.prefill_steps} steps...")
            self._prefill()
        
        if self.global_step > 0:
            print(f"Resuming training from step {self.global_step}...")
        
        # Main training loop
        print("Starting training...")
        pbar = tqdm(total=self.total_steps, initial=self.global_step, desc="Training")
        
        obs, info = self.env.reset()
        self._current_state = None
        is_first = True
        
        metrics_accumulator = {}
        
        while self.global_step < self.total_steps:
            # Get action from policy
            with torch.no_grad():
                action, state = self._get_action(obs, is_first)
                self._current_state = state
            
            # Environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
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
                # Add episode to replay buffer
                episode = self.online_buffer.get_episode()
                self.replay_buffer.add(episode)
                self.episodes_collected += 1
                self.online_buffer.reset()
                
                # Log episode stats
                if self.logger:
                    self.logger.log_scalar(
                        "episode/return", 
                        info.get('episode_reward', 0),
                        self.global_step
                    )
                    self.logger.log_scalar(
                        "episode/length",
                        info.get('step_count', 0),
                        self.global_step
                    )
                
                # Reset environment
                obs, info = self.env.reset()
                self._current_state = None
                is_first = True
            
            # Training step
            if self.global_step % self.train_every == 0:
                for _ in range(self.train_steps):
                    metrics = self._train_step()
                    self._accumulate_metrics(metrics_accumulator, metrics)
                    self.train_steps_done += 1
            
            # Logging
            if self.global_step % self.log_every == 0 and metrics_accumulator:
                avg_metrics = self._average_metrics(metrics_accumulator)
                if self.logger:
                    for key, value in avg_metrics.items():
                        self.logger.log_scalar(key, value, self.global_step)
                metrics_accumulator = {}
            
            # Image logging (reconstructions)
            if self.global_step % self.image_log_freq == 0 and self.global_step > 0:
                self._log_reconstructions()
            
            # Video logging (dream rollouts)
            if self.global_step % self.video_log_freq == 0 and self.global_step > 0:
                self._log_dream_video()
            
            # Evaluation
            if self.global_step % self.eval_every == 0:
                eval_metrics = self._evaluate()
                if self.logger:
                    for key, value in eval_metrics.items():
                        self.logger.log_scalar(f"eval/{key}", value, self.global_step)
            
            # Checkpointing
            if self.global_step % self.checkpoint_every == 0:
                self._save_checkpoint()
            
            pbar.update(1)
        
        pbar.close()
        
        # Save final checkpoint
        self._save_checkpoint(is_final=True)
        
        # Final evaluation
        final_metrics = self._evaluate()
        
        return final_metrics
    
    def _prefill(self):
        """Prefill replay buffer with random actions."""
        obs, info = self.env.reset()
        self.online_buffer.reset()
        steps = 0
        is_first = True
        
        while steps < self.prefill_steps:
            # Random action
            action = np.random.uniform(-1, 1, self.env.action_dim)
            
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
                self.replay_buffer.add(episode)
                self.online_buffer.reset()
                obs, info = self.env.reset()
                is_first = True
    
    def _get_action(
        self,
        obs: np.ndarray,
        is_first: bool,
    ) -> Tuple[np.ndarray, RSSMState]:
        """Get action from policy."""
        # Convert observation to tensor
        obs_tensor = torch.tensor(
            obs[np.newaxis], 
            dtype=torch.float32,
            device=self.device
        )
        is_first_tensor = torch.tensor(
            [is_first], 
            dtype=torch.bool,
            device=self.device
        )
        action_tensor = torch.zeros(
            1, self.env.action_dim,
            dtype=torch.float32,
            device=self.device
        )
        
        # Get RSSM state
        state, features = self.world_model.obs_step(
            obs_tensor,
            action_tensor,
            is_first_tensor,
            self._current_state,
        )
        
        # Get action from policy
        action, _ = self.actor_critic.act(features)
        
        return action[0].cpu().numpy(), state
    
    def _train_step(self) -> Dict[str, float]:
        """Single training step."""
        # Sample from replay buffer
        batch = self.replay_buffer.sample(
            self.batch_size,
            self.batch_length,
            device=self.device,
        )
        
        # Train world model
        wm_metrics = self._train_world_model(batch)
        
        # Train actor-critic via imagination
        ac_metrics = self._train_actor_critic(batch)
        
        # Update slow target
        self.actor_critic.update_slow_target()
        
        return {**wm_metrics, **ac_metrics}
    
    def _train_world_model(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """Train world model on batch."""
        self.world_model_optimizer.zero_grad()
        
        # Forward pass through world model
        output = self.world_model(
            batch['obs'],
            batch['action'],
            batch['is_first'],
        )
        
        # Compute loss
        loss, metrics = self.world_model_loss(
            prior=output.prior,
            posterior=output.posterior,
            image_pred=output.image_pred,
            image_target=batch['obs'],
            reward_logits=output.reward_logits,
            reward_target=batch['reward'],
            continue_logits=output.continue_logits,
            continue_target=batch['cont'],
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(
            self.world_model.parameters(),
            self.wm_clip,
        )
        
        self.world_model_optimizer.step()
        
        metrics['wm_grad_norm'] = grad_norm.item()
        
        return {f"wm/{k}": v.item() if isinstance(v, Tensor) else v for k, v in metrics.items()}
    
    def _train_actor_critic(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """Train actor-critic via imagination."""
        # Get initial state for imagination
        with torch.no_grad():
            output = self.world_model(
                batch['obs'],
                batch['action'],
                batch['is_first'],
            )
            # Use a random timestep from each sequence as start
            batch_size = batch['obs'].shape[0]
            t_idx = torch.randint(0, batch['obs'].shape[1], (batch_size,))
            
            start_state = RSSMState(
                deter=output.posterior.deter[torch.arange(batch_size), t_idx],
                stoch=output.posterior.stoch[torch.arange(batch_size), t_idx],
                logits=output.posterior.logits[torch.arange(batch_size), t_idx],
            )
            start_features = self.world_model.rssm.get_features(start_state)
        
        # Imagine trajectories
        features, actions, rewards, continues = self.world_model.imagine_with_policy(
            start_features,
            start_state,
            self.actor_critic.actor,
            self.imagination_horizon,
        )
        
        # Compute values
        with torch.no_grad():
            values = self.actor_critic.value(features)
            bootstrap = self.actor_critic.target_value(features[:, -1])
        
        # Compute returns
        returns = self.actor_critic_loss.compute_returns(
            rewards, values, continues, bootstrap
        )
        
        # Normalize returns
        returns = self.actor_critic_loss.normalize_returns(returns)
        
        # Compute advantages
        advantages = returns - values
        
        # Get log probs and entropy for actor loss
        _, actor_info = self.actor_critic.actor(features, deterministic=False)
        log_probs = self.actor_critic.actor.log_prob(features, actions)
        entropy = self.actor_critic.actor.entropy(features)
        
        # Train actor
        self.actor_optimizer.zero_grad()
        actor_loss, actor_metrics = self.actor_critic_loss.actor_loss(
            log_probs, entropy, advantages
        )
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(
            self.actor_critic.actor.parameters(),
            self.actor_clip,
        )
        self.actor_optimizer.step()
        
        # Train critic (with fresh forward pass)
        features_detached = features.detach()
        values_pred = self.actor_critic.value(features_detached)
        
        self.critic_optimizer.zero_grad()
        critic_loss, critic_metrics = self.actor_critic_loss.critic_loss(
            values_pred, returns.detach()
        )
        critic_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.actor_critic.critic.parameters(),
            self.critic_clip,
        )
        self.critic_optimizer.step()
        
        metrics = {**actor_metrics, **critic_metrics}
        metrics['actor_grad_norm'] = actor_grad_norm.item()
        metrics['critic_grad_norm'] = critic_grad_norm.item()
        
        return {f"ac/{k}": v.item() if isinstance(v, Tensor) else v for k, v in metrics.items()}
    
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate current policy."""
        returns = []
        lengths = []
        
        for _ in range(self.eval_episodes):
            obs, info = self.env.reset()
            state = None
            is_first = True
            episode_return = 0.0
            episode_length = 0
            
            while True:
                with torch.no_grad():
                    action, state = self._get_action(obs, is_first)
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                is_first = False
                
                episode_return += reward
                episode_length += 1
                
                if done:
                    break
            
            returns.append(episode_return)
            lengths.append(episode_length)
        
        return {
            'return_mean': np.mean(returns),
            'return_std': np.std(returns),
            'length_mean': np.mean(lengths),
        }
    
    def _save_checkpoint(self, is_final: bool = False):
        """Save training checkpoint."""
        # Get WandB run ID if available
        wandb_run_id = None
        if self.logger and hasattr(self.logger, 'run_id'):
            wandb_run_id = self.logger.run_id
        
        checkpoint = {
            'global_step': self.global_step,
            'episodes_collected': self.episodes_collected,
            'train_steps_done': self.train_steps_done,
            'world_model': self.world_model.state_dict(),
            'actor_critic': self.actor_critic.state_dict(),
            'world_model_optimizer': self.world_model_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'wandb_run_id': wandb_run_id,
            'config': self.config,
        }
        
        path = self.checkpoint_dir / f"checkpoint_{self.global_step}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}", flush=True)
        
        # Create/update 'latest' symlink for easy resuming
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        latest_path.symlink_to(path.name)
        
        # Keep only last 3 checkpoints (plus latest symlink)
        checkpoints = sorted([
            c for c in self.checkpoint_dir.glob("checkpoint_*.pt") 
            if not c.is_symlink() and c.name != "checkpoint_latest.pt"
        ])
        for ckpt in checkpoints[:-3]:
            ckpt.unlink()
    
    def load_checkpoint(self, path: str) -> Optional[str]:
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            WandB run ID if saved in checkpoint, None otherwise
        """
        print(f"Loading checkpoint from: {path}", flush=True)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.global_step = checkpoint['global_step']
        self.episodes_collected = checkpoint['episodes_collected']
        self.train_steps_done = checkpoint['train_steps_done']
        
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.world_model_optimizer.load_state_dict(checkpoint['world_model_optimizer'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        print(f"Resumed from step {self.global_step}, episodes: {self.episodes_collected}", flush=True)
        
        return checkpoint.get('wandb_run_id')
    
    @staticmethod
    def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
        """Find the latest checkpoint in a directory."""
        latest = checkpoint_dir / "checkpoint_latest.pt"
        if latest.exists():
            # Resolve symlink to actual file
            return latest.resolve() if latest.is_symlink() else latest
        
        # Fallback: find checkpoint with highest step number
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
        
        # Extract step numbers and find max
        def get_step(p):
            try:
                return int(p.stem.split('_')[1])
            except:
                return -1
        
        return max(checkpoints, key=get_step)
    
    def _accumulate_metrics(
        self,
        accumulator: Dict[str, list],
        metrics: Dict[str, float],
    ):
        """Accumulate metrics for averaging."""
        for key, value in metrics.items():
            if key not in accumulator:
                accumulator[key] = []
            accumulator[key].append(value)
    
    def _average_metrics(self, accumulator: Dict[str, list]) -> Dict[str, float]:
        """Average accumulated metrics."""
        return {key: np.mean(values) for key, values in accumulator.items()}
    
    def _log_reconstructions(self):
        """Log image reconstructions to WandB."""
        # Need at least batch_length steps to sample
        min_steps = self.batch_length
        if not self.logger or self.replay_buffer.num_steps < min_steps:
            return
        
        try:
            # Sample 1 sequence for visualization
            batch = self.replay_buffer.sample(1, self.batch_length)
            
            with torch.no_grad():
                # Move to device
                obs = batch['obs'].to(self.device)
                action = batch['action'].to(self.device)
                is_first = batch['is_first'].to(self.device)
                
                # Get reconstructions
                output = self.world_model(obs, action, is_first)
                recon = self.world_model.decode(output.features)
                
                # Log single pair: original vs reconstructed (C, H, W)
                original = obs[0, 0]  # First sample, first timestep
                reconstructed = recon[0, 0]
                
                # Log as two separate images
                self.logger.log_image(
                    name="train/original",
                    image=original,
                    step=self.global_step,
                )
                self.logger.log_image(
                    name="train/reconstructed", 
                    image=reconstructed,
                    step=self.global_step,
                )
        except Exception as e:
            print(f"Warning: Failed to log reconstructions: {e}", flush=True)
    
    def _log_dream_video(self):
        """Log observation video and reconstructed video to WandB."""
        # Need at least batch_length * num_samples steps to sample
        min_steps = self.batch_length * 4  # 4 samples for video
        if not self.logger or self.replay_buffer.num_steps < min_steps:
            return
        
        try:
            # Sample a batch
            batch = self.replay_buffer.sample(2, min(32, self.batch_length))
            
            with torch.no_grad():
                # Move to device
                obs = batch['obs'].to(self.device)
                action = batch['action'].to(self.device)
                is_first = batch['is_first'].to(self.device)
                
                # Get reconstructions
                output = self.world_model(obs, action, is_first)
                recon = self.world_model.decode(output.features)
                
                # Log original observation video (first sample)
                self.logger.log_video(
                    name="train/observation_video",
                    video=obs[:1],  # (1, T, C, H, W)
                    step=self.global_step,
                    fps=10,
                )
                
                # Log reconstructed video (first sample)
                self.logger.log_video(
                    name="train/reconstruction_video",
                    video=recon[:1],  # (1, T, C, H, W)
                    step=self.global_step,
                    fps=10,
                )
        except Exception as e:
            print(f"Warning: Failed to log video: {e}", flush=True)

