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


import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class Trainer:
    """
    DreamerV3 Trainer.
    
    Handles:
    - World model training
    - Actor-critic training via imagination
    - Environment interaction
    - Logging and checkpointing
    - Distributed Data Parallel (DDP) training
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
        # Check for distributed setup
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        
        self.world_model = world_model.to(device)
        self.actor_critic = actor_critic.to(device)
        
        # Wrap with DDP if distributed
        if self.is_distributed:
             # Need to ensure buffers are broadcasted if needed, but usually fine.
             # find_unused_parameters might be needed if not all parameters are used in every forward pass
             # (e.g. actor critic might use different parts).
            self.world_model = DDP(self.world_model, device_ids=[int(os.environ.get("LOCAL_RANK", 0))], find_unused_parameters=False)
            self.actor_critic = DDP(self.actor_critic, device_ids=[int(os.environ.get("LOCAL_RANK", 0))], find_unused_parameters=False)
            
        self.env = env
        self.config = config
        self.logger = logger if self.rank == 0 else None
        self.device = device
        
        # Training config
        train_cfg = config['training']
        self.total_steps = train_cfg['total_steps']
        # Scale batch size by world size if it's a global batch size (often config is per GPU or global, assuming global here needs care)
        # Usually config 'batch_size' is per-gpu. If it's global, we divide.
        # Let's assume config is PER-GPU batch size for simplicity, or GLOBAL?
        # Standard practice: config is per-GPU batch size.
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
        # Access underlying module if DDP wrapped
        wm_params = self.world_model.module.parameters() if self.is_distributed else self.world_model.parameters()
        wm_cfg = config['world_model']['optimizer']
        self.world_model_optimizer = Adam(
            wm_params,
            lr=wm_cfg['lr'],
            eps=wm_cfg['eps'],
            weight_decay=wm_cfg['weight_decay'],
        )
        
        ac_module = self.actor_critic.module if self.is_distributed else self.actor_critic
        actor_cfg = config['actor']['optimizer']
        self.actor_optimizer = Adam(
            ac_module.actor.parameters(),
            lr=actor_cfg['lr'],
            eps=actor_cfg['eps'],
        )
        
        critic_cfg = config['critic']['optimizer']
        self.critic_optimizer = Adam(
            ac_module.critic.parameters(),
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
        
        # Online buffer for current episode(s)
        if hasattr(env, 'num_envs'):
            self.num_envs = env.num_envs
        else:
            self.num_envs = 1

        self.online_buffers = [
            OnlineBuffer(
                obs_shape=env.observation_space['image'] if isinstance(env.observation_space, dict) else env.observation_space,
                action_dim=env.action_dim,
            ) for _ in range(self.num_envs)
        ]
        
        # Current RSSM state for acting
        self._current_states = [None] * self.num_envs
        
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
        
        obs, infos = self.env.reset()
        if self.num_envs == 1 and not hasattr(self.env, 'num_envs'):
             obs = np.stack([obs])
             infos = [infos]
        
        self._current_state = None
        is_first = np.ones(self.num_envs, dtype=bool)
        
        metrics_accumulator = {}
        
        while self.global_step < self.total_steps:
            # Get action from policy
            with torch.no_grad():
                action, state = self._get_action(obs, is_first)
                self._current_state = state
            
            # Environment step
            next_obs, reward, terminated, truncated, infos = self.env.step(action)
            
            # Normalize for single env
            if self.num_envs == 1 and not hasattr(self.env, 'num_envs'):
                next_obs = np.stack([next_obs])
                reward = np.stack([reward])
                terminated = np.stack([terminated])
                truncated = np.stack([truncated])
                infos = [infos]
            
            done = terminated | truncated
            
            # Iterate over environments
            for i in range(self.num_envs):
                # Call hook
                self._on_step(i, obs[i], action[i], reward[i], done[i], infos[i])
                
                self.online_buffers[i].add(
                    obs=obs[i],
                    action=action[i],
                    reward=reward[i],
                    is_first=is_first[i],
                    is_terminal=terminated[i],
                )
                
                if done[i]:
                    # Add episode to replay buffer
                    episode = self.online_buffers[i].get_episode()
                    
                    # Call hook
                    self._on_episode_end(i, episode, infos[i])
                    
                    self.replay_buffer.add(episode)
                    self.episodes_collected += 1
                    self.online_buffers[i].reset()
                    
                    # Log episode stats
                    if self.logger:
                        ep_reward = 0
                        ep_len = 0
                        # Try to find episode stats in info
                        if isinstance(infos[i], dict):
                            if 'episode' in infos[i]:
                                ep_reward = infos[i]['episode'].get('r', 0)
                                ep_len = infos[i]['episode'].get('l', 0)
                            elif 'episode_reward' in infos[i]:
                                ep_reward = infos[i]['episode_reward']
                                ep_len = infos[i].get('step_count', 0)
                        
                        self.logger.log_scalar("episode/return", ep_reward, self.global_step)
                        self.logger.log_scalar("episode/length", ep_len, self.global_step)
                    
                    # Manual reset for single env if needed (ParallelEnv auto-resets)
                    if self.num_envs == 1 and not hasattr(self.env, 'num_envs'):
                        next_obs_i, info_i = self.env.reset()
                        next_obs[i] = next_obs_i
            
            obs = next_obs
            is_first = done
            
            # Determine if we should train
            prev_step = self.global_step
            self.global_step += self.num_envs
            
            should_train = (self.global_step // self.train_every) > (prev_step // self.train_every)
            should_log = (self.global_step // self.log_every) > (prev_step // self.log_every)
            should_eval = (self.global_step // self.eval_every) > (prev_step // self.eval_every)
            should_ckpt = (self.global_step // self.checkpoint_every) > (prev_step // self.checkpoint_every)
            
            # Training step
            if should_train:
                for _ in range(self.train_steps):
                    metrics = self._train_step()
                    self._accumulate_metrics(metrics_accumulator, metrics)
                    self.train_steps_done += 1
            
            # Logging
            if should_log and metrics_accumulator:
                avg_metrics = self._average_metrics(metrics_accumulator)
                if self.logger:
                    for key, value in avg_metrics.items():
                        self.logger.log_scalar(key, value, self.global_step)
                metrics_accumulator = {}
            
            # Image logging (reconstructions)
            if self.global_step % self.image_log_freq < self.num_envs and self.global_step > 0:
                self._log_reconstructions()
            
            # Video logging (dream rollouts)
            if self.global_step % self.video_log_freq < self.num_envs and self.global_step > 0:
                self._log_dream_video()
            
            # Evaluation
            if should_eval:
                eval_metrics = self._evaluate()
                if self.logger:
                    for key, value in eval_metrics.items():
                        self.logger.log_scalar(f"eval/{key}", value, self.global_step)
            
            # Checkpointing
            if should_ckpt:
                self._save_checkpoint()
            
            pbar.update(self.num_envs)
        
        pbar.close()
        
        # Save final checkpoint
        self._save_checkpoint(is_final=True)
        
        # Final evaluation
        final_metrics = self._evaluate()
        
        return final_metrics
    
    def _prefill(self):
        """Prefill replay buffer with random actions."""
        obs, infos = self.env.reset()
        if self.num_envs == 1 and not hasattr(self.env, 'num_envs'):
             obs = np.stack([obs])
             infos = [infos]
             
        for buffer in self.online_buffers:
            buffer.reset()
            
        steps = 0
        is_first = np.ones(self.num_envs, dtype=bool)
        
        while steps < self.prefill_steps:
            # Random action (B, action_dim)
            action = np.random.uniform(-1, 1, (self.num_envs, self.env.action_dim))
            
            next_obs, reward, terminated, truncated, infos = self.env.step(action)
            
            if self.num_envs == 1 and not hasattr(self.env, 'num_envs'):
                next_obs = np.stack([next_obs])
                reward = np.stack([reward])
                terminated = np.stack([terminated])
                truncated = np.stack([truncated])
                infos = [infos]
            
            done = terminated | truncated
            
            for i in range(self.num_envs):
                # Call hook
                self._on_step(i, obs[i], action[i], reward[i], done[i], infos[i])
                
                self.online_buffers[i].add(
                    obs=obs[i],
                    action=action[i],
                    reward=reward[i],
                    is_first=is_first[i],
                    is_terminal=terminated[i],
                )
                
                if done[i]:
                    episode = self.online_buffers[i].get_episode()
                    
                    # Call hook
                    self._on_episode_end(i, episode, infos[i])
                    
                    self.replay_buffer.add(episode)
                    self.online_buffers[i].reset()
                    
                    if self.num_envs == 1 and not hasattr(self.env, 'num_envs'):
                        next_obs_i, info_i = self.env.reset()
                        next_obs[i] = next_obs_i
            
            obs = next_obs
            is_first = done
            steps += self.num_envs
    
    def _get_action(
        self,
        obs: np.ndarray,
        is_first: np.ndarray,
    ) -> Tuple[np.ndarray, RSSMState]:
        """Get action from policy."""
        # ... (tensor conversions)
        obs_tensor = torch.tensor(
            obs, 
            dtype=torch.float32,
            device=self.device
        )
        is_first_tensor = torch.tensor(
            is_first, 
            dtype=torch.bool,
            device=self.device
        )
        
        batch_size = obs.shape[0]
        action_tensor = torch.zeros(
            batch_size, self.env.action_dim,
            dtype=torch.float32,
            device=self.device
        )
        
        # Handle DDP wrapping for method calls
        wm_module = self.world_model.module if self.is_distributed else self.world_model
        ac_module = self.actor_critic.module if self.is_distributed else self.actor_critic

        # Get RSSM state
        # ... (rest of logic)
        state, features = wm_module.obs_step(
            obs_tensor,
            action_tensor,
            is_first_tensor,
            self._current_state,
        )
        
        # Get action from policy
        action, _ = ac_module.act(features)
        
        return action.cpu().numpy(), state
    
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
        loss, metrics = self._compute_wm_loss(output, batch)
        
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
        # Access modules
        wm_module = self.world_model.module if self.is_distributed else self.world_model
        ac_module = self.actor_critic.module if self.is_distributed else self.actor_critic
        
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
            start_features = wm_module.rssm.get_features(start_state)
        
        # Imagine trajectories
        features, actions, rewards, continues = wm_module.imagine_with_policy(
            start_features,
            start_state,
            ac_module.actor,
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
        _, actor_info = ac_module.actor(features, deterministic=False)
        log_probs = ac_module.actor.log_prob(features, actions)
        entropy = ac_module.actor.entropy(features)
        
        # Train actor
        self.actor_optimizer.zero_grad()
        actor_loss, actor_metrics = self.actor_critic_loss.actor_loss(
            log_probs, entropy, advantages
        )
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(
            ac_module.actor.parameters(),
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
            ac_module.critic.parameters(),
            self.critic_clip,
        )
        self.critic_optimizer.step()
        
        metrics = {**actor_metrics, **critic_metrics}
        metrics['actor_grad_norm'] = actor_grad_norm.item()
        metrics['critic_grad_norm'] = critic_grad_norm.item()
        
        return {f"ac/{k}": v.item() if isinstance(v, Tensor) else v for k, v in metrics.items()}

    def _on_step(self, env_idx: int, obs: np.ndarray, action: np.ndarray, reward: float, done: bool, info: Dict):
        """Hook for per-step processing."""
        pass

    def _on_episode_end(self, env_idx: int, episode: Episode, info: Dict):
        """Hook for episode end."""
        pass
    
    def _compute_wm_loss(self, output, batch) -> Tuple[Tensor, Dict[str, float]]:
        """Hook to compute world model loss."""
        return self.world_model_loss(
            prior=output.prior,
            posterior=output.posterior,
            image_pred=output.image_pred,
            image_target=batch['obs'],
            reward_logits=output.reward_logits,
            reward_target=batch['reward'],
            continue_logits=output.continue_logits,
            continue_target=batch['cont'],
        )
    
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate current policy."""
        returns = []
        lengths = []
        
        obs, infos = self.env.reset()
        if self.num_envs == 1 and not hasattr(self.env, 'num_envs'):
             obs = np.stack([obs])
        
        self._current_state = None
        is_first = np.ones(self.num_envs, dtype=bool)
        
        current_returns = np.zeros(self.num_envs)
        current_lengths = np.zeros(self.num_envs)
        
        while len(returns) < self.eval_episodes:
            with torch.no_grad():
                action, state = self._get_action(obs, is_first)
                self._current_state = state
            
            next_obs, reward, terminated, truncated, infos = self.env.step(action)
            
            if self.num_envs == 1 and not hasattr(self.env, 'num_envs'):
                next_obs = np.stack([next_obs])
                reward = np.stack([reward])
                terminated = np.stack([terminated])
                truncated = np.stack([truncated])
            
            done = terminated | truncated
            
            current_returns += reward
            current_lengths += 1
            
            for i in range(self.num_envs):
                if done[i]:
                    returns.append(current_returns[i])
                    lengths.append(current_lengths[i])
                    current_returns[i] = 0
                    current_lengths[i] = 0
                    
                    if self.num_envs == 1 and not hasattr(self.env, 'num_envs'):
                        next_obs_i, _ = self.env.reset()
                        next_obs[i] = next_obs_i
            
            obs = next_obs
            is_first = done
            
        return {
            'return_mean': np.mean(returns),
            'return_std': np.std(returns),
            'length_mean': np.mean(lengths),
        }
    
    def _save_checkpoint(self, is_final: bool = False):
        """Save training checkpoint."""
        # Only save on rank 0
        if self.rank != 0:
            return
            
        # Get WandB run ID if available
        wandb_run_id = None
        if self.logger and hasattr(self.logger, 'run_id'):
            wandb_run_id = self.logger.run_id
        
        # Unwrap DDP models for saving
        wm_state = self.world_model.module.state_dict() if self.is_distributed else self.world_model.state_dict()
        ac_state = self.actor_critic.module.state_dict() if self.is_distributed else self.actor_critic.state_dict()
        
        checkpoint = {
            'global_step': self.global_step,
            'episodes_collected': self.episodes_collected,
            'train_steps_done': self.train_steps_done,
            'world_model': wm_state,
            'actor_critic': ac_state,
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
        
        # Unwrap DDP models for loading if needed, or load into wrapped
        # DDP wrapped models have 'module.' prefix, but we saved unwrapped
        if self.is_distributed:
            self.world_model.module.load_state_dict(checkpoint['world_model'])
            self.actor_critic.module.load_state_dict(checkpoint['actor_critic'])
        else:
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
        # Only log on rank 0
        if not self.logger:
             return
             
        # Need at least batch_length steps to sample
        min_steps = self.batch_length
        if self.replay_buffer.num_steps < min_steps:
            return
        
        try:
            # Sample 1 sequence for visualization
            batch = self.replay_buffer.sample(1, self.batch_length)
            
            # Use base model for inference to avoid DDP overhead/sync issues
            wm_module = self.world_model.module if self.is_distributed else self.world_model
            
            with torch.no_grad():
                # Move to device
                obs = batch['obs'].to(self.device)
                action = batch['action'].to(self.device)
                is_first = batch['is_first'].to(self.device)
                
                # Get reconstructions
                output = wm_module(obs, action, is_first)
                recon = wm_module.decode(output.features)
                
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
        # Only log on rank 0
        if not self.logger:
             return

        # Need at least batch_length * num_samples steps to sample
        min_steps = self.batch_length * 4  # 4 samples for video
        if self.replay_buffer.num_steps < min_steps:
            return
        
        try:
            # Sample a batch
            batch = self.replay_buffer.sample(2, min(32, self.batch_length))
            
            wm_module = self.world_model.module if self.is_distributed else self.world_model
            
            with torch.no_grad():
                # Move to device
                obs = batch['obs'].to(self.device)
                action = batch['action'].to(self.device)
                is_first = batch['is_first'].to(self.device)
                
                # Get reconstructions
                output = wm_module(obs, action, is_first)
                recon = wm_module.decode(output.features)
                
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

