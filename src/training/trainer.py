"""
Trainer for DreamerV3.

Implements the full training loop including:
- World model training from replay buffer
- Actor-critic training via imagination
- Environment interaction and data collection
- Distributed Data Parallel (DDP) training support
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from src.models.dreamer_v3.world_model import WorldModel
from src.models.dreamer_v3.actor_critic import ActorCritic
from src.models.dreamer_v3.rssm import RSSMState
from src.data.replay_buffer import ReplayBuffer, Episode, OnlineBuffer
from src.training.losses import WorldModelLoss, ActorCriticLoss
from src.utils.logging import WandbLogger


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_rank() -> int:
    """Get the rank of this process."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get the total number of processes."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


class Trainer:
    """
    DreamerV3 Trainer with Multi-GPU (DDP) support.
    
    Handles:
    - World model training
    - Actor-critic training via imagination
    - Environment interaction
    - Logging and checkpointing
    - Distributed Data Parallel (DDP) training
    
    Design:
    - Only rank 0 logs to WandB
    - Only rank 0 saves checkpoints
    - Each rank has its own environment(s) and replay buffer
    - Gradients are synchronized across GPUs via DDP
    - Extensible via hooks (_on_step, _on_episode_end, _compute_wm_loss)
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
            world_model: World model (will be wrapped with DDP if distributed)
            actor_critic: Actor-critic module (will be wrapped with DDP if distributed)
            env: Training environment (can be single or ParallelEnv)
            config: Training configuration
            logger: WandB logger (only used on rank 0)
            device: Device to train on (e.g., "cuda:0")
        """
        # Distributed setup
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.is_main = is_main_process()
        
        self.device = device
        self.config = config
        self.env = env
        
        # Only main process logs
        self.logger = logger if self.is_main else None
        
        # Move models to device
        self.world_model = world_model.to(device)
        self.actor_critic = actor_critic.to(device)
        
        # Store unwrapped references for methods that need direct access
        self._world_model = world_model
        self._actor_critic = actor_critic
        
        # Wrap with DDP if distributed
        if self.is_distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_model = DDP(
                self.world_model, 
                device_ids=[local_rank],
                find_unused_parameters=False,
            )
            self.actor_critic = DDP(
                self.actor_critic, 
                device_ids=[local_rank],
                find_unused_parameters=False,
            )
        
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
        
        # Initialize replay buffer (each rank has its own)
        replay_cfg = config['replay']
        self.replay_buffer = ReplayBuffer(
            capacity=replay_cfg['capacity'],
            min_length=replay_cfg['min_length'],
            prioritize_ends=replay_cfg['prioritize_ends'],
        )
        
        # Initialize optimizers (use unwrapped model parameters)
        wm_cfg = config['world_model']['optimizer']
        self.world_model_optimizer = Adam(
            self._world_model.parameters(),
            lr=wm_cfg['lr'],
            eps=wm_cfg['eps'],
            weight_decay=wm_cfg['weight_decay'],
        )
        
        actor_cfg = config['actor']['optimizer']
        self.actor_optimizer = Adam(
            self._actor_critic.actor.parameters(),
            lr=actor_cfg['lr'],
            eps=actor_cfg['eps'],
        )
        
        critic_cfg = config['critic']['optimizer']
        self.critic_optimizer = Adam(
            self._actor_critic.critic.parameters(),
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

        obs_space = env.observation_space
        if isinstance(obs_space, dict):
            obs_shape = obs_space['image']
        else:
            obs_shape = obs_space
            
        self.online_buffers = [
            OnlineBuffer(obs_shape=obs_shape, action_dim=env.action_dim)
            for _ in range(self.num_envs)
        ]
        
        # Current RSSM state for acting
        self._current_state = None
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        if self.is_main:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self) -> Dict[str, float]:
        """
        Main training loop.
        
        Returns:
            Final metrics
        """
        # Prefill replay buffer if empty
        if len(self.replay_buffer) == 0 or self.replay_buffer.num_steps < self.prefill_steps:
            if self.is_main:
                print(f"Prefilling replay buffer with {self.prefill_steps} steps...")
            self._prefill()
        
        if self.global_step > 0 and self.is_main:
            print(f"Resuming training from step {self.global_step}...")
        
        # Main training loop
        if self.is_main:
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
                    
                    # Log episode stats (only rank 0)
                    if self.logger:
                        ep_reward = 0
                        ep_len = 0
                        if isinstance(infos[i], dict):
                            if 'episode' in infos[i]:
                                ep_reward = infos[i]['episode'].get('r', 0)
                                ep_len = infos[i]['episode'].get('l', 0)
                            elif 'episode_reward' in infos[i]:
                                ep_reward = infos[i]['episode_reward']
                                ep_len = infos[i].get('step_count', 0)
                        
                        self.logger.log_scalar("episode/return", ep_reward, self.global_step)
                        self.logger.log_scalar("episode/length", ep_len, self.global_step)
                    
                    # Manual reset for single env
                    if self.num_envs == 1 and not hasattr(self.env, 'num_envs'):
                        next_obs_i, info_i = self.env.reset()
                        next_obs[i] = next_obs_i
            
            obs = next_obs
            is_first = done
            
            # Update step count
            prev_step = self.global_step
            self.global_step += self.num_envs
            
            # Training step
            should_train = (self.global_step // self.train_every) > (prev_step // self.train_every)
            if should_train:
                for _ in range(self.train_steps):
                    metrics = self._train_step()
                    self._accumulate_metrics(metrics_accumulator, metrics)
                    self.train_steps_done += 1
            
            # Logging (only rank 0)
            should_log = (self.global_step // self.log_every) > (prev_step // self.log_every)
            if should_log and metrics_accumulator and self.logger:
                avg_metrics = self._average_metrics(metrics_accumulator)
                for key, value in avg_metrics.items():
                    self.logger.log_scalar(key, value, self.global_step)
                metrics_accumulator = {}
            
            # Image logging
            if self.global_step % self.image_log_freq < self.num_envs and self.global_step > 0:
                self._log_reconstructions()
            
            # Video logging
            if self.global_step % self.video_log_freq < self.num_envs and self.global_step > 0:
                self._log_dream_video()
                self._log_imagination_video()  # Log imagined future rollouts
            
            # Evaluation (only rank 0)
            should_eval = (self.global_step // self.eval_every) > (prev_step // self.eval_every)
            if should_eval and self.is_main:
                eval_metrics = self._evaluate()
                if self.logger:
                    for key, value in eval_metrics.items():
                        self.logger.log_scalar(f"eval/{key}", value, self.global_step)
            
            # Checkpointing (only rank 0)
            should_ckpt = (self.global_step // self.checkpoint_every) > (prev_step // self.checkpoint_every)
            if should_ckpt:
                self._save_checkpoint()
            
            if self.is_main:
                pbar.update(self.num_envs)
        
        if self.is_main:
            pbar.close()
        
        # Save final checkpoint
        self._save_checkpoint(is_final=True)
        
        # Final evaluation
        final_metrics = self._evaluate() if self.is_main else {}
        
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
            # Random action
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
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        is_first_tensor = torch.tensor(is_first, dtype=torch.bool, device=self.device)
        
        batch_size = obs.shape[0]
        action_tensor = torch.zeros(
            batch_size, self.env.action_dim,
            dtype=torch.float32,
            device=self.device
        )
        
        # Use unwrapped model for inference
        state, features = self._world_model.obs_step(
            obs_tensor,
            action_tensor,
            is_first_tensor,
            self._current_state,
        )
        
        action, _ = self._actor_critic.act(features)
        
        return action.cpu().numpy(), state
    
    def _train_step(self) -> Dict[str, float]:
        """Single training step."""
        batch = self.replay_buffer.sample(
            self.batch_size,
            self.batch_length,
            device=self.device,
        )
        
        # Train world model
        wm_metrics = self._train_world_model(batch)
        
        # Train actor-critic via imagination
        ac_metrics = self._train_actor_critic(batch)
        
        # Update slow target (use unwrapped model)
        self._actor_critic.update_slow_target()
        
        return {**wm_metrics, **ac_metrics}
    
    def _train_world_model(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """Train world model on batch."""
        self.world_model_optimizer.zero_grad()
        
        # Forward pass (DDP handles gradient sync)
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
            self._world_model.parameters(),
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
            batch_size = batch['obs'].shape[0]
            t_idx = torch.randint(0, batch['obs'].shape[1], (batch_size,))
            
            start_state = RSSMState(
                deter=output.posterior.deter[torch.arange(batch_size), t_idx],
                stoch=output.posterior.stoch[torch.arange(batch_size), t_idx],
                logits=output.posterior.logits[torch.arange(batch_size), t_idx],
            )
            start_features = self._world_model.rssm.get_features(start_state)
        
        # Imagine trajectories (use unwrapped for imagination)
        features, actions, rewards, continues = self._world_model.imagine_with_policy(
            start_features,
            start_state,
            self._actor_critic.actor,
            self.imagination_horizon,
        )
        
        # Compute values
        with torch.no_grad():
            values = self._actor_critic.value(features)
            bootstrap = self._actor_critic.target_value(features[:, -1])
        
        # Compute returns
        returns = self.actor_critic_loss.compute_returns(
            rewards, values, continues, bootstrap
        )
        returns = self.actor_critic_loss.normalize_returns(returns)
        advantages = returns - values
        
        # Get log probs and entropy
        _, actor_info = self._actor_critic.actor(features, deterministic=False)
        log_probs = self._actor_critic.actor.log_prob(features, actions)
        entropy = self._actor_critic.actor.entropy(features)
        
        # Train actor
        self.actor_optimizer.zero_grad()
        actor_loss, actor_metrics = self.actor_critic_loss.actor_loss(
            log_probs, entropy, advantages
        )
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(
            self._actor_critic.actor.parameters(),
            self.actor_clip,
        )
        self.actor_optimizer.step()
        
        # Train critic
        features_detached = features.detach()
        values_pred = self._actor_critic.value(features_detached)
        
        self.critic_optimizer.zero_grad()
        critic_loss, critic_metrics = self.actor_critic_loss.critic_loss(
            values_pred, returns.detach()
        )
        critic_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self._actor_critic.critic.parameters(),
            self.critic_clip,
        )
        self.critic_optimizer.step()
        
        metrics = {**actor_metrics, **critic_metrics}
        metrics['actor_grad_norm'] = actor_grad_norm.item()
        metrics['critic_grad_norm'] = critic_grad_norm.item()
        
        return {f"ac/{k}": v.item() if isinstance(v, Tensor) else v for k, v in metrics.items()}

    # ==================== Hooks for extensibility ====================
    
    def _on_step(self, env_idx: int, obs: np.ndarray, action: np.ndarray, reward: float, done: bool, info: Dict):
        """Hook for per-step processing. Override in subclasses."""
        pass

    def _on_episode_end(self, env_idx: int, episode: Episode, info: Dict):
        """Hook for episode end. Override in subclasses."""
        pass
    
    def _compute_wm_loss(self, output, batch) -> Tuple[Tensor, Dict[str, float]]:
        """Hook to compute world model loss. Override in subclasses for custom losses."""
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
    
    # ==================== Evaluation ====================
    
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
    
    # ==================== Checkpointing ====================
    
    def _save_checkpoint(self, is_final: bool = False):
        """
        Save training checkpoint with atomic writes and DDP synchronization.
        
        Only rank 0 saves, but all ranks wait at a barrier to prevent
        NCCL timeouts during checkpoint saving.
        """
        # Non-main ranks wait at barrier for rank 0 to finish saving
        if not self.is_main:
            if self.is_distributed:
                dist.barrier()
            return
        
        wandb_run_id = None
        if self.logger and hasattr(self.logger, 'run_id'):
            wandb_run_id = self.logger.run_id
        
        checkpoint = {
            'global_step': self.global_step,
            'episodes_collected': self.episodes_collected,
            'train_steps_done': self.train_steps_done,
            'world_model': self._world_model.state_dict(),
            'actor_critic': self._actor_critic.state_dict(),
            'world_model_optimizer': self.world_model_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'wandb_run_id': wandb_run_id,
            'config': self.config,
        }
        
        path = self.checkpoint_dir / f"checkpoint_{self.global_step}.pt"
        temp_path = path.with_suffix('.pt.tmp')
        
        try:
            # Atomic write: save to temp file first, then rename
            torch.save(checkpoint, temp_path)
            temp_path.rename(path)  # Atomic on POSIX filesystems
            print(f"Saved checkpoint: {path}", flush=True)
            
            # Create/update 'latest' symlink
            latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
            if latest_path.exists() or latest_path.is_symlink():
                latest_path.unlink()
            latest_path.symlink_to(path.name)
            
            # Keep only last 3 checkpoints to save disk space
            checkpoints = sorted([
                c for c in self.checkpoint_dir.glob("checkpoint_*.pt") 
                if not c.is_symlink() and c.name != "checkpoint_latest.pt"
            ])
            for ckpt in checkpoints[:-3]:
                ckpt.unlink()
                
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}", flush=True)
            # Clean up temp file if it exists
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
        
        # Signal other ranks that save is complete
        if self.is_distributed:
            dist.barrier()
    
    def load_checkpoint(self, path: str) -> Optional[str]:
        """Load training checkpoint."""
        print(f"Loading checkpoint from: {path}", flush=True)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.global_step = checkpoint['global_step']
        self.episodes_collected = checkpoint['episodes_collected']
        self.train_steps_done = checkpoint['train_steps_done']
        
        self._world_model.load_state_dict(checkpoint['world_model'])
        self._actor_critic.load_state_dict(checkpoint['actor_critic'])
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
            return latest.resolve() if latest.is_symlink() else latest
        
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
        
        def get_step(p):
            try:
                return int(p.stem.split('_')[1])
            except:
                return -1
        
        return max(checkpoints, key=get_step)
    
    # ==================== Metrics ====================
    
    def _accumulate_metrics(self, accumulator: Dict[str, list], metrics: Dict[str, float]):
        """Accumulate metrics for averaging."""
        for key, value in metrics.items():
            if key not in accumulator:
                accumulator[key] = []
            accumulator[key].append(value)
    
    def _average_metrics(self, accumulator: Dict[str, list]) -> Dict[str, float]:
        """Average accumulated metrics."""
        return {key: np.mean(values) for key, values in accumulator.items()}
    
    # ==================== Logging ====================
    
    def _log_reconstructions(self):
        """Log image reconstructions to WandB. Only rank 0."""
        if not self.logger:
            return
             
        min_steps = self.batch_length
        if self.replay_buffer.num_steps < min_steps:
            return
        
        try:
            batch = self.replay_buffer.sample(1, self.batch_length)
            
            with torch.no_grad():
                obs = batch['obs'].to(self.device)
                action = batch['action'].to(self.device)
                is_first = batch['is_first'].to(self.device)
                
                output = self._world_model(obs, action, is_first)
                recon = self._world_model.decode(output.features)
                
                original = obs[0, 0]
                reconstructed = recon[0, 0]
                
                self.logger.log_image("train/original", original, self.global_step)
                self.logger.log_image("train/reconstructed", reconstructed, self.global_step)
        except Exception as e:
            print(f"Warning: Failed to log reconstructions: {e}", flush=True)
    
    def _log_dream_video(self):
        """Log observation and reconstructed video to WandB. Only rank 0."""
        if not self.logger:
            return

        min_steps = self.batch_length * 4
        if self.replay_buffer.num_steps < min_steps:
            return
        
        try:
            batch = self.replay_buffer.sample(2, min(32, self.batch_length))
            
            with torch.no_grad():
                obs = batch['obs'].to(self.device)
                action = batch['action'].to(self.device)
                is_first = batch['is_first'].to(self.device)
                
                output = self._world_model(obs, action, is_first)
                recon = self._world_model.decode(output.features)
                
                self.logger.log_video("train/observation_video", obs[:1], self.global_step, fps=10)
                self.logger.log_video("train/reconstruction_video", recon[:1], self.global_step, fps=10)
        except Exception as e:
            print(f"Warning: Failed to log video: {e}", flush=True)
    
    def _log_imagination_video(self):
        """
        Log imagined future video rollout to WandB. Only rank 0.
        
        This is the key DreamerV3 visualization: starting from a real observation,
        the model imagines future frames using the learned world model and policy.
        The imagined frames are compared with the actual ground truth futures.
        
        Logs:
        - train/imagination_ground_truth: Actual future frames from environment
        - train/imagination_predicted: Imagined future frames from world model
        """
        if not self.logger:
            return
        
        # Need enough data for context + imagination horizon
        imagination_horizon = getattr(self, 'imagination_horizon', 15)
        context_frames = 5  # Use first few frames to establish state
        total_needed = context_frames + imagination_horizon
        
        min_steps = self.batch_length * 4
        if self.replay_buffer.num_steps < min_steps:
            return
        
        try:
            # Sample a sequence long enough for context + future ground truth
            seq_length = min(total_needed + 5, self.batch_length)
            batch = self.replay_buffer.sample(1, seq_length)
            
            with torch.no_grad():
                obs = batch['obs'].to(self.device)  # (1, T, C, H, W)
                action = batch['action'].to(self.device)  # (1, T, action_dim)
                is_first = batch['is_first'].to(self.device)  # (1, T)
                
                # Get context frames to establish state
                context_obs = obs[:, :context_frames]
                context_action = action[:, :context_frames]
                context_is_first = is_first[:, :context_frames]
                
                # Run world model on context to get posterior state
                output = self._world_model(context_obs, context_action, context_is_first)
                
                # Get the final state from context as starting point for imagination
                # posterior shape: (batch, time, ...) - we want the last timestep
                start_state = RSSMState(
                    deter=output.posterior.deter[:, -1],  # (batch, deter_size)
                    stoch=output.posterior.stoch[:, -1],  # (batch, stoch_size, classes)
                    logits=output.posterior.logits[:, -1],  # (batch, stoch_size, classes)
                )
                start_features = self._world_model.rssm.get_features(start_state)
                
                # Imagine future trajectories using policy
                imagined_features, imagined_actions, imagined_rewards, imagined_continues = \
                    self._world_model.imagine_with_policy(
                        start_features,
                        start_state,
                        self._actor_critic.actor,
                        imagination_horizon,
                    )
                
                # Decode imagined features to images
                # imagined_features: (batch, horizon, feature_dim)
                imagined_obs = self._world_model.decode(imagined_features)  # (batch, horizon, C, H, W)
                
                # Get ground truth future frames for comparison
                future_start = context_frames
                future_end = min(context_frames + imagination_horizon, obs.shape[1])
                true_future_obs = obs[:, future_start:future_end]  # (1, horizon, C, H, W)
                
                # Match lengths (in case ground truth is shorter)
                actual_horizon = true_future_obs.shape[1]
                imagined_obs_matched = imagined_obs[:, :actual_horizon]
                
                # Log videos using same format as _log_dream_video (keep batch dim, use [:1])
                # This ensures consistency with observation_video and reconstruction_video
                self.logger.log_video(
                    "train/imagination_ground_truth",
                    true_future_obs[:1],  # (1, T, C, H, W) - keep batch dim like other videos
                    self.global_step,
                    fps=10,
                    caption="Ground Truth Future"
                )
                self.logger.log_video(
                    "train/imagination_predicted",
                    imagined_obs_matched[:1],  # (1, T, C, H, W) - keep batch dim
                    self.global_step,
                    fps=10,
                    caption="Imagined Future (World Model + Policy)"
                )
                
                # Also log predicted rewards
                if imagined_rewards is not None:
                    # Log mean predicted reward over imagination
                    mean_reward = imagined_rewards.mean().item()
                    self.logger.log_scalar("train/imagination_mean_reward", mean_reward, self.global_step)
                    
        except Exception as e:
            import traceback
            print(f"Warning: Failed to log imagination video: {e}", flush=True)
            traceback.print_exc()
