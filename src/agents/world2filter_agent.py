"""
World2Filter Agent.

Agent that uses the FG/BG segmented world model for decision making.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.models.world2filter.fg_bg_world_model import FGBGWorldModel
from src.models.dreamer_v3.actor_critic import ActorCritic
from src.models.dreamer_v3.rssm import RSSMState
from src.models.segmentation.mask_processor import MaskProcessor, OnlineMaskProcessor


class World2FilterAgent(nn.Module):
    """
    World2Filter Agent.
    
    Combines:
    - FG/BG World Model for segmented dynamics learning
    - SAM3-based mask generation
    - Actor-Critic for policy learning
    """
    
    def __init__(
        self,
        world_model: FGBGWorldModel,
        actor_critic: ActorCritic,
        mask_processor: Optional[MaskProcessor] = None,
        expl_noise: float = 0.0,
        expl_decay: float = 0.0,
        expl_min: float = 0.0,
    ):
        """
        Args:
            world_model: FG/BG World Model
            actor_critic: Actor-Critic module
            mask_processor: Mask processor for SAM3
            expl_noise: Initial exploration noise
            expl_decay: Exploration noise decay
            expl_min: Minimum exploration noise
        """
        super().__init__()
        
        self.world_model = world_model
        self.actor_critic = actor_critic
        self.mask_processor = mask_processor
        
        # Exploration settings
        self.expl_noise = expl_noise
        self.expl_decay = expl_decay
        self.expl_min = expl_min
        
        # State tracking
        self._prev_state: Optional[RSSMState] = None
        self._prev_action: Optional[Tensor] = None
        
        # Online mask processor for environment interaction
        if mask_processor is not None:
            self._online_mask_processor = OnlineMaskProcessor(mask_processor)
        else:
            self._online_mask_processor = None
        
        # Step counter
        self.global_step = 0
    
    def reset(self):
        """Reset agent state for new episode."""
        self._prev_state = None
        self._prev_action = None
        
        if self._online_mask_processor is not None:
            self._online_mask_processor.reset()
    
    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        is_first: bool = False,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select action given observation.
        
        Args:
            obs: Observation (C, H, W) or (H, W, C)
            is_first: Whether this is the first step of an episode
            deterministic: Whether to use deterministic policy
        
        Returns:
            Action as numpy array
        """
        device = next(self.parameters()).device
        
        # Ensure CHW format
        if obs.ndim == 3 and obs.shape[-1] in [1, 3]:
            obs = np.transpose(obs, (2, 0, 1))
        
        # Process mask if available
        if self._online_mask_processor is not None:
            fg_mask, bg_mask = self._online_mask_processor.add_observation(obs)
        
        # Convert to tensor
        obs_tensor = torch.tensor(
            obs[np.newaxis],
            dtype=torch.float32,
            device=device,
        )
        is_first_tensor = torch.tensor(
            [is_first],
            dtype=torch.bool,
            device=device,
        )
        
        # Initialize action if needed
        if self._prev_action is None or is_first:
            self._prev_action = torch.zeros(
                1, self.world_model.action_dim,
                dtype=torch.float32,
                device=device,
            )
        
        # Get RSSM state and features
        if is_first:
            self._prev_state = None
        
        state, features = self.world_model.obs_step(
            obs_tensor,
            self._prev_action,
            is_first_tensor,
            self._prev_state,
        )
        
        # Get action from policy
        action, info = self.actor_critic.act(
            features,
            deterministic=deterministic,
        )
        
        # Add exploration noise
        if not deterministic and self.expl_noise > 0:
            noise = torch.randn_like(action) * self.current_expl_noise
            action = (action + noise).clamp(-1, 1)
        
        # Update state
        self._prev_state = state
        self._prev_action = action
        
        return action[0].cpu().numpy()
    
    @property
    def current_expl_noise(self) -> float:
        """Get current exploration noise level."""
        return max(
            self.expl_min,
            self.expl_noise - self.global_step * self.expl_decay
        )
    
    def imagine(
        self,
        initial_features: Tensor,
        initial_state: RSSMState,
        horizon: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Imagine future trajectories from initial state.
        
        Args:
            initial_features: Starting features
            initial_state: Starting RSSM state
            horizon: Imagination horizon
        
        Returns:
            Tuple of (features, actions, rewards, continues)
        """
        return self.world_model.imagine_with_policy(
            initial_features,
            initial_state,
            self.actor_critic.actor,
            horizon,
        )
    
    def get_features(self, state: RSSMState) -> Tensor:
        """Get features from RSSM state."""
        return self.world_model.rssm.get_features(state)
    
    def get_episode_masks(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get masks for current episode."""
        if self._online_mask_processor is None:
            return None
        return self._online_mask_processor.get_episode_masks()
    
    def save(self, path: str):
        """Save agent to file."""
        torch.save({
            'world_model': self.world_model.state_dict(),
            'actor_critic': self.actor_critic.state_dict(),
            'global_step': self.global_step,
            'expl_noise': self.expl_noise,
        }, path)
    
    def load(self, path: str, device: str = 'cuda'):
        """Load agent from file."""
        checkpoint = torch.load(path, map_location=device)
        
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.global_step = checkpoint.get('global_step', 0)
        self.expl_noise = checkpoint.get('expl_noise', self.expl_noise)


def build_world2filter_agent(
    config: Dict,
    use_sam3: bool = True,
) -> World2FilterAgent:
    """
    Build World2Filter agent from configuration.
    
    Args:
        config: Configuration dictionary
        use_sam3: Whether to use SAM3 for mask generation
    
    Returns:
        Configured World2FilterAgent
    """
    from src.models.world2filter.fg_bg_world_model import build_fg_bg_world_model
    from src.models.segmentation.mask_processor import MaskProcessor
    
    # Get environment info
    obs_shape = (
        config['environment']['obs']['channels'],
        config['environment']['obs']['image_size'],
        config['environment']['obs']['image_size'],
    )
    action_dim = config.get('action_dim', 6)
    
    # Build world model
    world_model = build_fg_bg_world_model(config)
    
    # Build actor-critic
    feature_dim = world_model.feature_dim
    
    actor_config = {
        'hidden_dim': config['actor']['units'],
        'num_layers': config['actor']['layers'],
        'activation': config['actor']['act'],
        'norm': config['actor']['norm'],
        'dist_type': config['actor']['dist'],
        'min_std': config['actor']['min_std'],
        'max_std': config['actor']['max_std'],
        'unimix_ratio': config['actor']['unimix_ratio'],
    }
    
    critic_config = {
        'hidden_dim': config['critic']['units'],
        'num_layers': config['critic']['layers'],
        'activation': config['critic']['act'],
        'norm': config['critic']['norm'],
        'dist_type': config['critic']['dist'],
        'num_bins': config['critic']['bins'],
        'slow_target': config['critic']['slow_target'],
        'slow_target_update': config['critic']['slow_target_update'],
        'slow_target_fraction': config['critic']['slow_target_fraction'],
    }
    
    actor_critic = ActorCritic(
        feature_dim=feature_dim,
        action_dim=action_dim,
        actor_config=actor_config,
        critic_config=critic_config,
        discount=config['imagination']['discount'],
        lambda_=config['imagination']['lambda_'],
        return_normalization=config['imagination']['return_normalization'],
        return_ema_decay=config['imagination']['return_ema_decay'],
    )
    
    # Build mask processor
    seg_config = config.get('segmentation', {})
    if seg_config.get('enabled', False) and use_sam3:
        mask_processor = MaskProcessor(
            cache_dir=seg_config.get('cache_dir', './mask_cache'),
            prompt=seg_config.get('prompt', 'robot'),
            device=seg_config.get('device', 'cuda'),
            use_sam3=True,
        )
    else:
        mask_processor = None
    
    # Build agent
    agent = World2FilterAgent(
        world_model=world_model,
        actor_critic=actor_critic,
        mask_processor=mask_processor,
    )
    
    return agent


class World2FilterAgentWrapper:
    """
    Convenience wrapper for World2FilterAgent.
    """
    
    def __init__(
        self,
        agent: World2FilterAgent,
        device: str = 'cuda',
    ):
        self.agent = agent.to(device)
        self.device = device
        self._is_first = True
    
    def reset(self):
        """Reset for new episode."""
        self.agent.reset()
        self._is_first = True
    
    def __call__(
        self,
        obs: np.ndarray,
        reward: float = 0.0,
        done: bool = False,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Get action given observation."""
        if done:
            self.reset()
            self._is_first = True
        
        action = self.agent.act(
            obs,
            is_first=self._is_first,
            deterministic=deterministic,
        )
        
        self._is_first = False
        
        return action
    
    def save(self, path: str):
        """Save agent."""
        self.agent.save(path)
    
    def load(self, path: str):
        """Load agent."""
        self.agent.load(path, device=self.device)

