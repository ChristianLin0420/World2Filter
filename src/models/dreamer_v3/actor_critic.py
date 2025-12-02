"""
Actor-Critic networks for DreamerV3.

The actor learns a policy for action selection, while the critic
estimates value functions for policy optimization via imagination.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.dreamer_v3.networks import (
    MLP, 
    SymlogDist, 
    OneHotDist, 
    TruncNormalDist,
    get_activation,
    get_norm,
    symlog,
    symexp,
)


class Actor(nn.Module):
    """
    Actor network for policy learning.
    
    Outputs action distributions for both discrete and continuous action spaces.
    Uses truncated normal distribution for continuous actions.
    """
    
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 5,
        activation: str = "silu",
        norm: str = "layer",
        dist_type: str = "trunc_normal",
        min_std: float = 0.1,
        max_std: float = 1.0,
        init_std: float = 0.0,
        unimix_ratio: float = 0.01,
    ):
        """
        Args:
            feature_dim: Dimension of input features (from RSSM)
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            activation: Activation function
            norm: Normalization type
            dist_type: Distribution type ('trunc_normal', 'onehot')
            min_std: Minimum standard deviation for continuous actions
            max_std: Maximum standard deviation for continuous actions
            init_std: Initial standard deviation (0 means learned from scratch)
            unimix_ratio: Uniform mixing ratio for exploration
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.dist_type = dist_type
        self.min_std = min_std
        self.max_std = max_std
        self.unimix_ratio = unimix_ratio
        
        # Build MLP backbone
        self.net = MLP(
            feature_dim,
            hidden_dim,  # Output to hidden before heads
            hidden_dim=hidden_dim,
            num_layers=num_layers - 1,
            activation=activation,
            norm=norm,
        )
        
        if dist_type == "trunc_normal":
            # Mean and std heads for continuous actions
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.std_head = nn.Linear(hidden_dim, action_dim)
            
            # Initialize std head to produce init_std
            if init_std > 0:
                nn.init.zeros_(self.std_head.weight)
                init_bias = torch.log(torch.exp(torch.tensor(init_std)) - 1)
                nn.init.constant_(self.std_head.bias, init_bias)
        else:
            # Logits head for discrete actions
            self.logits_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(
        self,
        features: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute action from features.
        
        Args:
            features: Input features from RSSM
            deterministic: If True, return mode instead of sample
        
        Returns:
            Tuple of (action, info_dict)
        """
        hidden = self.net(features)
        
        if self.dist_type == "trunc_normal":
            mean = self.mean_head(hidden)
            std = self.std_head(hidden)
            
            # Constrain std to [min_std, max_std]
            std = self.max_std * torch.sigmoid(std) + self.min_std
            
            dist = TruncNormalDist(mean, std, low=-1.0, high=1.0)
            
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()
            
            info = {
                'mean': mean,
                'std': std,
                'entropy': dist.entropy(),
                'log_prob': dist.log_prob(action),
            }
        else:
            logits = self.logits_head(hidden)
            dist = OneHotDist(logits, unimix_ratio=self.unimix_ratio)
            
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()
            
            info = {
                'logits': logits,
                'entropy': dist.entropy(),
                'log_prob': dist.log_prob(action),
            }
        
        return action, info
    
    def log_prob(self, features: Tensor, action: Tensor) -> Tensor:
        """Compute log probability of action given features."""
        hidden = self.net(features)
        
        if self.dist_type == "trunc_normal":
            mean = self.mean_head(hidden)
            std = self.std_head(hidden)
            std = self.max_std * torch.sigmoid(std) + self.min_std
            dist = TruncNormalDist(mean, std, low=-1.0, high=1.0)
        else:
            logits = self.logits_head(hidden)
            dist = OneHotDist(logits, unimix_ratio=self.unimix_ratio)
        
        return dist.log_prob(action)
    
    def entropy(self, features: Tensor) -> Tensor:
        """Compute entropy of action distribution."""
        hidden = self.net(features)
        
        if self.dist_type == "trunc_normal":
            mean = self.mean_head(hidden)
            std = self.std_head(hidden)
            std = self.max_std * torch.sigmoid(std) + self.min_std
            dist = TruncNormalDist(mean, std, low=-1.0, high=1.0)
        else:
            logits = self.logits_head(hidden)
            dist = OneHotDist(logits, unimix_ratio=self.unimix_ratio)
        
        return dist.entropy()


class Critic(nn.Module):
    """
    Critic network for value estimation.
    
    Uses symlog discrete distribution for stable value prediction.
    Includes slow target network for stable training.
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 5,
        activation: str = "silu",
        norm: str = "layer",
        dist_type: str = "symlog_disc",
        num_bins: int = 255,
        slow_target: bool = True,
        slow_target_update: float = 0.02,
        slow_target_fraction: float = 0.95,
    ):
        """
        Args:
            feature_dim: Dimension of input features
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            activation: Activation function
            norm: Normalization type
            dist_type: Value distribution type ('symlog_disc', 'mse')
            num_bins: Number of bins for discrete distribution
            slow_target: Whether to use slow target network
            slow_target_update: EMA update rate for slow target
            slow_target_fraction: Fraction of slow target in combined estimate
        """
        super().__init__()
        
        self.dist_type = dist_type
        self.num_bins = num_bins
        self.slow_target = slow_target
        self.slow_target_update = slow_target_update
        self.slow_target_fraction = slow_target_fraction
        
        # Main value network
        if dist_type == "symlog_disc":
            self.net = MLP(
                feature_dim,
                num_bins,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                activation=activation,
                norm=norm,
            )
            self.symlog_dist = SymlogDist(num_bins)
        else:
            self.net = MLP(
                feature_dim,
                1,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                activation=activation,
                norm=norm,
            )
        
        # Slow target network (EMA copy)
        if slow_target:
            if dist_type == "symlog_disc":
                self.target_net = MLP(
                    feature_dim,
                    num_bins,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    activation=activation,
                    norm=norm,
                )
            else:
                self.target_net = MLP(
                    feature_dim,
                    1,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    activation=activation,
                    norm=norm,
                )
            # Initialize target with same weights
            self.target_net.load_state_dict(self.net.state_dict())
            # Freeze target network
            for param in self.target_net.parameters():
                param.requires_grad = False
    
    def forward(self, features: Tensor) -> Tensor:
        """
        Compute value estimate.
        
        Args:
            features: Input features from RSSM
        
        Returns:
            Value estimates
        """
        if self.dist_type == "symlog_disc":
            logits = self.net(features)
            value = self.symlog_dist(logits)
        else:
            value = self.net(features).squeeze(-1)
        
        return value
    
    def target(self, features: Tensor) -> Tensor:
        """
        Compute value estimate from slow target network.
        """
        if not self.slow_target:
            return self.forward(features)
        
        with torch.no_grad():
            if self.dist_type == "symlog_disc":
                logits = self.target_net(features)
                value = self.symlog_dist(logits)
            else:
                value = self.target_net(features).squeeze(-1)
        
        return value
    
    def loss(self, features: Tensor, target_value: Tensor) -> Tensor:
        """
        Compute value loss.
        
        Args:
            features: Input features
            target_value: Target value (e.g., lambda returns)
        
        Returns:
            Value loss
        """
        if self.dist_type == "symlog_disc":
            logits = self.net(features)
            loss = -self.symlog_dist.log_prob(logits, target_value)
        else:
            pred = self.net(features).squeeze(-1)
            # Symlog transformation for stability
            loss = (symlog(pred) - symlog(target_value)) ** 2
        
        return loss
    
    def update_target(self):
        """Update slow target network using EMA."""
        if not self.slow_target:
            return
        
        with torch.no_grad():
            for param, target_param in zip(
                self.net.parameters(),
                self.target_net.parameters()
            ):
                target_param.data.lerp_(param.data, self.slow_target_update)


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic module for policy learning.
    
    Handles:
    - Action selection (actor)
    - Value estimation (critic)
    - Lambda returns computation
    - Return normalization
    """
    
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        actor_config: Dict,
        critic_config: Dict,
        discount: float = 0.997,
        lambda_: float = 0.95,
        return_normalization: bool = True,
        return_ema_decay: float = 0.99,
        return_scale_lo: float = 1.0,
        return_scale_hi: float = 1.0,
    ):
        """
        Args:
            feature_dim: Dimension of input features
            action_dim: Dimension of action space
            actor_config: Configuration for actor network
            critic_config: Configuration for critic network
            discount: Discount factor for returns
            lambda_: Lambda for TD(lambda) returns
            return_normalization: Whether to normalize returns
            return_ema_decay: EMA decay for return statistics
            return_scale_lo: Lower percentile for return scaling
            return_scale_hi: Upper percentile for return scaling
        """
        super().__init__()
        
        self.discount = discount
        self.lambda_ = lambda_
        self.return_normalization = return_normalization
        self.return_ema_decay = return_ema_decay
        
        # Build actor and critic
        self.actor = Actor(feature_dim, action_dim, **actor_config)
        self.critic = Critic(feature_dim, **critic_config)
        
        # Return normalization statistics
        if return_normalization:
            self.register_buffer('return_ema_lo', torch.tensor(return_scale_lo))
            self.register_buffer('return_ema_hi', torch.tensor(return_scale_hi))
    
    def act(
        self,
        features: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Select action given features."""
        return self.actor(features, deterministic=deterministic)
    
    def value(self, features: Tensor) -> Tensor:
        """Estimate value given features."""
        return self.critic(features)
    
    def target_value(self, features: Tensor) -> Tensor:
        """Estimate value using slow target network."""
        return self.critic.target(features)
    
    def compute_lambda_returns(
        self,
        rewards: Tensor,
        values: Tensor,
        continues: Tensor,
        bootstrap: Tensor,
    ) -> Tensor:
        """
        Compute TD(lambda) returns.
        
        Args:
            rewards: Rewards (batch, horizon)
            values: Value estimates (batch, horizon)
            continues: Continue flags (batch, horizon)
            bootstrap: Bootstrap value (batch,)
        
        Returns:
            Lambda returns (batch, horizon)
        """
        # Append bootstrap value
        next_values = torch.cat([values[:, 1:], bootstrap.unsqueeze(1)], dim=1)
        
        # Compute lambda returns backwards
        returns = torch.zeros_like(rewards)
        last_return = bootstrap
        
        for t in reversed(range(rewards.shape[1])):
            td_target = rewards[:, t] + self.discount * continues[:, t] * next_values[:, t]
            returns[:, t] = td_target + self.discount * self.lambda_ * continues[:, t] * (
                last_return - next_values[:, t]
            )
            last_return = returns[:, t]
        
        return returns
    
    def normalize_returns(self, returns: Tensor) -> Tensor:
        """
        Normalize returns using EMA statistics.
        
        Args:
            returns: Unnormalized returns
        
        Returns:
            Normalized returns
        """
        if not self.return_normalization:
            return returns
        
        device = returns.device
        
        # Update EMA statistics
        with torch.no_grad():
            lo = returns.min()
            hi = returns.max()
            # Ensure buffers are on the same device
            self.return_ema_lo = self.return_ema_lo.to(device)
            self.return_ema_hi = self.return_ema_hi.to(device)
            self.return_ema_lo.lerp_(lo, 1 - self.return_ema_decay)
            self.return_ema_hi.lerp_(hi, 1 - self.return_ema_decay)
        
        # Normalize
        scale = torch.clamp(self.return_ema_hi - self.return_ema_lo, min=1.0)
        normalized = (returns - self.return_ema_lo) / scale
        
        return normalized
    
    def actor_loss(
        self,
        features: Tensor,
        actions: Tensor,
        advantages: Tensor,
        entropy_scale: float = 3e-4,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute actor loss using policy gradient.
        
        Args:
            features: RSSM features
            actions: Actions taken
            advantages: Advantage estimates (normalized returns - values)
            entropy_scale: Entropy bonus coefficient
        
        Returns:
            Tuple of (loss, info_dict)
        """
        # Get action distribution info
        _, info = self.actor(features)
        log_prob = self.actor.log_prob(features, actions)
        entropy = self.actor.entropy(features)
        
        # Policy gradient loss
        pg_loss = -(log_prob * advantages.detach()).mean()
        
        # Entropy bonus
        entropy_loss = -entropy_scale * entropy.mean()
        
        # Total loss
        loss = pg_loss + entropy_loss
        
        return loss, {
            'actor_loss': loss.detach(),
            'pg_loss': pg_loss.detach(),
            'entropy': entropy.mean().detach(),
            'log_prob': log_prob.mean().detach(),
        }
    
    def critic_loss(
        self,
        features: Tensor,
        returns: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute critic loss.
        
        Args:
            features: RSSM features
            returns: Target returns
        
        Returns:
            Tuple of (loss, info_dict)
        """
        loss = self.critic.loss(features, returns).mean()
        value = self.critic(features)
        
        return loss, {
            'critic_loss': loss.detach(),
            'value_mean': value.mean().detach(),
            'value_std': value.std().detach(),
        }
    
    def update_slow_target(self):
        """Update critic's slow target network."""
        self.critic.update_target()

