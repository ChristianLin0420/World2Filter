"""
Loss Functions for DreamerV3.

Implements all losses for world model and actor-critic training:
- Image reconstruction loss
- KL divergence loss with free bits and balancing
- Reward prediction loss (symlog discrete)
- Continue prediction loss (binary cross-entropy)
- Actor loss (policy gradient + entropy)
- Critic loss (value regression)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.dreamer_v3.networks import symlog, symexp, SymlogDist
from src.models.dreamer_v3.rssm import RSSMState


class WorldModelLoss(nn.Module):
    """
    Combined loss for DreamerV3 world model training.
    
    Components:
    - Image reconstruction loss (MSE)
    - KL divergence (with free bits and balancing)
    - Reward prediction loss (symlog discrete)
    - Continue prediction loss (binary)
    """
    
    def __init__(
        self,
        kl_free: float = 1.0,
        kl_forward: bool = False,
        kl_balance: float = 0.8,
        kl_weight: float = 1.0,
        image_weight: float = 1.0,
        reward_weight: float = 1.0,
        continue_weight: float = 1.0,
        reward_bins: int = 255,
    ):
        """
        Args:
            kl_free: Free bits for KL loss
            kl_forward: If True, use forward KL
            kl_balance: Balance between prior and posterior KL
            kl_weight: Weight for KL loss
            image_weight: Weight for image reconstruction loss
            reward_weight: Weight for reward prediction loss
            continue_weight: Weight for continue prediction loss
            reward_bins: Number of bins for reward distribution
        """
        super().__init__()
        
        self.kl_free = kl_free
        self.kl_forward = kl_forward
        self.kl_balance = kl_balance
        self.kl_weight = kl_weight
        self.image_weight = image_weight
        self.reward_weight = reward_weight
        self.continue_weight = continue_weight
        
        self.reward_dist = SymlogDist(reward_bins)
    
    def forward(
        self,
        prior: RSSMState,
        posterior: RSSMState,
        image_pred: Tensor,
        image_target: Tensor,
        reward_logits: Tensor,
        reward_target: Tensor,
        continue_logits: Tensor,
        continue_target: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute total world model loss.
        
        Args:
            prior: Prior RSSM state
            posterior: Posterior RSSM state
            image_pred: Predicted images
            image_target: Target images
            reward_logits: Reward prediction logits
            reward_target: Target rewards
            continue_logits: Continue prediction logits
            continue_target: Target continue signals
            mask: Optional mask for valid timesteps
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        batch_size, seq_len = image_target.shape[:2]
        
        # Normalize image target
        if image_target.dtype == torch.uint8:
            image_target = image_target.float() / 255.0 - 0.5
        elif image_target.max() > 1.0:
            image_target = image_target / 255.0 - 0.5
        
        # Image reconstruction loss (MSE)
        image_loss = self._image_loss(image_pred, image_target)
        
        # KL divergence loss
        kl_loss, kl_value = self._kl_loss(prior, posterior)
        
        # Reward prediction loss
        reward_loss = self._reward_loss(reward_logits, reward_target)
        
        # Continue prediction loss
        continue_loss = self._continue_loss(continue_logits, continue_target)
        
        # Apply mask if provided
        if mask is not None:
            image_loss = image_loss * mask
            kl_loss = kl_loss * mask
            reward_loss = reward_loss * mask
            continue_loss = continue_loss * mask
        
        # Average over batch and sequence
        image_loss = image_loss.mean()
        kl_loss = kl_loss.mean()
        kl_value = kl_value.mean()
        reward_loss = reward_loss.mean()
        continue_loss = continue_loss.mean()
        
        # Weighted total loss
        total_loss = (
            self.image_weight * image_loss +
            self.kl_weight * kl_loss +
            self.reward_weight * reward_loss +
            self.continue_weight * continue_loss
        )
        
        # Collect metrics
        metrics = {
            'total_loss': total_loss.detach(),
            'image_loss': image_loss.detach(),
            'kl_loss': kl_loss.detach(),
            'kl_value': kl_value.detach(),
            'reward_loss': reward_loss.detach(),
            'continue_loss': continue_loss.detach(),
        }
        
        return total_loss, metrics
    
    def _image_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute image reconstruction loss."""
        # MSE loss per pixel, averaged over channels
        loss = (pred - target) ** 2
        loss = loss.mean(dim=(-3, -2, -1))  # Average over C, H, W
        return loss
    
    def _kl_loss(
        self,
        prior: RSSMState,
        posterior: RSSMState,
    ) -> Tuple[Tensor, Tensor]:
        """Compute KL divergence loss with free bits and balancing."""
        prior_logits = prior.logits
        posterior_logits = posterior.logits
        
        prior_probs = F.softmax(prior_logits, dim=-1)
        posterior_probs = F.softmax(posterior_logits, dim=-1)
        
        # Compute KL divergence
        if self.kl_forward:
            kl = prior_probs * (
                torch.log(prior_probs + 1e-8) - torch.log(posterior_probs + 1e-8)
            )
        else:
            kl = posterior_probs * (
                torch.log(posterior_probs + 1e-8) - torch.log(prior_probs + 1e-8)
            )
        
        # Sum over classes and stoch dimensions
        kl = kl.sum(dim=-1).sum(dim=-1)
        
        # Store raw KL value
        kl_value = kl
        
        # Apply free bits
        kl = torch.clamp(kl, min=self.kl_free)
        
        # KL balancing
        sg_posterior = posterior_probs.detach()
        kl_prior = sg_posterior * (
            torch.log(sg_posterior + 1e-8) - torch.log(prior_probs + 1e-8)
        )
        kl_prior = kl_prior.sum(dim=-1).sum(dim=-1)
        kl_prior = torch.clamp(kl_prior, min=self.kl_free)
        
        sg_prior = prior_probs.detach()
        kl_posterior = posterior_probs * (
            torch.log(posterior_probs + 1e-8) - torch.log(sg_prior + 1e-8)
        )
        kl_posterior = kl_posterior.sum(dim=-1).sum(dim=-1)
        kl_posterior = torch.clamp(kl_posterior, min=self.kl_free)
        
        # Balanced loss
        kl_loss = self.kl_balance * kl_prior + (1 - self.kl_balance) * kl_posterior
        
        return kl_loss, kl_value
    
    def _reward_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute reward prediction loss using symlog discrete distribution."""
        # Negative log probability under symlog discrete distribution
        loss = -self.reward_dist.log_prob(logits, target)
        return loss
    
    def _continue_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute continue prediction loss using binary cross-entropy."""
        loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1),
            target.float(),
            reduction='none',
        )
        return loss


class ActorCriticLoss(nn.Module):
    """
    Loss functions for actor-critic training via imagination.
    """
    
    def __init__(
        self,
        discount: float = 0.997,
        lambda_: float = 0.95,
        entropy_scale: float = 3e-4,
        return_normalization: bool = True,
        return_ema_decay: float = 0.99,
    ):
        """
        Args:
            discount: Discount factor
            lambda_: TD(lambda) parameter
            entropy_scale: Entropy bonus coefficient
            return_normalization: Normalize returns
            return_ema_decay: EMA decay for return statistics
        """
        super().__init__()
        
        self.discount = discount
        self.lambda_ = lambda_
        self.entropy_scale = entropy_scale
        self.return_normalization = return_normalization
        self.return_ema_decay = return_ema_decay
        
        # EMA buffers for return normalization
        self.register_buffer('return_lo', torch.tensor(0.0))
        self.register_buffer('return_hi', torch.tensor(1.0))
    
    def compute_returns(
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
            continues: Continue probabilities (batch, horizon)
            bootstrap: Bootstrap value (batch,)
        
        Returns:
            Lambda returns (batch, horizon)
        """
        # Append bootstrap
        next_values = torch.cat([values[:, 1:], bootstrap.unsqueeze(1)], dim=1)
        
        # Compute returns backwards
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
        """Normalize returns using EMA statistics."""
        if not self.return_normalization:
            return returns
        
        device = returns.device
        
        with torch.no_grad():
            lo = returns.min()
            hi = returns.max()
            # Ensure buffers are on the same device
            self.return_lo = self.return_lo.to(device)
            self.return_hi = self.return_hi.to(device)
            self.return_lo.lerp_(lo, 1 - self.return_ema_decay)
            self.return_hi.lerp_(hi, 1 - self.return_ema_decay)
        
        scale = torch.clamp(self.return_hi - self.return_lo, min=1.0)
        return (returns - self.return_lo) / scale
    
    def actor_loss(
        self,
        log_probs: Tensor,
        entropy: Tensor,
        advantages: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute actor loss.
        
        Args:
            log_probs: Log probabilities of actions
            entropy: Entropy of action distribution
            advantages: Advantage estimates
        
        Returns:
            Tuple of (loss, metrics)
        """
        # Policy gradient loss
        pg_loss = -(log_probs * advantages.detach()).mean()
        
        # Entropy bonus
        entropy_loss = -self.entropy_scale * entropy.mean()
        
        total_loss = pg_loss + entropy_loss
        
        metrics = {
            'actor_loss': total_loss.detach(),
            'pg_loss': pg_loss.detach(),
            'entropy': entropy.mean().detach(),
            'log_prob': log_probs.mean().detach(),
            'advantage_mean': advantages.mean().detach(),
            'advantage_std': advantages.std().detach(),
        }
        
        return total_loss, metrics
    
    def critic_loss(
        self,
        values: Tensor,
        returns: Tensor,
        critic_dist: Optional[SymlogDist] = None,
        value_logits: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute critic loss.
        
        Args:
            values: Value predictions
            returns: Target returns
            critic_dist: Optional symlog distribution for discrete values
            value_logits: Logits for discrete value prediction
        
        Returns:
            Tuple of (loss, metrics)
        """
        if critic_dist is not None and value_logits is not None:
            # Discrete value prediction
            loss = -critic_dist.log_prob(value_logits, returns)
            loss = loss.mean()
        else:
            # MSE loss with symlog transformation
            loss = (symlog(values) - symlog(returns.detach())) ** 2
            loss = loss.mean()
        
        metrics = {
            'critic_loss': loss.detach(),
            'value_mean': values.mean().detach(),
            'value_std': values.std().detach(),
            'return_mean': returns.mean().detach(),
            'return_std': returns.std().detach(),
        }
        
        return loss, metrics


class FGBGReconstructionLoss(nn.Module):
    """
    Foreground/Background reconstruction loss for World2Filter.
    
    Computes separate losses for foreground and background reconstruction
    using SAM3-generated masks.
    """
    
    def __init__(
        self,
        fg_weight: float = 1.0,
        bg_weight: float = 1.0,
        mask_smoothing: float = 0.1,
    ):
        """
        Args:
            fg_weight: Weight for foreground reconstruction loss
            bg_weight: Weight for background reconstruction loss
            mask_smoothing: Smoothing factor for mask edges
        """
        super().__init__()
        
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        self.mask_smoothing = mask_smoothing
    
    def forward(
        self,
        fg_pred: Tensor,
        bg_pred: Tensor,
        target: Tensor,
        fg_mask: Tensor,
        bg_mask: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute FG/BG reconstruction losses.
        
        Args:
            fg_pred: Foreground prediction
            bg_pred: Background prediction
            target: Target image
            fg_mask: Foreground mask (1 = foreground)
            bg_mask: Background mask (1 = background)
        
        Returns:
            Tuple of (total_loss, metrics)
        """
        # Normalize target
        if target.dtype == torch.uint8:
            target = target.float() / 255.0 - 0.5
        elif target.max() > 1.0:
            target = target / 255.0 - 0.5
        
        # Expand masks to match image channels
        if fg_mask.dim() == 4:  # (B, T, H, W)
            fg_mask = fg_mask.unsqueeze(-3)  # (B, T, 1, H, W)
            bg_mask = bg_mask.unsqueeze(-3)
        
        # Optional mask smoothing for soft edges
        if self.mask_smoothing > 0:
            fg_mask = torch.clamp(
                fg_mask + self.mask_smoothing * (1 - fg_mask), 0, 1
            )
            bg_mask = torch.clamp(
                bg_mask + self.mask_smoothing * (1 - bg_mask), 0, 1
            )
        
        # Foreground loss (only on foreground regions)
        fg_loss = ((fg_pred - target) ** 2) * fg_mask
        fg_loss = fg_loss.sum() / (fg_mask.sum() + 1e-8)
        
        # Background loss (only on background regions)
        bg_loss = ((bg_pred - target) ** 2) * bg_mask
        bg_loss = bg_loss.sum() / (bg_mask.sum() + 1e-8)
        
        # Total loss
        total_loss = self.fg_weight * fg_loss + self.bg_weight * bg_loss
        
        metrics = {
            'fg_recon_loss': fg_loss.detach(),
            'bg_recon_loss': bg_loss.detach(),
            'fg_bg_total_loss': total_loss.detach(),
        }
        
        return total_loss, metrics

