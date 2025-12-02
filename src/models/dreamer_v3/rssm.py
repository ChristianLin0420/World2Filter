"""
Recurrent State Space Model (RSSM) for DreamerV3.

The RSSM is the core of the world model, maintaining both deterministic
and stochastic state representations to model environment dynamics.

Reference: "Mastering Diverse Domains through World Models" (Hafner et al., 2023)
"""

from typing import Dict, Optional, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.dreamer_v3.networks import MLP, GRUCell, get_activation, get_norm


class RSSMState(NamedTuple):
    """Container for RSSM state."""
    deter: Tensor  # Deterministic state: (batch, deter_size)
    stoch: Tensor  # Stochastic state: (batch, stoch_size * classes)
    logits: Tensor  # Logits for stochastic state: (batch, stoch_size, classes)


class RSSM(nn.Module):
    """
    Recurrent State Space Model with discrete latent states.
    
    The RSSM maintains:
    - Deterministic state (h): Captures temporal dependencies via GRU
    - Stochastic state (z): Discrete categorical latents for diversity
    
    Components:
    - Sequence model: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
    - Dynamics predictor (prior): p(z_t | h_t)
    - Representation model (posterior): q(z_t | h_t, x_t)
    """
    
    def __init__(
        self,
        deter_size: int = 4096,
        stoch_size: int = 32,
        classes: int = 32,
        hidden_size: int = 1024,
        action_dim: int = 6,
        embed_dim: int = 1024,
        gru_layers: int = 1,
        unimix_ratio: float = 0.01,
        initial: str = "learned",
        activation: str = "silu",
        norm: str = "layer",
    ):
        super().__init__()
        
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.classes = classes
        self.hidden_size = hidden_size
        self.unimix_ratio = unimix_ratio
        self.initial = initial
        
        # Stochastic state size when flattened
        self.stoch_dim = stoch_size * classes
        
        # Input projections
        self.action_proj = nn.Linear(action_dim, hidden_size)
        self.stoch_proj = nn.Linear(self.stoch_dim, hidden_size)
        
        # Sequence model (GRU)
        self.gru_input = nn.Linear(hidden_size * 2, hidden_size)
        self.gru_input_norm = get_norm(norm, hidden_size)
        self.gru = GRUCell(hidden_size, deter_size, norm=norm)
        
        # Prior network (dynamics predictor): p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_size, hidden_size),
            get_norm(norm, hidden_size),
            get_activation(activation),
            nn.Linear(hidden_size, stoch_size * classes),
        )
        
        # Posterior network (representation model): q(z_t | h_t, x_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_size + embed_dim, hidden_size),
            get_norm(norm, hidden_size),
            get_activation(activation),
            nn.Linear(hidden_size, stoch_size * classes),
        )
        
        # Initial state (learned or zeros)
        if initial == "learned":
            self.initial_deter = nn.Parameter(torch.zeros(1, deter_size))
            self.initial_stoch = nn.Parameter(torch.zeros(1, self.stoch_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        """
        Get initial RSSM state.
        
        Args:
            batch_size: Number of parallel environments
            device: Device to create tensors on
        
        Returns:
            Initial RSSM state
        """
        if self.initial == "learned":
            deter = self.initial_deter.expand(batch_size, -1)
            stoch = self.initial_stoch.expand(batch_size, -1)
        else:
            deter = torch.zeros(batch_size, self.deter_size, device=device)
            stoch = torch.zeros(batch_size, self.stoch_dim, device=device)
        
        logits = torch.zeros(batch_size, self.stoch_size, self.classes, device=device)
        
        return RSSMState(deter=deter, stoch=stoch, logits=logits)
    
    def observe(
        self,
        embed: Tensor,
        action: Tensor,
        is_first: Tensor,
        state: Optional[RSSMState] = None,
    ) -> Tuple[RSSMState, RSSMState]:
        """
        Process a sequence of observations and actions.
        
        Args:
            embed: Encoded observations (batch, time, embed_dim)
            action: Actions (batch, time, action_dim)
            is_first: Whether each timestep is first in episode (batch, time)
            state: Previous RSSM state
        
        Returns:
            Tuple of (prior_states, posterior_states) for the sequence
        """
        batch_size, seq_len = embed.shape[:2]
        device = embed.device
        
        # Initialize state if not provided
        if state is None:
            state = self.initial_state(batch_size, device)
        
        # Lists to collect states
        prior_states = []
        posterior_states = []
        
        # Process sequence
        for t in range(seq_len):
            # Reset state for new episodes
            if is_first[:, t].any():
                init_state = self.initial_state(batch_size, device)
                mask = is_first[:, t].unsqueeze(-1)
                state = RSSMState(
                    deter=torch.where(mask, init_state.deter, state.deter),
                    stoch=torch.where(mask, init_state.stoch, state.stoch),
                    logits=torch.where(mask.unsqueeze(-1), init_state.logits, state.logits),
                )
            
            # Get prior and posterior for current timestep
            prior, posterior = self.obs_step(
                state.deter,
                state.stoch,
                action[:, t],
                embed[:, t],
            )
            
            prior_states.append(prior)
            posterior_states.append(posterior)
            
            # Update state for next step
            state = posterior
        
        # Stack states along time dimension
        prior = RSSMState(
            deter=torch.stack([s.deter for s in prior_states], dim=1),
            stoch=torch.stack([s.stoch for s in prior_states], dim=1),
            logits=torch.stack([s.logits for s in prior_states], dim=1),
        )
        posterior = RSSMState(
            deter=torch.stack([s.deter for s in posterior_states], dim=1),
            stoch=torch.stack([s.stoch for s in posterior_states], dim=1),
            logits=torch.stack([s.logits for s in posterior_states], dim=1),
        )
        
        return prior, posterior
    
    def imagine(
        self,
        action: Tensor,
        state: RSSMState,
    ) -> RSSMState:
        """
        Imagine future states given actions (for imagination-based training).
        
        Args:
            action: Actions to imagine (batch, horizon, action_dim)
            state: Initial state to start imagination from
        
        Returns:
            Imagined states over the horizon
        """
        batch_size, horizon = action.shape[:2]
        
        states = []
        for t in range(horizon):
            state = self.img_step(state.deter, state.stoch, action[:, t])
            states.append(state)
        
        return RSSMState(
            deter=torch.stack([s.deter for s in states], dim=1),
            stoch=torch.stack([s.stoch for s in states], dim=1),
            logits=torch.stack([s.logits for s in states], dim=1),
        )
    
    def obs_step(
        self,
        prev_deter: Tensor,
        prev_stoch: Tensor,
        action: Tensor,
        embed: Tensor,
    ) -> Tuple[RSSMState, RSSMState]:
        """
        Single observation step: compute prior and posterior.
        
        Args:
            prev_deter: Previous deterministic state (batch, deter_size)
            prev_stoch: Previous stochastic state (batch, stoch_dim)
            action: Action taken (batch, action_dim)
            embed: Encoded observation (batch, embed_dim)
        
        Returns:
            Tuple of (prior, posterior) states
        """
        # Sequence model: update deterministic state
        deter = self._sequence_model(prev_deter, prev_stoch, action)
        
        # Prior: p(z_t | h_t)
        prior_logits = self.prior_net(deter)
        prior_logits = prior_logits.view(-1, self.stoch_size, self.classes)
        prior_stoch = self._sample_stoch(prior_logits)
        prior = RSSMState(deter=deter, stoch=prior_stoch, logits=prior_logits)
        
        # Posterior: q(z_t | h_t, x_t)
        posterior_input = torch.cat([deter, embed], dim=-1)
        posterior_logits = self.posterior_net(posterior_input)
        posterior_logits = posterior_logits.view(-1, self.stoch_size, self.classes)
        posterior_stoch = self._sample_stoch(posterior_logits)
        posterior = RSSMState(deter=deter, stoch=posterior_stoch, logits=posterior_logits)
        
        return prior, posterior
    
    def img_step(
        self,
        prev_deter: Tensor,
        prev_stoch: Tensor,
        action: Tensor,
    ) -> RSSMState:
        """
        Single imagination step: compute prior only (no observation).
        
        Args:
            prev_deter: Previous deterministic state (batch, deter_size)
            prev_stoch: Previous stochastic state (batch, stoch_dim)
            action: Action to imagine (batch, action_dim)
        
        Returns:
            Prior state
        """
        # Sequence model: update deterministic state
        deter = self._sequence_model(prev_deter, prev_stoch, action)
        
        # Prior: p(z_t | h_t)
        prior_logits = self.prior_net(deter)
        prior_logits = prior_logits.view(-1, self.stoch_size, self.classes)
        prior_stoch = self._sample_stoch(prior_logits)
        
        return RSSMState(deter=deter, stoch=prior_stoch, logits=prior_logits)
    
    def _sequence_model(
        self,
        prev_deter: Tensor,
        prev_stoch: Tensor,
        action: Tensor,
    ) -> Tensor:
        """
        Sequence model: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        """
        # Project inputs
        action_feat = self.action_proj(action)
        stoch_feat = self.stoch_proj(prev_stoch)
        
        # Combine features
        combined = torch.cat([action_feat, stoch_feat], dim=-1)
        combined = self.gru_input(combined)
        combined = self.gru_input_norm(combined)
        combined = F.silu(combined)
        
        # GRU update
        deter = self.gru(combined, prev_deter)
        
        return deter
    
    def _sample_stoch(self, logits: Tensor) -> Tensor:
        """
        Sample stochastic state from logits using straight-through gradients.
        
        Args:
            logits: Logits for categorical distribution (batch, stoch_size, classes)
        
        Returns:
            Sampled stochastic state (batch, stoch_dim)
        """
        # Mix with uniform distribution for exploration
        probs = F.softmax(logits, dim=-1)
        uniform = torch.ones_like(probs) / self.classes
        probs = (1 - self.unimix_ratio) * probs + self.unimix_ratio * uniform
        
        # Sample with straight-through gradients
        indices = torch.multinomial(
            probs.view(-1, self.classes), 
            num_samples=1
        ).view(*probs.shape[:-1])
        
        one_hot = F.one_hot(indices, self.classes).float()
        
        # Straight-through gradient: forward uses sample, backward uses soft probs
        stoch = one_hot + probs - probs.detach()
        
        # Flatten to (batch, stoch_dim)
        stoch = stoch.view(-1, self.stoch_dim)
        
        return stoch
    
    def get_features(self, state: RSSMState) -> Tensor:
        """
        Get combined features from RSSM state.
        
        Args:
            state: RSSM state
        
        Returns:
            Combined features (batch, ..., deter_size + stoch_dim)
        """
        return torch.cat([state.deter, state.stoch], dim=-1)
    
    @property
    def feature_dim(self) -> int:
        """Dimension of combined features."""
        return self.deter_size + self.stoch_dim
    
    def kl_divergence(
        self,
        prior: RSSMState,
        posterior: RSSMState,
        free_bits: float = 1.0,
        forward: bool = False,
        balance: float = 0.8,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute KL divergence between prior and posterior.
        
        Uses free bits and KL balancing as in DreamerV3.
        
        Args:
            prior: Prior state
            posterior: Posterior state
            free_bits: Minimum KL value (free nats)
            forward: If True, compute KL(prior || posterior)
            balance: Balance between optimizing prior vs posterior
        
        Returns:
            Tuple of (kl_loss, kl_value) where kl_loss includes balancing
        """
        # Get distributions
        prior_logits = prior.logits
        posterior_logits = posterior.logits
        
        prior_probs = F.softmax(prior_logits, dim=-1)
        posterior_probs = F.softmax(posterior_logits, dim=-1)
        
        # Compute KL divergence per category
        if forward:
            kl = prior_probs * (
                torch.log(prior_probs + 1e-8) - torch.log(posterior_probs + 1e-8)
            )
        else:
            kl = posterior_probs * (
                torch.log(posterior_probs + 1e-8) - torch.log(prior_probs + 1e-8)
            )
        
        # Sum over classes and stoch dimensions
        kl = kl.sum(dim=-1).sum(dim=-1)
        
        # Free bits: don't optimize if KL is below threshold
        kl_value = kl
        kl = torch.clamp(kl, min=free_bits)
        
        # KL balancing: train posterior more than prior
        # Loss for prior (detached posterior)
        sg_posterior_probs = posterior_probs.detach()
        kl_prior = posterior_probs.detach() * (
            torch.log(sg_posterior_probs + 1e-8) - torch.log(prior_probs + 1e-8)
        )
        kl_prior = kl_prior.sum(dim=-1).sum(dim=-1)
        kl_prior = torch.clamp(kl_prior, min=free_bits)
        
        # Loss for posterior (detached prior)
        sg_prior_probs = prior_probs.detach()
        kl_posterior = posterior_probs * (
            torch.log(posterior_probs + 1e-8) - torch.log(sg_prior_probs + 1e-8)
        )
        kl_posterior = kl_posterior.sum(dim=-1).sum(dim=-1)
        kl_posterior = torch.clamp(kl_posterior, min=free_bits)
        
        # Balanced loss
        kl_loss = balance * kl_prior + (1 - balance) * kl_posterior
        
        return kl_loss, kl_value

