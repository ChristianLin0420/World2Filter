"""
World2Filter World Model.

Extends DreamerV3 with foreground/background segmentation for
disentangled visual representation learning.
"""

from typing import Dict, List, Optional, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.dreamer_v3.rssm import RSSM, RSSMState
from src.models.dreamer_v3.encoder import Encoder
from src.models.dreamer_v3.networks import MLP, SymlogDist
from src.models.world2filter.fg_bg_decoder import FGBGDecoder, IndependentFGBGDecoder


class FGBGWorldModelOutput(NamedTuple):
    """Container for World2Filter model outputs."""
    prior: RSSMState
    posterior: RSSMState
    features: Tensor
    fg_pred: Tensor
    bg_pred: Tensor
    reward_logits: Tensor
    continue_logits: Tensor


class FGBGWorldModel(nn.Module):
    """
    World2Filter World Model.
    
    Extends DreamerV3 with:
    - Dual decoder heads for foreground and background
    - Mask-guided reconstruction loss
    - Same RSSM latent space for dynamics
    
    The key insight is that by separating FG/BG reconstruction,
    the model can learn to disentangle agent-centric information
    from background/distractor information in the latent space.
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, int, int] = (3, 64, 64),
        action_dim: int = 6,
        # RSSM config
        deter_size: int = 4096,
        stoch_size: int = 32,
        classes: int = 32,
        hidden_size: int = 1024,
        gru_layers: int = 1,
        unimix_ratio: float = 0.01,
        # Encoder config
        encoder_channels: List[int] = [96, 192, 384, 768],
        encoder_kernels: List[int] = [4, 4, 4, 4],
        encoder_strides: List[int] = [2, 2, 2, 2],
        # Decoder config
        decoder_channels: List[int] = [768, 384, 192, 96],
        decoder_kernels: List[int] = [4, 4, 4, 4],
        decoder_strides: List[int] = [2, 2, 2, 2],
        output_dist: str = "mse",
        shared_decoder_layers: int = 2,
        # Head config
        reward_layers: int = 5,
        reward_units: int = 1024,
        reward_bins: int = 255,
        continue_layers: int = 5,
        continue_units: int = 1024,
        # General
        activation: str = "silu",
        norm: str = "layer",
        # FG/BG specific
        independent_decoders: bool = False,
    ):
        """
        Args:
            obs_shape: Observation shape (C, H, W)
            action_dim: Action dimension
            deter_size: Deterministic state size
            stoch_size: Stochastic state size (per class)
            classes: Number of stochastic classes
            hidden_size: Hidden layer size
            gru_layers: Number of GRU layers
            unimix_ratio: Uniform mixing ratio for exploration
            encoder_channels: Encoder channel sizes
            encoder_kernels: Encoder kernel sizes
            encoder_strides: Encoder stride sizes
            decoder_channels: Decoder channel sizes
            decoder_kernels: Decoder kernel sizes
            decoder_strides: Decoder stride sizes
            output_dist: Output distribution type
            shared_decoder_layers: Number of shared decoder layers
            reward_layers: Reward head layers
            reward_units: Reward head units
            reward_bins: Reward distribution bins
            continue_layers: Continue head layers
            continue_units: Continue head units
            activation: Activation function
            norm: Normalization type
            independent_decoders: Use completely independent FG/BG decoders
        """
        super().__init__()
        
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.classes = classes
        
        # Build encoder
        self.encoder = Encoder(
            in_channels=obs_shape[0],
            channels=encoder_channels,
            kernels=encoder_kernels,
            strides=encoder_strides,
            activation=activation,
            norm=norm,
            image_size=obs_shape[1],
        )
        
        # Build RSSM
        self.rssm = RSSM(
            deter_size=deter_size,
            stoch_size=stoch_size,
            classes=classes,
            hidden_size=hidden_size,
            action_dim=action_dim,
            embed_dim=self.encoder.embed_dim,
            gru_layers=gru_layers,
            unimix_ratio=unimix_ratio,
            activation=activation,
            norm=norm,
        )
        
        # Feature dimension
        self.feature_dim = self.rssm.feature_dim
        
        # Build FG/BG decoder
        if independent_decoders:
            self.fg_bg_decoder = IndependentFGBGDecoder(
                feature_dim=self.feature_dim,
                out_channels=obs_shape[0],
                channels=decoder_channels,
                kernels=decoder_kernels,
                strides=decoder_strides,
                activation=activation,
                norm=norm,
                output_dist=output_dist,
                image_size=obs_shape[1],
            )
        else:
            self.fg_bg_decoder = FGBGDecoder(
                feature_dim=self.feature_dim,
                out_channels=obs_shape[0],
                channels=decoder_channels,
                kernels=decoder_kernels,
                strides=decoder_strides,
                activation=activation,
                norm=norm,
                output_dist=output_dist,
                image_size=obs_shape[1],
                shared_layers=shared_decoder_layers,
            )
        
        # Reward head
        self.reward_head = MLP(
            self.feature_dim,
            reward_bins,
            hidden_dim=reward_units,
            num_layers=reward_layers,
            activation=activation,
            norm=norm,
        )
        self.reward_dist = SymlogDist(reward_bins)
        
        # Continue head
        self.continue_head = MLP(
            self.feature_dim,
            1,
            hidden_dim=continue_units,
            num_layers=continue_layers,
            activation=activation,
            norm=norm,
        )
    
    def forward(
        self,
        obs: Tensor,
        action: Tensor,
        is_first: Tensor,
        state: Optional[RSSMState] = None,
    ) -> FGBGWorldModelOutput:
        """
        Process a sequence of observations.
        
        Args:
            obs: Observations (batch, time, C, H, W)
            action: Actions (batch, time, action_dim)
            is_first: Episode start flags (batch, time)
            state: Previous RSSM state
        
        Returns:
            FGBGWorldModelOutput containing predictions
        """
        # Encode observations
        embed = self.encoder(obs)
        
        # Get RSSM states
        prior, posterior = self.rssm.observe(embed, action, is_first, state)
        
        # Get features from posterior
        features = self.rssm.get_features(posterior)
        
        # Decode to FG/BG
        fg_pred, bg_pred = self.fg_bg_decoder(features)
        
        # Predict reward and continue
        reward_logits = self.reward_head(features)
        continue_logits = self.continue_head(features)
        
        return FGBGWorldModelOutput(
            prior=prior,
            posterior=posterior,
            features=features,
            fg_pred=fg_pred,
            bg_pred=bg_pred,
            reward_logits=reward_logits,
            continue_logits=continue_logits,
        )
    
    def imagine(
        self,
        action: Tensor,
        start_state: RSSMState,
    ) -> Tuple[RSSMState, Tensor]:
        """
        Imagine future states given actions.
        
        Args:
            action: Actions to imagine (batch, horizon, action_dim)
            start_state: Initial state
        
        Returns:
            Tuple of (imagined_states, features)
        """
        states = self.rssm.imagine(action, start_state)
        features = self.rssm.get_features(states)
        return states, features
    
    def imagine_with_policy(
        self,
        start_features: Tensor,
        start_state: RSSMState,
        actor: nn.Module,
        horizon: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Imagine future trajectories using a policy.
        
        Args:
            start_features: Initial features
            start_state: Initial RSSM state
            actor: Policy network
            horizon: Imagination horizon
        
        Returns:
            Tuple of (features, actions, rewards, continues)
        """
        features_list = [start_features]
        actions_list = []
        rewards_list = []
        continues_list = []
        
        state = start_state
        features = start_features
        
        for _ in range(horizon):
            # Get action from policy
            action, _ = actor(features, deterministic=False)
            actions_list.append(action)
            
            # Imagine next state
            state = self.rssm.img_step(state.deter, state.stoch, action)
            features = self.rssm.get_features(state)
            features_list.append(features)
            
            # Predict reward and continue
            reward_logits = self.reward_head(features)
            reward = self.reward_dist(reward_logits)
            rewards_list.append(reward)
            
            continue_logits = self.continue_head(features)
            continue_prob = torch.sigmoid(continue_logits).squeeze(-1)
            continues_list.append(continue_prob)
        
        # Stack along horizon dimension
        features = torch.stack(features_list[:-1], dim=1)
        actions = torch.stack(actions_list, dim=1)
        rewards = torch.stack(rewards_list, dim=1)
        continues = torch.stack(continues_list, dim=1)
        
        return features, actions, rewards, continues
    
    def get_initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> RSSMState:
        """Get initial RSSM state."""
        return self.rssm.initial_state(batch_size, device)
    
    def encode(self, obs: Tensor) -> Tensor:
        """Encode observations to embeddings."""
        return self.encoder(obs)
    
    def decode_fg_bg(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """Decode features to FG/BG images."""
        return self.fg_bg_decoder(features)
    
    def compose_image(
        self,
        fg_pred: Tensor,
        bg_pred: Tensor,
        fg_mask: Tensor,
    ) -> Tensor:
        """Compose FG and BG into full image."""
        return self.fg_bg_decoder.compose(fg_pred, bg_pred, fg_mask)
    
    def predict_reward(self, features: Tensor) -> Tensor:
        """Predict reward from features."""
        logits = self.reward_head(features)
        return self.reward_dist(logits)
    
    def predict_continue(self, features: Tensor) -> Tensor:
        """Predict continue probability from features."""
        logits = self.continue_head(features)
        return torch.sigmoid(logits).squeeze(-1)
    
    def obs_step(
        self,
        obs: Tensor,
        action: Tensor,
        is_first: Tensor,
        state: Optional[RSSMState] = None,
    ) -> Tuple[RSSMState, Tensor]:
        """
        Single observation step for online inference.
        
        Args:
            obs: Single observation (batch, C, H, W)
            action: Previous action (batch, action_dim)
            is_first: Episode start flag (batch,)
            state: Previous RSSM state
        
        Returns:
            Tuple of (new_state, features)
        """
        batch_size = obs.shape[0]
        device = obs.device
        
        # Initialize state if needed
        if state is None:
            state = self.rssm.initial_state(batch_size, device)
        
        # Reset state for new episodes
        if is_first.any():
            init_state = self.rssm.initial_state(batch_size, device)
            mask = is_first.unsqueeze(-1)
            state = RSSMState(
                deter=torch.where(mask, init_state.deter, state.deter),
                stoch=torch.where(mask, init_state.stoch, state.stoch),
                logits=torch.where(mask.unsqueeze(-1), init_state.logits, state.logits),
            )
        
        # Encode observation
        embed = self.encoder(obs)
        
        # Get posterior state
        _, posterior = self.rssm.obs_step(
            state.deter, state.stoch, action, embed
        )
        
        # Get features
        features = self.rssm.get_features(posterior)
        
        return posterior, features


def build_fg_bg_world_model(config: Dict) -> FGBGWorldModel:
    """Build World2Filter world model from configuration."""
    return FGBGWorldModel(
        obs_shape=(
            config['environment']['obs']['channels'],
            config['environment']['obs']['image_size'],
            config['environment']['obs']['image_size'],
        ),
        action_dim=config.get('action_dim', 6),
        deter_size=config['world_model']['rssm']['deter_size'],
        stoch_size=config['world_model']['rssm']['stoch_size'],
        classes=config['world_model']['rssm']['classes'],
        hidden_size=config['world_model']['rssm']['hidden_size'],
        gru_layers=config['world_model']['rssm']['gru_layers'],
        unimix_ratio=config['world_model']['rssm']['unimix_ratio'],
        encoder_channels=list(config['world_model']['encoder']['channels']),
        encoder_kernels=list(config['world_model']['encoder']['kernels']),
        encoder_strides=list(config['world_model']['encoder']['strides']),
        decoder_channels=list(config['world_model']['decoder']['channels']),
        decoder_kernels=list(config['world_model']['decoder']['kernels']),
        decoder_strides=list(config['world_model']['decoder']['strides']),
        output_dist=config['world_model']['decoder']['output_dist'],
        reward_layers=config['world_model']['reward_head']['layers'],
        reward_units=config['world_model']['reward_head']['units'],
        reward_bins=config['world_model']['reward_head']['bins'],
        continue_layers=config['world_model']['continue_head']['layers'],
        continue_units=config['world_model']['continue_head']['units'],
        activation=config['world_model']['encoder']['act'],
        norm=config['world_model']['encoder']['norm'],
    )

