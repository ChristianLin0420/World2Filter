"""
CNN Encoder for DreamerV3.

Encodes visual observations into latent representations for the RSSM.
Uses a multi-layer CNN with normalization and SiLU activation.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.dreamer_v3.networks import get_activation, Conv2dBlock


class Encoder(nn.Module):
    """
    CNN encoder for visual observations.
    
    Converts (B, C, H, W) images to (B, embed_dim) latent vectors.
    Uses strided convolutions for downsampling with layer normalization.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [96, 192, 384, 768],
        kernels: List[int] = [4, 4, 4, 4],
        strides: List[int] = [2, 2, 2, 2],
        activation: str = "silu",
        norm: str = "layer",
        image_size: int = 64,
    ):
        """
        Args:
            in_channels: Number of input image channels
            channels: Number of channels at each layer
            kernels: Kernel size at each layer
            strides: Stride at each layer
            activation: Activation function name
            norm: Normalization type ('layer', 'batch', 'none')
            image_size: Input image size (assumes square images)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.channels = channels
        self.image_size = image_size
        
        # Build convolutional layers
        layers = []
        current_channels = in_channels
        current_size = image_size
        
        for i, (out_ch, kernel, stride) in enumerate(zip(channels, kernels, strides)):
            padding = kernel // 2 - 1 if stride > 1 else kernel // 2
            
            layers.append(Conv2dBlock(
                current_channels, out_ch,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                activation=activation,
                norm=norm,
                transpose=False,
            ))
            
            current_channels = out_ch
            current_size = (current_size - kernel + 2 * padding) // stride + 1
        
        self.conv = nn.Sequential(*layers)
        
        # Calculate output dimension
        self.output_size = current_size
        self.output_channels = channels[-1]
        self.embed_dim = self.output_channels * self.output_size * self.output_size
        
        # Optional projection to fixed embedding dimension
        self.flatten = nn.Flatten()
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, obs: Tensor) -> Tensor:
        """
        Encode observations.
        
        Args:
            obs: Observations of shape (batch, channels, height, width)
                 or (batch, time, channels, height, width)
        
        Returns:
            Encoded features of shape (batch, embed_dim)
            or (batch, time, embed_dim)
        """
        # Handle sequence dimension
        if obs.dim() == 5:
            batch, time, c, h, w = obs.shape
            obs = obs.view(batch * time, c, h, w)
            features = self._encode(obs)
            features = features.view(batch, time, -1)
        else:
            features = self._encode(obs)
        
        return features
    
    def _encode(self, obs: Tensor) -> Tensor:
        """Encode a batch of observations."""
        # Normalize observations to [-0.5, 0.5]
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0 - 0.5
        elif obs.max() > 1.0:
            obs = obs / 255.0 - 0.5
        
        # Apply convolutional layers
        features = self.conv(obs)
        
        # Flatten to vector
        features = self.flatten(features)
        
        return features
    
    def get_output_shape(self) -> Tuple[int, int, int]:
        """Get the output shape before flattening."""
        return (self.output_channels, self.output_size, self.output_size)


class MultiModalEncoder(nn.Module):
    """
    Encoder for multiple input modalities (e.g., image + proprioception).
    """
    
    def __init__(
        self,
        image_encoder: Encoder,
        proprio_dim: int = 0,
        hidden_dim: int = 1024,
        activation: str = "silu",
    ):
        super().__init__()
        
        self.image_encoder = image_encoder
        self.proprio_dim = proprio_dim
        
        # Combine image and proprioceptive features
        if proprio_dim > 0:
            self.proprio_net = nn.Sequential(
                nn.Linear(proprio_dim, hidden_dim),
                get_activation(activation),
            )
            self.combine = nn.Sequential(
                nn.Linear(image_encoder.embed_dim + hidden_dim, hidden_dim),
                get_activation(activation),
            )
            self.embed_dim = hidden_dim
        else:
            self.embed_dim = image_encoder.embed_dim
    
    def forward(
        self,
        image: Tensor,
        proprio: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode multimodal observations.
        
        Args:
            image: Image observations
            proprio: Proprioceptive observations (optional)
        
        Returns:
            Combined encoded features
        """
        image_feat = self.image_encoder(image)
        
        if self.proprio_dim > 0 and proprio is not None:
            proprio_feat = self.proprio_net(proprio)
            features = torch.cat([image_feat, proprio_feat], dim=-1)
            features = self.combine(features)
        else:
            features = image_feat
        
        return features

