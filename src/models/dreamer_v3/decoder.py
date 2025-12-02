"""
CNN Decoder for DreamerV3.

Decodes latent representations back to visual observations.
Uses transposed convolutions for upsampling with layer normalization.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.dreamer_v3.networks import get_activation, Conv2dBlock, symlog, symexp


class Decoder(nn.Module):
    """
    CNN decoder for visual observations.
    
    Converts (B, feature_dim) latent vectors to (B, C, H, W) images.
    Uses transposed convolutions for upsampling.
    """
    
    def __init__(
        self,
        feature_dim: int,
        out_channels: int = 3,
        channels: List[int] = [768, 384, 192, 96],
        kernels: List[int] = [4, 4, 4, 4],
        strides: List[int] = [2, 2, 2, 2],
        activation: str = "silu",
        norm: str = "layer",
        output_dist: str = "mse",
        image_size: int = 64,
    ):
        """
        Args:
            feature_dim: Dimension of input features (from RSSM)
            out_channels: Number of output image channels
            channels: Number of channels at each layer (reversed order from encoder)
            kernels: Kernel size at each layer
            strides: Stride at each layer
            activation: Activation function name
            norm: Normalization type
            output_dist: Output distribution type ('mse', 'binary')
            image_size: Target output image size
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.out_channels = out_channels
        self.output_dist = output_dist
        self.image_size = image_size
        
        # Calculate initial spatial size (after projection)
        # Reverse the encoder's downsampling
        self.init_size = image_size
        for stride in strides:
            self.init_size = self.init_size // stride
        
        self.init_channels = channels[0]
        
        # Project features to spatial representation
        self.project = nn.Sequential(
            nn.Linear(feature_dim, self.init_channels * self.init_size * self.init_size),
            get_activation(activation),
        )
        
        # Build transposed convolutional layers
        layers = []
        current_channels = channels[0]
        
        for i, (out_ch, kernel, stride) in enumerate(zip(channels[1:] + [out_channels], kernels, strides)):
            padding = kernel // 2 - 1 if stride > 1 else kernel // 2
            
            # Last layer: no norm/activation, just output
            if i == len(channels) - 1:
                layers.append(nn.ConvTranspose2d(
                    current_channels, out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    output_padding=stride - 1 if stride > 1 else 0,
                ))
            else:
                layers.append(Conv2dBlock(
                    current_channels, out_ch,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    activation=activation,
                    norm=norm,
                    transpose=True,
                ))
                current_channels = out_ch
        
        self.deconv = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features: Tensor) -> Tensor:
        """
        Decode features to images.
        
        Args:
            features: Latent features of shape (batch, feature_dim)
                     or (batch, time, feature_dim)
        
        Returns:
            Reconstructed images of shape (batch, channels, height, width)
            or (batch, time, channels, height, width)
        """
        # Handle sequence dimension
        if features.dim() == 3:
            batch, time, feat = features.shape
            features = features.view(batch * time, feat)
            images = self._decode(features)
            c, h, w = images.shape[-3:]
            images = images.view(batch, time, c, h, w)
        else:
            images = self._decode(features)
        
        return images
    
    def _decode(self, features: Tensor) -> Tensor:
        """Decode a batch of features."""
        # Project to spatial representation
        x = self.project(features)
        x = x.view(-1, self.init_channels, self.init_size, self.init_size)
        
        # Apply transposed convolutions
        x = self.deconv(x)
        
        # Ensure correct output size (interpolate if needed)
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        return x
    
    def loss(
        self,
        features: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute reconstruction loss.
        
        Args:
            features: Latent features
            target: Target images (normalized to [0, 1] or [-0.5, 0.5])
            mask: Optional mask for weighted loss
        
        Returns:
            Tuple of (loss, reconstructed images)
        """
        recon = self.forward(features)
        
        # Normalize target
        if target.dtype == torch.uint8:
            target = target.float() / 255.0 - 0.5
        elif target.max() > 1.0:
            target = target / 255.0 - 0.5
        
        # Compute loss based on output distribution type
        if self.output_dist == "mse":
            # MSE loss with symlog transformation for stability
            loss = (recon - target) ** 2
        elif self.output_dist == "binary":
            # Binary cross-entropy (assumes target in [0, 1])
            target_01 = target + 0.5  # Convert to [0, 1]
            recon_prob = torch.sigmoid(recon)
            loss = F.binary_cross_entropy(recon_prob, target_01, reduction='none')
        else:
            raise ValueError(f"Unknown output distribution: {self.output_dist}")
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask.unsqueeze(-3)  # Expand mask for channels
        
        # Average over spatial and channel dimensions
        loss = loss.mean(dim=(-3, -2, -1))
        
        return loss, recon
    
    def sample(self, features: Tensor) -> Tensor:
        """
        Sample/generate images from features.
        
        Args:
            features: Latent features
        
        Returns:
            Generated images in [0, 1] range
        """
        recon = self.forward(features)
        
        if self.output_dist == "binary":
            recon = torch.sigmoid(recon)
        else:
            # Convert from [-0.5, 0.5] to [0, 1]
            recon = recon + 0.5
        
        return recon.clamp(0, 1)


class MultiHeadDecoder(nn.Module):
    """
    Decoder with multiple output heads (e.g., image, reward, continue).
    Shares feature processing but has separate output projections.
    """
    
    def __init__(
        self,
        feature_dim: int,
        image_decoder: Decoder,
        reward_bins: int = 255,
        hidden_dim: int = 1024,
        num_layers: int = 5,
        activation: str = "silu",
        norm: str = "layer",
    ):
        super().__init__()
        
        self.image_decoder = image_decoder
        self.feature_dim = feature_dim
        
        # Reward prediction head (symlog discrete distribution)
        from src.models.dreamer_v3.networks import MLP
        self.reward_head = MLP(
            feature_dim, reward_bins,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            norm=norm,
        )
        
        # Continue prediction head (binary)
        self.continue_head = MLP(
            feature_dim, 1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            norm=norm,
        )
    
    def forward(self, features: Tensor) -> dict:
        """
        Decode features to all outputs.
        
        Args:
            features: Latent features
        
        Returns:
            Dictionary with 'image', 'reward_logits', 'continue_logits'
        """
        return {
            'image': self.image_decoder(features),
            'reward_logits': self.reward_head(features),
            'continue_logits': self.continue_head(features),
        }

