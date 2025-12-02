"""
Foreground/Background Decoders for World2Filter.

Implements dual decoder heads that separately reconstruct
foreground (agent) and background regions using SAM3 masks.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.dreamer_v3.decoder import Decoder
from src.models.dreamer_v3.networks import get_activation, Conv2dBlock


class FGBGDecoder(nn.Module):
    """
    Dual decoder for foreground and background reconstruction.
    
    Uses separate decoder heads that share a common architecture
    but learn to reconstruct different regions of the observation.
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
        shared_layers: int = 2,
    ):
        """
        Args:
            feature_dim: Dimension of input features (from RSSM)
            out_channels: Number of output image channels
            channels: Number of channels at each decoder layer
            kernels: Kernel sizes
            strides: Stride values
            activation: Activation function
            norm: Normalization type
            output_dist: Output distribution type
            image_size: Target output image size
            shared_layers: Number of shared layers before branching
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.out_channels = out_channels
        self.output_dist = output_dist
        self.image_size = image_size
        
        # Calculate initial spatial size
        self.init_size = image_size
        for stride in strides:
            self.init_size = self.init_size // stride
        
        self.init_channels = channels[0]
        
        # Shared projection from features to spatial
        self.shared_project = nn.Sequential(
            nn.Linear(feature_dim, self.init_channels * self.init_size * self.init_size),
            get_activation(activation),
        )
        
        # Shared initial layers
        shared_layers_list = []
        current_channels = channels[0]
        
        for i in range(min(shared_layers, len(channels) - 1)):
            out_ch = channels[i + 1] if i + 1 < len(channels) else channels[-1]
            shared_layers_list.append(Conv2dBlock(
                current_channels, out_ch,
                kernel_size=kernels[i],
                stride=strides[i],
                padding=kernels[i] // 2 - 1 if strides[i] > 1 else kernels[i] // 2,
                activation=activation,
                norm=norm,
                transpose=True,
            ))
            current_channels = out_ch
        
        self.shared_layers = nn.Sequential(*shared_layers_list)
        self.branch_channels = current_channels
        
        # Foreground-specific decoder head
        self.fg_decoder = self._build_branch(
            current_channels,
            out_channels,
            channels[shared_layers:] + [out_channels],
            kernels[shared_layers:],
            strides[shared_layers:],
            activation,
            norm,
        )
        
        # Background-specific decoder head
        self.bg_decoder = self._build_branch(
            current_channels,
            out_channels,
            channels[shared_layers:] + [out_channels],
            kernels[shared_layers:],
            strides[shared_layers:],
            activation,
            norm,
        )
        
        self._init_weights()
    
    def _build_branch(
        self,
        in_channels: int,
        out_channels: int,
        channels: List[int],
        kernels: List[int],
        strides: List[int],
        activation: str,
        norm: str,
    ) -> nn.Sequential:
        """Build a decoder branch."""
        layers = []
        current_channels = in_channels
        
        for i, (out_ch, kernel, stride) in enumerate(zip(channels[:-1], kernels, strides)):
            padding = kernel // 2 - 1 if stride > 1 else kernel // 2
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
        
        # Final output layer (no norm/activation)
        if len(kernels) > 0:
            layers.append(nn.ConvTranspose2d(
                current_channels, out_channels,
                kernel_size=kernels[-1] if len(kernels) > len(channels) - 1 else 4,
                stride=strides[-1] if len(strides) > len(channels) - 1 else 2,
                padding=1,
                output_padding=1 if strides[-1] > 1 else 0,
            ))
        else:
            # If no layers to add, just output projection
            layers.append(nn.Conv2d(current_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
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
    
    def forward(
        self,
        features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Decode features to foreground and background images.
        
        Args:
            features: Latent features (batch, feature_dim) or (batch, time, feature_dim)
        
        Returns:
            Tuple of (fg_image, bg_image)
        """
        # Handle sequence dimension
        if features.dim() == 3:
            batch, time, feat = features.shape
            features = features.view(batch * time, feat)
            fg_image, bg_image = self._decode(features)
            c, h, w = fg_image.shape[-3:]
            fg_image = fg_image.view(batch, time, c, h, w)
            bg_image = bg_image.view(batch, time, c, h, w)
        else:
            fg_image, bg_image = self._decode(features)
        
        return fg_image, bg_image
    
    def _decode(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """Decode a batch of features."""
        # Shared projection
        x = self.shared_project(features)
        x = x.view(-1, self.init_channels, self.init_size, self.init_size)
        
        # Shared layers
        x = self.shared_layers(x)
        
        # Branch-specific decoding
        fg_image = self.fg_decoder(x)
        bg_image = self.bg_decoder(x)
        
        # Ensure correct output size (interpolate if needed)
        if fg_image.shape[-1] != self.image_size or fg_image.shape[-2] != self.image_size:
            fg_image = F.interpolate(fg_image, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
            bg_image = F.interpolate(bg_image, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        return fg_image, bg_image
    
    def loss(
        self,
        features: Tensor,
        target: Tensor,
        fg_mask: Tensor,
        bg_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute reconstruction loss with masks.
        
        Args:
            features: Latent features
            target: Target images
            fg_mask: Foreground mask (1 = foreground)
            bg_mask: Background mask (1 = background)
        
        Returns:
            Tuple of (total_loss, fg_loss, bg_loss, (fg_image, bg_image))
        """
        fg_image, bg_image = self.forward(features)
        
        # Normalize target
        if target.dtype == torch.uint8:
            target = target.float() / 255.0 - 0.5
        elif target.max() > 1.0:
            target = target / 255.0 - 0.5
        
        # Expand masks for channel dimension
        if fg_mask.dim() == 4:  # (B, T, H, W)
            fg_mask = fg_mask.unsqueeze(-3)  # (B, T, 1, H, W)
            bg_mask = bg_mask.unsqueeze(-3)
        elif fg_mask.dim() == 3:  # (B, H, W)
            fg_mask = fg_mask.unsqueeze(-3)  # (B, 1, H, W)
            bg_mask = bg_mask.unsqueeze(-3)
        
        # Foreground loss (weighted by foreground mask)
        fg_error = (fg_image - target) ** 2
        fg_loss = (fg_error * fg_mask).sum() / (fg_mask.sum() + 1e-8)
        
        # Background loss (weighted by background mask)
        bg_error = (bg_image - target) ** 2
        bg_loss = (bg_error * bg_mask).sum() / (bg_mask.sum() + 1e-8)
        
        # Total loss
        total_loss = fg_loss + bg_loss
        
        return total_loss, fg_loss, bg_loss, (fg_image, bg_image)
    
    def sample(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Sample/generate images from features.
        
        Returns images in [0, 1] range.
        """
        fg_image, bg_image = self.forward(features)
        
        if self.output_dist == "binary":
            fg_image = torch.sigmoid(fg_image)
            bg_image = torch.sigmoid(bg_image)
        else:
            fg_image = (fg_image + 0.5).clamp(0, 1)
            bg_image = (bg_image + 0.5).clamp(0, 1)
        
        return fg_image, bg_image
    
    def compose(
        self,
        fg_image: Tensor,
        bg_image: Tensor,
        fg_mask: Tensor,
    ) -> Tensor:
        """
        Compose foreground and background into full image.
        
        Args:
            fg_image: Foreground reconstruction
            bg_image: Background reconstruction
            fg_mask: Foreground mask
        
        Returns:
            Composed image
        """
        # Expand mask for channels
        if fg_mask.dim() == fg_image.dim() - 1:
            fg_mask = fg_mask.unsqueeze(-3)
        
        # Compose: fg where mask is 1, bg where mask is 0
        composed = fg_image * fg_mask + bg_image * (1 - fg_mask)
        
        return composed


class IndependentFGBGDecoder(nn.Module):
    """
    Completely independent foreground and background decoders.
    
    No shared layers - each decoder learns its own representation.
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
        super().__init__()
        
        self.fg_decoder = Decoder(
            feature_dim=feature_dim,
            out_channels=out_channels,
            channels=channels,
            kernels=kernels,
            strides=strides,
            activation=activation,
            norm=norm,
            output_dist=output_dist,
            image_size=image_size,
        )
        
        self.bg_decoder = Decoder(
            feature_dim=feature_dim,
            out_channels=out_channels,
            channels=channels,
            kernels=kernels,
            strides=strides,
            activation=activation,
            norm=norm,
            output_dist=output_dist,
            image_size=image_size,
        )
    
    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """Decode to foreground and background."""
        fg_image = self.fg_decoder(features)
        bg_image = self.bg_decoder(features)
        return fg_image, bg_image
    
    def loss(
        self,
        features: Tensor,
        target: Tensor,
        fg_mask: Tensor,
        bg_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tuple[Tensor, Tensor]]:
        """Compute masked reconstruction loss."""
        fg_image, bg_image = self.forward(features)
        
        # Normalize target
        if target.dtype == torch.uint8:
            target = target.float() / 255.0 - 0.5
        elif target.max() > 1.0:
            target = target / 255.0 - 0.5
        
        # Expand masks
        if fg_mask.dim() < target.dim():
            fg_mask = fg_mask.unsqueeze(-3)
            bg_mask = bg_mask.unsqueeze(-3)
        
        # Losses
        fg_loss = ((fg_image - target) ** 2 * fg_mask).sum() / (fg_mask.sum() + 1e-8)
        bg_loss = ((bg_image - target) ** 2 * bg_mask).sum() / (bg_mask.sum() + 1e-8)
        
        total_loss = fg_loss + bg_loss
        
        return total_loss, fg_loss, bg_loss, (fg_image, bg_image)

