"""
Common neural network components for DreamerV3.
Includes MLP, GRU cells, normalization layers, and activation functions.
"""

import math
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
        "swish": nn.SiLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "none": nn.Identity(),
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name.lower()]


def get_norm(name: str, dim: int) -> nn.Module:
    """Get normalization layer by name."""
    norms = {
        "layer": nn.LayerNorm(dim),
        "batch": nn.BatchNorm1d(dim),
        "none": nn.Identity(),
    }
    if name.lower() not in norms:
        raise ValueError(f"Unknown normalization: {name}")
    return norms[name.lower()]


class LayerNorm(nn.Module):
    """Layer normalization with optional learnable parameters."""
    
    def __init__(
        self, 
        dim: int, 
        eps: float = 1e-5, 
        elementwise_affine: bool = True
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable layers, activation, and normalization.
    Used throughout DreamerV3 for heads and feature processing.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 5,
        activation: str = "silu",
        norm: str = "layer",
        output_activation: str = "none",
        output_norm: str = "none",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        layers = []
        current_dim = in_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(get_norm(norm, hidden_dim))
            layers.append(get_activation(activation))
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, out_dim))
        if output_norm != "none":
            layers.append(get_norm(output_norm, out_dim))
        if output_activation != "none":
            layers.append(get_activation(output_activation))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GRUCell(nn.Module):
    """
    GRU cell with layer normalization for stable training.
    Used in the RSSM deterministic path.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        norm: str = "layer",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Gates
        self.linear_ih = nn.Linear(input_dim, 3 * hidden_dim, bias=False)
        self.linear_hh = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        
        # Layer normalization for gates
        if norm == "layer":
            self.norm_ih = nn.LayerNorm(3 * hidden_dim)
            self.norm_hh = nn.LayerNorm(3 * hidden_dim)
        else:
            self.norm_ih = nn.Identity()
            self.norm_hh = nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize weights orthogonally
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.orthogonal_(param)
    
    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch, input_dim)
            h: Hidden state of shape (batch, hidden_dim)
        
        Returns:
            New hidden state of shape (batch, hidden_dim)
        """
        # Compute gates
        gates_ih = self.norm_ih(self.linear_ih(x))
        gates_hh = self.norm_hh(self.linear_hh(h))
        
        # Split into reset, update, and new gates
        r_ih, z_ih, n_ih = gates_ih.chunk(3, dim=-1)
        r_hh, z_hh, n_hh = gates_hh.chunk(3, dim=-1)
        
        # Apply gates
        r = torch.sigmoid(r_ih + r_hh)
        z = torch.sigmoid(z_ih + z_hh)
        n = torch.tanh(n_ih + r * n_hh)
        
        # Compute new hidden state
        h_new = (1 - z) * n + z * h
        
        return h_new


class Conv2dBlock(nn.Module):
    """Convolutional block with normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        activation: str = "silu",
        norm: str = "layer",
        transpose: bool = False,
    ):
        super().__init__()
        
        if transpose:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, 
                kernel_size, stride, padding,
                output_padding=stride - 1 if stride > 1 else 0
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, 
                kernel_size, stride, padding
            )
        
        if norm == "layer":
            self.norm = nn.GroupNorm(1, out_channels)  # Equivalent to LayerNorm for conv
        elif norm == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
        
        self.act = get_activation(activation)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SymlogDist(nn.Module):
    """
    Symlog distribution for stable prediction of unbounded values.
    Used for reward and value prediction in DreamerV3.
    """
    
    def __init__(self, num_bins: int = 255, low: float = -20.0, high: float = 20.0):
        super().__init__()
        self.num_bins = num_bins
        self.low = low
        self.high = high
        
        # Create bin edges in symlog space
        edges = torch.linspace(low, high, num_bins + 1)
        self.register_buffer("edges", edges)
        self.register_buffer("centers", (edges[:-1] + edges[1:]) / 2)
    
    def forward(self, logits: Tensor) -> Tensor:
        """Convert logits to expected value using two-hot encoding."""
        # Ensure centers are on the same device as input
        centers = self.centers.to(logits.device)
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Compute expected value in symlog space
        symlog_value = (probs * centers).sum(dim=-1)
        
        # Convert back to original space
        return symexp(symlog_value)
    
    def log_prob(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute log probability of target under the distribution."""
        # Ensure edges are on the same device as input
        device = logits.device
        edges = self.edges.to(device)
        
        # Convert target to symlog space
        symlog_target = symlog(target)
        
        # Clamp to valid range
        symlog_target = symlog_target.clamp(self.low, self.high)
        
        # Find which bins the target falls into (two-hot encoding)
        below = (edges[:-1] <= symlog_target.unsqueeze(-1)).sum(dim=-1) - 1
        below = below.clamp(0, self.num_bins - 2)
        above = below + 1
        
        # Compute interpolation weights
        below_edge = edges[below]
        above_edge = edges[above]
        weight_above = (symlog_target - below_edge) / (above_edge - below_edge + 1e-8)
        weight_below = 1 - weight_above
        
        # Create two-hot target
        target_probs = torch.zeros_like(logits)
        target_probs.scatter_(-1, below.unsqueeze(-1), weight_below.unsqueeze(-1))
        target_probs.scatter_(-1, above.unsqueeze(-1), weight_above.unsqueeze(-1))
        
        # Compute cross-entropy
        log_probs = F.log_softmax(logits, dim=-1)
        return (target_probs * log_probs).sum(dim=-1)


def symlog(x: Tensor) -> Tensor:
    """Symmetric logarithm for stable gradients with large values."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    """Inverse of symlog."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class OneHotDist:
    """
    Categorical distribution with straight-through gradients.
    Used for discrete action spaces.
    """
    
    def __init__(self, logits: Tensor, unimix_ratio: float = 0.01):
        self.logits = logits
        self.unimix_ratio = unimix_ratio
        
        # Mix with uniform distribution for exploration
        probs = F.softmax(logits, dim=-1)
        uniform = torch.ones_like(probs) / probs.shape[-1]
        self.probs = (1 - unimix_ratio) * probs + unimix_ratio * uniform
    
    def sample(self) -> Tensor:
        """Sample with straight-through gradients."""
        indices = torch.multinomial(self.probs.view(-1, self.probs.shape[-1]), 1)
        indices = indices.view(*self.probs.shape[:-1])
        one_hot = F.one_hot(indices, self.probs.shape[-1]).float()
        # Straight-through gradient
        return one_hot + self.probs - self.probs.detach()
    
    def mode(self) -> Tensor:
        """Return mode of distribution."""
        indices = self.probs.argmax(dim=-1)
        one_hot = F.one_hot(indices, self.probs.shape[-1]).float()
        return one_hot + self.probs - self.probs.detach()
    
    def entropy(self) -> Tensor:
        """Compute entropy of distribution."""
        return -(self.probs * torch.log(self.probs + 1e-8)).sum(dim=-1)
    
    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability of x."""
        return (x * torch.log(self.probs + 1e-8)).sum(dim=-1)


class TruncNormalDist:
    """
    Truncated normal distribution for continuous actions.
    Used in DreamerV3 for bounded continuous action spaces.
    """
    
    def __init__(
        self, 
        mean: Tensor, 
        std: Tensor, 
        low: float = -1.0, 
        high: float = 1.0
    ):
        self.mean = mean
        self.std = std
        self.low = low
        self.high = high
        
        # Create base normal distribution
        self.dist = torch.distributions.Normal(mean, std)
    
    def sample(self) -> Tensor:
        """Sample and clamp to bounds."""
        sample = self.dist.rsample()
        return sample.clamp(self.low, self.high)
    
    def mode(self) -> Tensor:
        """Return mode (mean) clamped to bounds."""
        return self.mean.clamp(self.low, self.high)
    
    def entropy(self) -> Tensor:
        """Approximate entropy of truncated normal."""
        return self.dist.entropy()
    
    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability."""
        return self.dist.log_prob(x).sum(dim=-1)

