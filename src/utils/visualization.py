"""
Visualization utilities for DreamerV3.

Provides functions for creating visual outputs for debugging
and analysis of world model and agent behavior.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False


def create_reconstruction_grid(
    original: Union[np.ndarray, Tensor],
    reconstructed: Union[np.ndarray, Tensor],
    nrow: int = 8,
    padding: int = 2,
) -> np.ndarray:
    """
    Create a grid comparing original and reconstructed images.
    
    Args:
        original: Original images (N, C, H, W) or (N, H, W, C)
        reconstructed: Reconstructed images (same shape)
        nrow: Number of columns in the grid
        padding: Padding between images
    
    Returns:
        Grid image as numpy array (H, W, C)
    """
    # Convert to numpy if needed
    if isinstance(original, Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()
    
    # Ensure HWC format
    if original.shape[1] in [1, 3]:
        original = np.transpose(original, (0, 2, 3, 1))
        reconstructed = np.transpose(reconstructed, (0, 2, 3, 1))
    
    # Normalize to [0, 1]
    original = _normalize_images(original)
    reconstructed = _normalize_images(reconstructed)
    
    # Take first nrow images
    n = min(original.shape[0], nrow)
    original = original[:n]
    reconstructed = reconstructed[:n]
    
    # Create comparison (2 rows: original on top, reconstructed on bottom)
    h, w, c = original.shape[1:]
    
    grid_h = 2 * h + padding
    grid_w = n * w + (n - 1) * padding
    
    grid = np.ones((grid_h, grid_w, c), dtype=np.float32)
    
    for i in range(n):
        x = i * (w + padding)
        grid[:h, x:x+w] = original[i]
        grid[h+padding:, x:x+w] = reconstructed[i]
    
    return (grid * 255).astype(np.uint8)


def create_video_grid(
    videos: List[Union[np.ndarray, Tensor]],
    labels: Optional[List[str]] = None,
    nrow: int = 2,
    padding: int = 2,
) -> np.ndarray:
    """
    Create a grid of videos side by side.
    
    Args:
        videos: List of videos (T, C, H, W) or (T, H, W, C)
        labels: Optional labels for each video
        nrow: Number of videos per row
        padding: Padding between videos
    
    Returns:
        Combined video as numpy array (T, H, W, C)
    """
    # Convert to numpy and ensure THWC format
    processed = []
    for video in videos:
        if isinstance(video, Tensor):
            video = video.detach().cpu().numpy()
        if video.shape[1] in [1, 3]:
            video = np.transpose(video, (0, 2, 3, 1))
        video = _normalize_images(video)
        processed.append(video)
    
    # Find common length
    T = min(v.shape[0] for v in processed)
    h, w, c = processed[0].shape[1:]
    
    # Calculate grid dimensions
    ncol = min(len(processed), nrow)
    nrows = (len(processed) + ncol - 1) // ncol
    
    grid_h = nrows * h + (nrows - 1) * padding
    grid_w = ncol * w + (ncol - 1) * padding
    
    # Create grid video
    grid_video = np.ones((T, grid_h, grid_w, c), dtype=np.float32)
    
    for i, video in enumerate(processed):
        row = i // ncol
        col = i % ncol
        y = row * (h + padding)
        x = col * (w + padding)
        grid_video[:T, y:y+h, x:x+w] = video[:T]
    
    return (grid_video * 255).astype(np.uint8)


def visualize_latent_space(
    logits: Union[np.ndarray, Tensor],
    save_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Visualize the latent space distribution.
    
    Args:
        logits: Latent logits (batch, stoch_size, classes)
        save_path: Optional path to save figure
    
    Returns:
        Figure as numpy array if matplotlib available
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    if isinstance(logits, Tensor):
        logits = logits.detach().cpu().numpy()
    
    # Compute probabilities
    probs = _softmax(logits)
    mean_probs = probs.mean(axis=0)  # (stoch_size, classes)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Heatmap of probabilities
    im = axes[0].imshow(mean_probs, aspect='auto', cmap='viridis')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Stochastic Dimension')
    axes[0].set_title('Latent Distribution (Mean)')
    plt.colorbar(im, ax=axes[0])
    
    # Entropy per dimension
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=-1)
    axes[1].bar(range(len(entropy)), entropy)
    axes[1].set_xlabel('Stochastic Dimension')
    axes[1].set_ylabel('Entropy')
    axes[1].set_title('Entropy per Dimension')
    axes[1].axhline(y=np.log(mean_probs.shape[-1]), color='r', linestyle='--', label='Max Entropy')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Convert to image
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return image


def visualize_imagination(
    start_obs: Union[np.ndarray, Tensor],
    imagined_obs: Union[np.ndarray, Tensor],
    true_obs: Optional[Union[np.ndarray, Tensor]] = None,
    rewards: Optional[Union[np.ndarray, Tensor]] = None,
    save_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Visualize imagination rollout vs ground truth.
    
    Args:
        start_obs: Starting observation
        imagined_obs: Imagined observations (T, C, H, W)
        true_obs: Optional ground truth observations
        rewards: Optional reward predictions
        save_path: Optional path to save figure
    
    Returns:
        Figure as numpy array if matplotlib available
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    # Convert to numpy
    if isinstance(imagined_obs, Tensor):
        imagined_obs = imagined_obs.detach().cpu().numpy()
    if true_obs is not None and isinstance(true_obs, Tensor):
        true_obs = true_obs.detach().cpu().numpy()
    if rewards is not None and isinstance(rewards, Tensor):
        rewards = rewards.detach().cpu().numpy()
    
    T = imagined_obs.shape[0]
    n_cols = min(T, 10)
    n_rows = 2 if true_obs is not None else 1
    if rewards is not None:
        n_rows += 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2.5 * n_rows))
    
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    
    # Plot imagined observations
    for t in range(n_cols):
        idx = t * (T // n_cols) if T > n_cols else t
        img = _prepare_for_display(imagined_obs[idx])
        axes[0, t].imshow(img)
        axes[0, t].set_title(f't={idx}')
        axes[0, t].axis('off')
    axes[0, 0].set_ylabel('Imagined', fontsize=12)
    
    # Plot ground truth if available
    if true_obs is not None:
        for t in range(n_cols):
            idx = t * (T // n_cols) if T > n_cols else t
            img = _prepare_for_display(true_obs[idx])
            axes[1, t].imshow(img)
            axes[1, t].axis('off')
        axes[1, 0].set_ylabel('True', fontsize=12)
    
    # Plot rewards if available
    if rewards is not None:
        row_idx = 2 if true_obs is not None else 1
        for t in range(n_cols):
            idx = t * (T // n_cols) if T > n_cols else t
            axes[row_idx, t].bar([0], [rewards[idx]])
            axes[row_idx, t].set_ylim(-1, 1)
            axes[row_idx, t].set_xticks([])
        axes[row_idx, 0].set_ylabel('Reward', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Convert to image
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return image


def create_segmentation_overlay(
    image: Union[np.ndarray, Tensor],
    fg_mask: Union[np.ndarray, Tensor],
    bg_mask: Optional[Union[np.ndarray, Tensor]] = None,
    fg_color: Tuple[int, int, int] = (255, 0, 0),
    bg_color: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Create segmentation overlay visualization.
    
    Args:
        image: Original image (C, H, W) or (H, W, C)
        fg_mask: Foreground mask (H, W)
        bg_mask: Optional background mask
        fg_color: Color for foreground overlay
        bg_color: Color for background overlay
        alpha: Transparency for overlay
    
    Returns:
        Overlay image (H, W, C)
    """
    # Convert to numpy
    if isinstance(image, Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(fg_mask, Tensor):
        fg_mask = fg_mask.detach().cpu().numpy()
    if bg_mask is not None and isinstance(bg_mask, Tensor):
        bg_mask = bg_mask.detach().cpu().numpy()
    
    # Ensure HWC format
    if image.shape[0] in [1, 3]:
        image = np.transpose(image, (1, 2, 0))
    
    # Normalize image
    image = _normalize_images(image[np.newaxis])[0]
    image = (image * 255).astype(np.float32)
    
    # Normalize mask
    if fg_mask.max() > 1:
        fg_mask = fg_mask / 255.0
    
    # Create overlay
    overlay = image.copy()
    
    # Apply foreground mask
    fg_color = np.array(fg_color, dtype=np.float32)
    fg_mask_3ch = fg_mask[:, :, np.newaxis]
    overlay = (1 - alpha * fg_mask_3ch) * overlay + alpha * fg_mask_3ch * fg_color
    
    # Apply background mask if provided
    if bg_mask is not None:
        if bg_mask.max() > 1:
            bg_mask = bg_mask / 255.0
        bg_color = np.array(bg_color, dtype=np.float32)
        bg_mask_3ch = bg_mask[:, :, np.newaxis]
        overlay = (1 - alpha * bg_mask_3ch) * overlay + alpha * bg_mask_3ch * bg_color
    
    return overlay.astype(np.uint8)


def save_video(
    frames: Union[np.ndarray, List[np.ndarray]],
    path: str,
    fps: int = 15,
):
    """
    Save frames as video.
    
    Args:
        frames: Video frames (T, H, W, C) or list of frames
        path: Output path
        fps: Frames per second
    """
    if not IMAGEIO_AVAILABLE:
        print("Warning: imageio not available, cannot save video")
        return
    
    if isinstance(frames, list):
        frames = np.stack(frames)
    
    # Ensure uint8 format
    if frames.dtype != np.uint8:
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)
    
    imageio.mimwrite(path, frames, fps=fps)


def _normalize_images(images: np.ndarray) -> np.ndarray:
    """Normalize images to [0, 1]."""
    if images.dtype == np.uint8:
        return images.astype(np.float32) / 255.0
    elif images.max() > 1.0:
        return images / 255.0
    elif images.min() < 0:
        return (images + 0.5).clip(0, 1)
    return images.clip(0, 1)


def _prepare_for_display(image: np.ndarray) -> np.ndarray:
    """Prepare image for matplotlib display."""
    if image.shape[0] in [1, 3]:
        image = np.transpose(image, (1, 2, 0))
    image = _normalize_images(image[np.newaxis])[0]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return image


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

