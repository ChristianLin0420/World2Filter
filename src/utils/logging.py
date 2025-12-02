"""
WandB Logger for DreamerV3.

Provides comprehensive logging including:
- Scalar metrics (losses, returns, etc.)
- Image reconstructions
- Video rollouts
- Histograms of latent distributions
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False


class WandbLogger:
    """
    Weights & Biases logger for experiment tracking.
    
    Features:
    - Scalar logging (losses, metrics)
    - Image logging (reconstructions, comparisons)
    - Video logging (rollouts, dreams)
    - Histogram logging (latent distributions)
    - Model checkpointing
    """
    
    def __init__(
        self,
        project: str = "world2filter",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        log_dir: Optional[str] = None,
        enabled: bool = True,
        resume_run_id: Optional[str] = None,
    ):
        """
        Args:
            project: WandB project name
            entity: WandB entity (team/user)
            name: Run name
            config: Configuration to log
            log_dir: Directory for local logging
            enabled: Whether to enable WandB logging
            resume_run_id: WandB run ID to resume (if resuming from checkpoint)
        """
        self.enabled = enabled and WANDB_AVAILABLE
        self.log_dir = Path(log_dir) if log_dir else Path("./logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if self.enabled:
            if resume_run_id:
                # Resume existing run
                self.run = wandb.init(
                    project=project,
                    entity=entity,
                    id=resume_run_id,
                    resume="must",
                    dir=str(self.log_dir),
                )
                print(f"Resumed WandB run: {resume_run_id}")
            else:
                # Start new run
                self.run = wandb.init(
                    project=project,
                    entity=entity,
                    name=name,
                    config=config,
                    dir=str(self.log_dir),
                )
        else:
            self.run = None
            if not WANDB_AVAILABLE:
                print("Warning: wandb not installed, logging disabled")
    
    @property
    def run_id(self) -> Optional[str]:
        """Get current WandB run ID for checkpointing."""
        if self.enabled and self.run:
            return self.run.id
        return None
    
    def log_scalar(
        self,
        name: str,
        value: Union[float, int],
        step: int,
    ):
        """Log a scalar value."""
        if self.enabled:
            wandb.log({name: value}, step=step)
    
    def log_scalars(
        self,
        metrics: Dict[str, Union[float, int]],
        step: int,
    ):
        """Log multiple scalar values."""
        if self.enabled:
            wandb.log(metrics, step=step)
    
    def log_image(
        self,
        name: str,
        image: Union[np.ndarray, Tensor],
        step: int,
        caption: Optional[str] = None,
    ):
        """
        Log a single image.
        
        Args:
            name: Image name
            image: Image array (H, W, C) or (C, H, W), values in [0, 1] or [0, 255]
            step: Training step
            caption: Optional caption
        """
        if not self.enabled:
            return
        
        image = self._prepare_image(image)
        wandb.log({name: wandb.Image(image, caption=caption)}, step=step)
    
    def log_images(
        self,
        name: str,
        images: Union[np.ndarray, Tensor, List],
        step: int,
        nrow: int = 8,
        captions: Optional[List[str]] = None,
    ):
        """
        Log multiple images as a grid.
        
        Args:
            name: Image name
            images: Batch of images (N, H, W, C) or (N, C, H, W)
            step: Training step
            nrow: Number of images per row
            captions: Optional list of captions
        """
        if not self.enabled:
            return
        
        # Create grid
        grid = self._make_image_grid(images, nrow)
        
        wandb.log({name: wandb.Image(grid)}, step=step)
    
    def log_reconstruction(
        self,
        name: str,
        original: Union[np.ndarray, Tensor],
        reconstructed: Union[np.ndarray, Tensor],
        step: int,
        nrow: int = 8,
    ):
        """
        Log original and reconstructed images side by side.
        
        Args:
            name: Image name
            original: Original images
            reconstructed: Reconstructed images
            step: Training step
            nrow: Number of images per row
        """
        if not self.enabled:
            return
        
        # Prepare images
        orig = self._prepare_batch(original)
        recon = self._prepare_batch(reconstructed)
        
        # Interleave original and reconstructed
        n = min(orig.shape[0], nrow)
        orig = orig[:n]
        recon = recon[:n]
        
        # Create comparison grid
        comparison = np.stack([orig, recon], axis=1)
        comparison = comparison.reshape(-1, *orig.shape[1:])
        
        grid = self._make_image_grid(comparison, nrow=n * 2)
        
        wandb.log({name: wandb.Image(grid, caption="Top: Original, Bottom: Reconstructed")}, step=step)
    
    def log_video(
        self,
        name: str,
        video: Union[np.ndarray, Tensor],
        step: int,
        fps: int = 15,
        caption: Optional[str] = None,
    ):
        """
        Log a video.
        
        Args:
            name: Video name
            video: Video array (T, H, W, C) or (T, C, H, W)
            step: Training step
            fps: Frames per second
            caption: Optional caption
        """
        if not self.enabled:
            return
        
        video = self._prepare_video(video)
        
        wandb.log({name: wandb.Video(video, fps=fps, caption=caption)}, step=step)
    
    def log_histogram(
        self,
        name: str,
        values: Union[np.ndarray, Tensor],
        step: int,
        num_bins: int = 64,
    ):
        """
        Log a histogram.
        
        Args:
            name: Histogram name
            values: Values to histogram
            step: Training step
            num_bins: Number of bins
        """
        if not self.enabled:
            return
        
        if isinstance(values, Tensor):
            values = values.detach().cpu().numpy()
        
        values = values.flatten()
        
        wandb.log({name: wandb.Histogram(values, num_bins=num_bins)}, step=step)
    
    def log_latent_distribution(
        self,
        name: str,
        logits: Union[np.ndarray, Tensor],
        step: int,
    ):
        """
        Log latent distribution visualization.
        
        Args:
            name: Name prefix
            logits: Latent logits (batch, stoch_size, classes)
            step: Training step
        """
        if not self.enabled:
            return
        
        if isinstance(logits, Tensor):
            logits = logits.detach().cpu().numpy()
        
        # Average over batch
        probs = self._softmax(logits).mean(axis=0)  # (stoch_size, classes)
        
        # Log as heatmap
        wandb.log({
            f"{name}/distribution": wandb.plots.HeatMap(
                x_labels=list(range(probs.shape[1])),
                y_labels=list(range(probs.shape[0])),
                matrix_values=probs,
                show_text=False,
            )
        }, step=step)
        
        # Log entropy
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=-1)
        wandb.log({f"{name}/entropy": wandb.Histogram(entropy)}, step=step)
    
    def log_gradient_norms(
        self,
        model: torch.nn.Module,
        step: int,
        prefix: str = "gradients",
    ):
        """Log gradient norms for model parameters."""
        if not self.enabled:
            return
        
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[f"{prefix}/{name}"] = grad_norm
        
        if grad_norms:
            wandb.log(grad_norms, step=step)
    
    def log_world_model_outputs(
        self,
        step: int,
        original: Tensor,
        reconstructed: Tensor,
        prior_logits: Tensor,
        posterior_logits: Tensor,
        rewards_pred: Optional[Tensor] = None,
        rewards_true: Optional[Tensor] = None,
        prefix: str = "world_model",
    ):
        """
        Log comprehensive world model outputs.
        
        Args:
            step: Training step
            original: Original observations
            reconstructed: Reconstructed observations
            prior_logits: Prior latent logits
            posterior_logits: Posterior latent logits
            rewards_pred: Predicted rewards
            rewards_true: True rewards
            prefix: Logging prefix
        """
        if not self.enabled:
            return
        
        # Log reconstructions
        self.log_reconstruction(
            f"{prefix}/reconstruction",
            original, reconstructed,
            step=step,
        )
        
        # Log latent distributions
        self.log_latent_distribution(f"{prefix}/prior", prior_logits, step)
        self.log_latent_distribution(f"{prefix}/posterior", posterior_logits, step)
        
        # Log reward comparison if available
        if rewards_pred is not None and rewards_true is not None:
            if isinstance(rewards_pred, Tensor):
                rewards_pred = rewards_pred.detach().cpu().numpy().flatten()
            if isinstance(rewards_true, Tensor):
                rewards_true = rewards_true.detach().cpu().numpy().flatten()
            
            self.log_histogram(f"{prefix}/reward_pred", rewards_pred, step)
            self.log_histogram(f"{prefix}/reward_true", rewards_true, step)
    
    def log_dream_video(
        self,
        step: int,
        start_obs: Tensor,
        dreamed_obs: Tensor,
        true_obs: Optional[Tensor] = None,
        prefix: str = "dream",
    ):
        """
        Log imagined (dreamed) video rollout.
        
        Args:
            step: Training step
            start_obs: Starting observation
            dreamed_obs: Imagined observations (T, C, H, W)
            true_obs: Optional ground truth observations
            prefix: Logging prefix
        """
        if not self.enabled:
            return
        
        # Prepare dreamed video
        dreamed = self._prepare_video(dreamed_obs)
        self.log_video(f"{prefix}/imagined", dreamed, step, caption="Imagined Rollout")
        
        # Log comparison if ground truth available
        if true_obs is not None:
            true = self._prepare_video(true_obs)
            
            # Side by side comparison
            comparison = np.concatenate([true, dreamed], axis=2)  # Horizontal concat
            self.log_video(
                f"{prefix}/comparison", 
                comparison, 
                step, 
                caption="Left: True, Right: Imagined"
            )
    
    def log_segmentation(
        self,
        step: int,
        image: Tensor,
        fg_mask: Tensor,
        bg_mask: Tensor,
        fg_recon: Optional[Tensor] = None,
        bg_recon: Optional[Tensor] = None,
        prefix: str = "segmentation",
    ):
        """
        Log segmentation masks and reconstructions.
        
        Args:
            step: Training step
            image: Original image
            fg_mask: Foreground mask
            bg_mask: Background mask
            fg_recon: Foreground reconstruction
            bg_recon: Background reconstruction
            prefix: Logging prefix
        """
        if not self.enabled:
            return
        
        image = self._prepare_image(image)
        fg_mask = self._prepare_mask(fg_mask)
        bg_mask = self._prepare_mask(bg_mask)
        
        # Create overlay visualization
        fg_overlay = self._overlay_mask(image, fg_mask, color=[255, 0, 0])
        bg_overlay = self._overlay_mask(image, bg_mask, color=[0, 0, 255])
        
        wandb.log({
            f"{prefix}/original": wandb.Image(image),
            f"{prefix}/fg_mask": wandb.Image(fg_mask),
            f"{prefix}/bg_mask": wandb.Image(bg_mask),
            f"{prefix}/fg_overlay": wandb.Image(fg_overlay, caption="Foreground (red)"),
            f"{prefix}/bg_overlay": wandb.Image(bg_overlay, caption="Background (blue)"),
        }, step=step)
        
        if fg_recon is not None and bg_recon is not None:
            fg_recon = self._prepare_image(fg_recon)
            bg_recon = self._prepare_image(bg_recon)
            
            wandb.log({
                f"{prefix}/fg_reconstruction": wandb.Image(fg_recon),
                f"{prefix}/bg_reconstruction": wandb.Image(bg_recon),
            }, step=step)
    
    def finish(self):
        """Finish logging and close WandB run."""
        if self.enabled and self.run:
            wandb.finish()
    
    def _prepare_image(self, image: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Prepare image for logging."""
        if isinstance(image, Tensor):
            image = image.detach().cpu().numpy()
        
        # Handle different formats
        if image.ndim == 4:
            image = image[0]  # Take first from batch
        
        if image.shape[0] in [1, 3]:  # CHW format
            image = np.transpose(image, (1, 2, 0))
        
        # Normalize to [0, 255]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Handle grayscale
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        
        return image
    
    def _prepare_batch(self, images: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Prepare batch of images."""
        if isinstance(images, Tensor):
            images = images.detach().cpu().numpy()
        
        # Handle CHW format
        if images.shape[1] in [1, 3]:  # NCHW
            images = np.transpose(images, (0, 2, 3, 1))
        
        # Normalize to [0, 255]
        if images.max() <= 1.0:
            images = (images * 255).astype(np.uint8)
        else:
            images = images.astype(np.uint8)
        
        return images
    
    def _prepare_video(self, video: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Prepare video for logging."""
        if isinstance(video, Tensor):
            video = video.detach().cpu().numpy()
        
        # Handle CHW format
        if video.shape[1] in [1, 3]:  # TCHW
            video = np.transpose(video, (0, 2, 3, 1))
        
        # Normalize to [0, 255]
        if video.max() <= 1.0:
            video = (video * 255).astype(np.uint8)
        else:
            video = video.astype(np.uint8)
        
        # Handle grayscale
        if video.shape[-1] == 1:
            video = np.repeat(video, 3, axis=-1)
        
        return video
    
    def _prepare_mask(self, mask: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Prepare mask for visualization."""
        if isinstance(mask, Tensor):
            mask = mask.detach().cpu().numpy()
        
        # Handle different formats
        if mask.ndim == 4:
            mask = mask[0]
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        
        # Normalize to [0, 255]
        mask = (mask * 255).astype(np.uint8)
        
        return mask
    
    def _make_image_grid(
        self,
        images: Union[np.ndarray, Tensor, List],
        nrow: int,
        padding: int = 2,
    ) -> np.ndarray:
        """Create image grid."""
        if isinstance(images, Tensor):
            images = images.detach().cpu().numpy()
        elif isinstance(images, list):
            images = np.stack([self._prepare_image(img) for img in images])
        
        images = self._prepare_batch(images)
        
        n, h, w, c = images.shape
        ncol = (n + nrow - 1) // nrow
        
        # Create padded grid
        grid_h = nrow * (h + padding) - padding
        grid_w = ncol * (w + padding) - padding
        grid = np.zeros((grid_h, grid_w, c), dtype=images.dtype)
        
        for i, img in enumerate(images):
            row = i // ncol
            col = i % ncol
            y = row * (h + padding)
            x = col * (w + padding)
            grid[y:y+h, x:x+w] = img
        
        return grid
    
    def _overlay_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: List[int],
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Overlay mask on image with color."""
        overlay = image.copy()
        mask_3ch = np.stack([mask] * 3, axis=-1) / 255.0
        color_mask = np.array(color)[np.newaxis, np.newaxis, :]
        
        overlay = (1 - alpha * mask_3ch) * overlay + alpha * mask_3ch * color_mask
        
        return overlay.astype(np.uint8)
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

