"""
SAM3 Wrapper for Segmentation.

Provides an interface to Meta's SAM3 (Segment Anything Model 3)
for generating foreground/background masks of agents in DMControl environments.

Reference: https://github.com/facebookresearch/sam3
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.processor import Sam3Processor
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    print("Warning: SAM3 not installed. Install from https://github.com/facebookresearch/sam3")


class SAM3Wrapper:
    """
    Wrapper for SAM3 segmentation model.
    
    Uses SAM3's text-prompted segmentation to identify foreground (agent)
    and background regions in DMControl observations.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        prompt: str = "robot",
        batch_size: int = 32,
        cache_enabled: bool = True,
    ):
        """
        Args:
            device: Device to run SAM3 on
            prompt: Text prompt for identifying foreground
            batch_size: Batch size for inference
            cache_enabled: Whether to cache masks
        """
        self.device = device
        self.prompt = prompt
        self.batch_size = batch_size
        self.cache_enabled = cache_enabled
        
        self._model = None
        self._processor = None
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        
        if not SAM3_AVAILABLE:
            print("SAM3 not available. Using fallback mask generation.")
    
    def load_model(self):
        """Load SAM3 model (lazy loading)."""
        if self._model is not None:
            return
        
        if not SAM3_AVAILABLE:
            return
        
        print("Loading SAM3 model...")
        self._model = build_sam3_image_model()
        self._processor = Sam3Processor(self._model)
        
        # Move to device
        if hasattr(self._model, 'to'):
            self._model.to(self.device)
        
        print("SAM3 model loaded.")
    
    def segment(
        self,
        image: Union[np.ndarray, Tensor],
        prompt: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment foreground from background using SAM3.
        
        Args:
            image: Input image (H, W, C) or (C, H, W), uint8 or float
            prompt: Optional text prompt (defaults to self.prompt)
        
        Returns:
            Tuple of (fg_mask, bg_mask) as numpy arrays (H, W)
        """
        prompt = prompt or self.prompt
        
        # Check cache
        if self.cache_enabled:
            cache_key = self._compute_cache_key(image)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Prepare image
        image_np = self._prepare_image(image)
        
        if SAM3_AVAILABLE and self._model is not None:
            fg_mask, bg_mask = self._segment_sam3(image_np, prompt)
        else:
            fg_mask, bg_mask = self._segment_fallback(image_np)
        
        # Cache result
        if self.cache_enabled:
            self._cache[cache_key] = (fg_mask, bg_mask)
        
        return fg_mask, bg_mask
    
    def segment_batch(
        self,
        images: Union[np.ndarray, Tensor],
        prompt: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment a batch of images.
        
        Args:
            images: Batch of images (N, C, H, W) or (N, H, W, C)
            prompt: Optional text prompt
        
        Returns:
            Tuple of (fg_masks, bg_masks) as numpy arrays (N, H, W)
        """
        prompt = prompt or self.prompt
        
        # Prepare images
        if isinstance(images, Tensor):
            images = images.detach().cpu().numpy()
        
        # Ensure NHWC format
        if images.shape[1] in [1, 3]:  # NCHW
            images = np.transpose(images, (0, 2, 3, 1))
        
        # Normalize
        if images.dtype != np.uint8:
            if images.max() <= 1.0:
                images = (images * 255).astype(np.uint8)
            else:
                images = images.astype(np.uint8)
        
        fg_masks = []
        bg_masks = []
        
        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            
            for img in batch:
                fg, bg = self.segment(img, prompt)
                fg_masks.append(fg)
                bg_masks.append(bg)
        
        return np.stack(fg_masks), np.stack(bg_masks)
    
    def _segment_sam3(
        self,
        image: np.ndarray,
        prompt: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Segment using SAM3."""
        from PIL import Image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Run SAM3 inference
        inference_state = self._processor.set_image(pil_image)
        output = self._processor.set_text_prompt(
            state=inference_state,
            prompt=prompt,
        )
        
        masks = output["masks"]
        scores = output["scores"]
        
        if len(masks) > 0:
            # Take best mask
            best_idx = np.argmax(scores)
            fg_mask = masks[best_idx].astype(np.float32)
        else:
            # No detection, use empty mask
            fg_mask = np.zeros(image.shape[:2], dtype=np.float32)
        
        # Background is inverse of foreground
        bg_mask = 1.0 - fg_mask
        
        return fg_mask, bg_mask
    
    def _segment_fallback(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback segmentation when SAM3 is not available.
        
        Uses simple color-based heuristics for DMControl environments.
        The agent typically has different colors from the background.
        """
        # Convert to HSV for better color segmentation
        import cv2
        
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # DMControl agents often have distinct saturation/value
        # This is a simple heuristic - adjust thresholds as needed
        
        # Create mask based on saturation (agents often have high saturation)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        # Adaptive thresholding
        sat_thresh = np.percentile(saturation, 70)
        val_thresh = np.percentile(value, 30)
        
        # Foreground: high saturation OR non-dark areas
        fg_mask = ((saturation > sat_thresh) | (value > val_thresh)).astype(np.float32)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Gaussian blur for soft edges
        fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
        
        bg_mask = 1.0 - fg_mask
        
        return fg_mask, bg_mask
    
    def _prepare_image(self, image: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Prepare image for segmentation."""
        if isinstance(image, Tensor):
            image = image.detach().cpu().numpy()
        
        # Ensure HWC format
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))
        
        # Normalize to uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            elif image.min() < 0:
                image = ((image + 0.5) * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Ensure RGB
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        
        return image
    
    def _compute_cache_key(self, image: Union[np.ndarray, Tensor]) -> int:
        """Compute cache key for image."""
        if isinstance(image, Tensor):
            image = image.detach().cpu().numpy()
        return hash(image.tobytes())
    
    def clear_cache(self):
        """Clear the mask cache."""
        self._cache.clear()


class DummySAM3Wrapper:
    """
    Dummy wrapper for testing without SAM3.
    
    Returns random or heuristic masks for testing purposes.
    """
    
    def __init__(self, **kwargs):
        pass
    
    def load_model(self):
        pass
    
    def segment(
        self,
        image: Union[np.ndarray, Tensor],
        prompt: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return dummy masks."""
        if isinstance(image, Tensor):
            image = image.detach().cpu().numpy()
        
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            h, w = image.shape[1:3]
        else:
            h, w = image.shape[:2]
        
        # Create simple circular mask in center
        y, x = np.ogrid[:h, :w]
        center = (h // 2, w // 2)
        radius = min(h, w) // 3
        
        fg_mask = ((y - center[0]) ** 2 + (x - center[1]) ** 2 <= radius ** 2).astype(np.float32)
        bg_mask = 1.0 - fg_mask
        
        return fg_mask, bg_mask
    
    def segment_batch(
        self,
        images: Union[np.ndarray, Tensor],
        prompt: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return dummy masks for batch."""
        if isinstance(images, Tensor):
            images = images.detach().cpu().numpy()
        
        n = images.shape[0]
        fg_masks = []
        bg_masks = []
        
        for i in range(n):
            fg, bg = self.segment(images[i], prompt)
            fg_masks.append(fg)
            bg_masks.append(bg)
        
        return np.stack(fg_masks), np.stack(bg_masks)
    
    def clear_cache(self):
        pass


def create_sam3_wrapper(
    use_sam3: bool = True,
    **kwargs,
) -> Union[SAM3Wrapper, DummySAM3Wrapper]:
    """
    Factory function to create SAM3 wrapper.
    
    Args:
        use_sam3: Whether to use real SAM3 or dummy
        **kwargs: Arguments for SAM3Wrapper
    
    Returns:
        SAM3 wrapper instance
    """
    if use_sam3 and SAM3_AVAILABLE:
        wrapper = SAM3Wrapper(**kwargs)
        wrapper.load_model()
        return wrapper
    else:
        return DummySAM3Wrapper(**kwargs)

