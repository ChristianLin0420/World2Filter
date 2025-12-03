"""
Natural Video Source for Background Distractions.

Adapted from psp_camera_ready/tia/Dreamer/dmc2gym/natural_imgsource.py
"""

import glob
import os
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np


class RandomVideoSource:
    """
    Source of natural video frames for background distractions.
    Loads video files and provides sequential frames for each episode.
    """
    
    def __init__(
        self,
        shape: Tuple[int, int],
        filelist: List[str],
        total_frames: Optional[int] = None,
        grayscale: bool = False,
    ):
        """
        Args:
            shape: [height, width] of output images
            filelist: List of video file paths
            total_frames: Total number of frames to load (None = all frames)
            grayscale: Convert to grayscale
        """
        self.grayscale = grayscale
        self.total_frames = total_frames
        self.shape = shape
        self.filelist = filelist
        self._loc = 0
        
        print(f"Loading natural videos from {len(filelist)} files...")
        self.build_arr()
        self.reset()
    
    def build_arr(self):
        """Load and preprocess video frames."""
        try:
            import skvideo.io
        except ImportError:
            raise ImportError(
                "scikit-video required for natural video backgrounds. "
                "Install with: pip install scikit-video"
            )
        
        if not self.total_frames:
            # Load all frames from all videos
            self.total_frames = 0
            self.arr = None
            random.shuffle(self.filelist)
            
            for fname in self.filelist:
                print(f"  Loading: {os.path.basename(fname)}")
                try:
                    if self.grayscale:
                        frames = skvideo.io.vread(fname, outputdict={"-pix_fmt": "gray"})
                    else:
                        frames = skvideo.io.vread(fname)
                    
                    local_arr = np.zeros(
                        (frames.shape[0], self.shape[0], self.shape[1])
                        + ((3,) if not self.grayscale else (1,))
                    )
                    
                    for i in range(frames.shape[0]):
                        # cv2.resize uses (width, height) order
                        local_arr[i] = cv2.resize(
                            frames[i], (self.shape[1], self.shape[0])
                        )
                    
                    if self.arr is None:
                        self.arr = local_arr
                    else:
                        self.arr = np.concatenate([self.arr, local_arr], 0)
                    
                    self.total_frames += local_arr.shape[0]
                    print(f"    Loaded {local_arr.shape[0]} frames (total: {self.total_frames})")
                    
                except Exception as e:
                    print(f"    Warning: Failed to load {fname}: {e}")
                    continue
        
        else:
            # Load fixed number of frames from videos
            self.arr = np.zeros(
                (self.total_frames, self.shape[0], self.shape[1])
                + ((3,) if not self.grayscale else (1,))
            )
            total_frame_i = 0
            file_i = 0
            
            while total_frame_i < self.total_frames:
                if file_i % len(self.filelist) == 0:
                    random.shuffle(self.filelist)
                file_i += 1
                fname = self.filelist[file_i % len(self.filelist)]
                
                try:
                    if self.grayscale:
                        frames = skvideo.io.vread(fname, outputdict={"-pix_fmt": "gray"})
                    else:
                        frames = skvideo.io.vread(fname)
                    
                    for frame_i in range(frames.shape[0]):
                        if total_frame_i >= self.total_frames:
                            break
                        
                        if self.grayscale:
                            self.arr[total_frame_i] = cv2.resize(
                                frames[frame_i], (self.shape[1], self.shape[0])
                            )[..., None]
                        else:
                            self.arr[total_frame_i] = cv2.resize(
                                frames[frame_i], (self.shape[1], self.shape[0])
                            )
                        
                        total_frame_i += 1
                        
                        if total_frame_i % 100 == 0:
                            print(f"  Loaded {total_frame_i}/{self.total_frames} frames")
                
                except Exception as e:
                    print(f"  Warning: Failed to load {fname}: {e}")
                    continue
        
        print(f"Natural video loading complete: {self.total_frames} total frames")
    
    def reset(self):
        """Reset to random starting position for new episode."""
        self._loc = np.random.randint(0, self.total_frames)
    
    def get_image(self):
        """Get next frame sequentially."""
        img = self.arr[self._loc % self.total_frames]
        self._loc += 1
        return img


def load_natural_videos(
    shape: Tuple[int, int],
    video_pattern: str,
    total_frames: int = 1000,
    grayscale: bool = True,
) -> RandomVideoSource:
    """
    Load natural videos from a file pattern.
    
    Args:
        shape: [height, width] for output
        video_pattern: Glob pattern for video files (e.g., "/path/to/videos/*.mp4")
        total_frames: Number of frames to load
        grayscale: Convert to grayscale
    
    Returns:
        RandomVideoSource instance
    
    Example:
        >>> source = load_natural_videos(
        ...     shape=(64, 64),
        ...     video_pattern="/path/to/kinetics/driving_car/*.mp4",
        ...     total_frames=1000,
        ... )
        >>> bg_image = source.get_image()  # Get one frame
        >>> source.reset()  # Reset for new episode
    """
    files = glob.glob(os.path.expanduser(video_pattern))
    
    if not files:
        raise ValueError(
            f"Pattern '{video_pattern}' does not match any files. "
            f"Please check the path and ensure videos are downloaded."
        )
    
    print(f"Found {len(files)} video files matching pattern")
    
    return RandomVideoSource(
        shape=shape,
        filelist=files,
        total_frames=total_frames,
        grayscale=grayscale,
    )

