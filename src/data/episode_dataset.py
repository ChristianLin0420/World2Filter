"""
Episode Dataset for DreamerV3.

Provides PyTorch-compatible dataset for loading episodes from disk
or memory for efficient batched training.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, IterableDataset

from src.data.replay_buffer import Episode


class EpisodeDataset(Dataset):
    """
    PyTorch Dataset for episodes.
    
    Loads episodes from disk or memory and provides random sequence
    sampling for training.
    """
    
    def __init__(
        self,
        episodes: Optional[List[Episode]] = None,
        directory: Optional[Union[str, Path]] = None,
        sequence_length: int = 64,
        prioritize_ends: bool = True,
    ):
        """
        Args:
            episodes: List of episodes (if loading from memory)
            directory: Directory containing episode files (if loading from disk)
            sequence_length: Length of sequences to sample
            prioritize_ends: Prioritize sampling near episode ends
        """
        self.sequence_length = sequence_length
        self.prioritize_ends = prioritize_ends
        
        if episodes is not None:
            self.episodes = episodes
        elif directory is not None:
            self.episodes = self._load_episodes(Path(directory))
        else:
            self.episodes = []
        
        # Compute sampling weights (favor longer episodes)
        self._compute_weights()
    
    def _load_episodes(self, directory: Path) -> List[Episode]:
        """Load episodes from directory."""
        episodes = []
        episode_files = sorted(directory.glob("episode_*.npz"))
        
        for ep_file in episode_files:
            data = dict(np.load(ep_file))
            episode = Episode.from_dict(data)
            episodes.append(episode)
        
        return episodes
    
    def _compute_weights(self):
        """Compute sampling weights for episodes."""
        self.lengths = [len(ep) for ep in self.episodes]
        self.total_length = sum(self.lengths)
        
        # Weight by number of valid starting positions
        self.weights = [
            max(1, l - self.sequence_length + 1) 
            for l in self.lengths
        ]
        self.total_weight = sum(self.weights)
    
    def __len__(self) -> int:
        """Return number of valid samples (approximate)."""
        return self.total_weight
    
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        Get a random sequence.
        
        Note: idx is not used directly; we sample randomly each time.
        This is because DreamerV3 uses truly random sampling, not
        epoch-based iteration.
        """
        # Sample episode weighted by valid positions
        if self.total_weight > 0:
            ep_idx = np.random.choice(
                len(self.episodes),
                p=np.array(self.weights) / self.total_weight
            )
        else:
            ep_idx = np.random.randint(len(self.episodes))
        
        episode = self.episodes[ep_idx]
        
        # Sample starting position
        max_start = max(0, len(episode) - self.sequence_length)
        
        if self.prioritize_ends and max_start > 0:
            # Exponential distribution favoring later starts
            start = int(np.random.exponential(max_start / 2))
            start = min(start, max_start)
        else:
            start = np.random.randint(0, max_start + 1)
        
        # Extract sequence
        end = start + self.sequence_length
        
        if end <= len(episode):
            obs = episode.obs[start:end]
            action = episode.action[start:end]
            reward = episode.reward[start:end]
            is_first = episode.is_first[start:end]
            is_last = episode.is_last[start:end]
            is_terminal = episode.is_terminal[start:end]
        else:
            # Pad if necessary
            obs = self._pad(episode.obs[start:], self.sequence_length)
            action = self._pad(episode.action[start:], self.sequence_length)
            reward = self._pad(episode.reward[start:], self.sequence_length, 0)
            is_first = self._pad(episode.is_first[start:], self.sequence_length, False)
            is_last = self._pad(episode.is_last[start:], self.sequence_length, True)
            is_terminal = self._pad(episode.is_terminal[start:], self.sequence_length, False)
        
        # Convert to tensors
        return {
            'obs': torch.tensor(obs, dtype=torch.float32),
            'action': torch.tensor(action, dtype=torch.float32),
            'reward': torch.tensor(reward, dtype=torch.float32),
            'is_first': torch.tensor(is_first, dtype=torch.bool),
            'is_last': torch.tensor(is_last, dtype=torch.bool),
            'is_terminal': torch.tensor(is_terminal, dtype=torch.bool),
            'cont': torch.tensor(~is_terminal, dtype=torch.float32),
        }
    
    def _pad(
        self, 
        arr: np.ndarray, 
        target_length: int,
        pad_value: Union[float, bool] = 0,
    ) -> np.ndarray:
        """Pad array to target length."""
        current_length = len(arr)
        if current_length >= target_length:
            return arr[:target_length]
        
        pad_length = target_length - current_length
        
        if arr.ndim == 1:
            padding = np.full(pad_length, pad_value, dtype=arr.dtype)
        else:
            pad_shape = (pad_length,) + arr.shape[1:]
            if isinstance(pad_value, bool):
                padding = np.full(pad_shape, pad_value, dtype=arr.dtype)
            else:
                # Repeat last frame
                padding = np.repeat(arr[-1:], pad_length, axis=0)
        
        return np.concatenate([arr, padding], axis=0)
    
    def add_episodes(self, episodes: List[Episode]):
        """Add new episodes to the dataset."""
        self.episodes.extend(episodes)
        self._compute_weights()


class StreamingEpisodeDataset(IterableDataset):
    """
    Streaming dataset that continuously samples from a replay buffer.
    
    Useful for training where new episodes are added during training.
    """
    
    def __init__(
        self,
        replay_buffer,
        batch_size: int,
        sequence_length: int,
    ):
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.sequence_length = sequence_length
    
    def __iter__(self):
        """Yield batches indefinitely."""
        while True:
            batch = self.replay_buffer.sample(
                self.batch_size,
                self.sequence_length,
            )
            yield batch


def create_dataloader(
    episodes: List[Episode],
    batch_size: int,
    sequence_length: int,
    num_workers: int = 4,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Create a DataLoader for episode-based training.
    
    Args:
        episodes: List of episodes
        batch_size: Batch size
        sequence_length: Sequence length
        num_workers: Number of data loading workers
        prefetch_factor: Prefetch factor
    
    Returns:
        DataLoader for training
    """
    dataset = EpisodeDataset(
        episodes=episodes,
        sequence_length=sequence_length,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True,
        drop_last=True,
    )

