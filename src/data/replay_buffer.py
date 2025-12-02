"""
Replay Buffer for DreamerV3.

Stores episodes of experience and provides efficient sampling
for world model and actor-critic training.
"""

import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor


@dataclass
class Episode:
    """Container for a single episode of experience."""
    obs: np.ndarray  # (time, C, H, W)
    action: np.ndarray  # (time, action_dim)
    reward: np.ndarray  # (time,)
    is_first: np.ndarray  # (time,) - True at episode start
    is_last: np.ndarray  # (time,) - True at episode end
    is_terminal: np.ndarray  # (time,) - True if terminated (not truncated)
    
    # Optional: segmentation masks for World2Filter
    fg_mask: Optional[np.ndarray] = None  # (time, H, W)
    bg_mask: Optional[np.ndarray] = None  # (time, H, W)
    
    def __len__(self) -> int:
        return len(self.obs)
    
    def __getitem__(self, idx: Union[int, slice]) -> 'Episode':
        """Slice episode to get sub-episode."""
        return Episode(
            obs=self.obs[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            is_first=self.is_first[idx],
            is_last=self.is_last[idx],
            is_terminal=self.is_terminal[idx],
            fg_mask=self.fg_mask[idx] if self.fg_mask is not None else None,
            bg_mask=self.bg_mask[idx] if self.bg_mask is not None else None,
        )
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary for saving."""
        data = {
            'obs': self.obs,
            'action': self.action,
            'reward': self.reward,
            'is_first': self.is_first,
            'is_last': self.is_last,
            'is_terminal': self.is_terminal,
        }
        if self.fg_mask is not None:
            data['fg_mask'] = self.fg_mask
        if self.bg_mask is not None:
            data['bg_mask'] = self.bg_mask
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, np.ndarray]) -> 'Episode':
        """Create episode from dictionary."""
        return cls(
            obs=data['obs'],
            action=data['action'],
            reward=data['reward'],
            is_first=data['is_first'],
            is_last=data['is_last'],
            is_terminal=data['is_terminal'],
            fg_mask=data.get('fg_mask'),
            bg_mask=data.get('bg_mask'),
        )


class ReplayBuffer:
    """
    Experience replay buffer storing complete episodes.
    
    Features:
    - Episode-based storage for temporal learning
    - Efficient random sequence sampling
    - Optional prioritization of recent experience
    - Support for saving/loading to disk
    """
    
    def __init__(
        self,
        capacity: int = 1_000_000,
        min_length: int = 1,
        max_length: int = 0,
        prioritize_ends: bool = True,
        directory: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            capacity: Maximum number of timesteps to store
            min_length: Minimum episode length to store
            max_length: Maximum episode length (0 = no limit)
            prioritize_ends: If True, prioritize sampling near episode ends
            directory: Directory for saving/loading episodes
        """
        self.capacity = capacity
        self.min_length = min_length
        self.max_length = max_length
        self.prioritize_ends = prioritize_ends
        
        self.episodes: List[Episode] = []
        self.total_steps = 0
        self.total_episodes = 0
        
        # For efficient sampling
        self._episode_lengths: List[int] = []
        self._cumulative_lengths: List[int] = []
        
        # Directory for persistence
        if directory is not None:
            self.directory = Path(directory)
            self.directory.mkdir(parents=True, exist_ok=True)
        else:
            self.directory = None
    
    def add(self, episode: Episode) -> None:
        """
        Add an episode to the buffer.
        
        Args:
            episode: Episode to add
        """
        # Filter by length
        if len(episode) < self.min_length:
            return
        
        if self.max_length > 0 and len(episode) > self.max_length:
            # Truncate episode
            episode = episode[:self.max_length]
        
        # Add episode
        self.episodes.append(episode)
        self._episode_lengths.append(len(episode))
        
        if self._cumulative_lengths:
            self._cumulative_lengths.append(
                self._cumulative_lengths[-1] + len(episode)
            )
        else:
            self._cumulative_lengths.append(len(episode))
        
        self.total_steps += len(episode)
        self.total_episodes += 1
        
        # Remove old episodes if over capacity
        while self.total_steps > self.capacity and len(self.episodes) > 1:
            removed = self.episodes.pop(0)
            removed_len = self._episode_lengths.pop(0)
            self.total_steps -= removed_len
            
            # Update cumulative lengths
            self._cumulative_lengths = [
                c - removed_len for c in self._cumulative_lengths
            ]
            self._cumulative_lengths.pop(0)
    
    def sample(
        self,
        batch_size: int,
        sequence_length: int,
        device: torch.device = torch.device('cpu'),
    ) -> Dict[str, Tensor]:
        """
        Sample random sequences from the buffer.
        
        Args:
            batch_size: Number of sequences to sample
            sequence_length: Length of each sequence
            device: Device to place tensors on
        
        Returns:
            Dictionary of batched tensors
        """
        if len(self.episodes) == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Sample episodes weighted by length
        total = sum(max(0, l - sequence_length + 1) for l in self._episode_lengths)
        
        if total == 0:
            # All episodes shorter than sequence_length, use what we have
            valid_episodes = [
                (i, ep) for i, ep in enumerate(self.episodes) 
                if len(ep) >= self.min_length
            ]
            if not valid_episodes:
                raise ValueError(
                    f"No episodes with length >= {self.min_length}"
                )
        
        sequences = []
        for _ in range(batch_size):
            # Sample episode
            if total > 0:
                # Weighted by available starting positions
                weights = [
                    max(0, l - sequence_length + 1) 
                    for l in self._episode_lengths
                ]
                ep_idx = random.choices(range(len(self.episodes)), weights=weights)[0]
            else:
                ep_idx = random.randint(0, len(self.episodes) - 1)
            
            episode = self.episodes[ep_idx]
            
            # Sample starting position
            max_start = max(0, len(episode) - sequence_length)
            
            if self.prioritize_ends and max_start > 0:
                # Prioritize positions near episode end
                # Use exponential distribution favoring later starts
                start = int(np.random.exponential(max_start / 2))
                start = min(start, max_start)
            else:
                start = random.randint(0, max_start)
            
            # Extract sequence (pad if necessary)
            end = start + sequence_length
            if end <= len(episode):
                seq = episode[start:end]
            else:
                # Pad with last observation
                seq = episode[start:]
                pad_len = sequence_length - len(seq)
                seq = self._pad_episode(seq, pad_len)
            
            sequences.append(seq)
        
        # Stack into batch
        return self._stack_sequences(sequences, device)
    
    def _pad_episode(self, episode: Episode, pad_len: int) -> Episode:
        """Pad episode by repeating last frame."""
        def pad_array(arr: np.ndarray, axis: int = 0) -> np.ndarray:
            pad_shape = list(arr.shape)
            pad_shape[axis] = pad_len
            padding = np.repeat(
                np.expand_dims(arr[-1], axis), 
                pad_len, 
                axis=axis
            )
            return np.concatenate([arr, padding], axis=axis)
        
        return Episode(
            obs=pad_array(episode.obs),
            action=pad_array(episode.action),
            reward=np.concatenate([
                episode.reward, 
                np.zeros(pad_len, dtype=episode.reward.dtype)
            ]),
            is_first=np.concatenate([
                episode.is_first, 
                np.zeros(pad_len, dtype=bool)
            ]),
            is_last=np.concatenate([
                episode.is_last[:-1],
                np.ones(pad_len + 1, dtype=bool)
            ]),
            is_terminal=np.concatenate([
                episode.is_terminal,
                np.zeros(pad_len, dtype=bool)
            ]),
            fg_mask=pad_array(episode.fg_mask) if episode.fg_mask is not None else None,
            bg_mask=pad_array(episode.bg_mask) if episode.bg_mask is not None else None,
        )
    
    def _stack_sequences(
        self,
        sequences: List[Episode],
        device: torch.device,
    ) -> Dict[str, Tensor]:
        """Stack sequences into batched tensors."""
        batch = {}
        
        # Stack observations
        batch['obs'] = torch.tensor(
            np.stack([s.obs for s in sequences]),
            dtype=torch.float32,
            device=device,
        )
        
        # Stack actions
        batch['action'] = torch.tensor(
            np.stack([s.action for s in sequences]),
            dtype=torch.float32,
            device=device,
        )
        
        # Stack rewards
        batch['reward'] = torch.tensor(
            np.stack([s.reward for s in sequences]),
            dtype=torch.float32,
            device=device,
        )
        
        # Stack flags
        batch['is_first'] = torch.tensor(
            np.stack([s.is_first for s in sequences]),
            dtype=torch.bool,
            device=device,
        )
        
        batch['is_last'] = torch.tensor(
            np.stack([s.is_last for s in sequences]),
            dtype=torch.bool,
            device=device,
        )
        
        batch['is_terminal'] = torch.tensor(
            np.stack([s.is_terminal for s in sequences]),
            dtype=torch.bool,
            device=device,
        )
        
        # Continue signal (1 if not terminal, 0 if terminal)
        batch['cont'] = (~batch['is_terminal']).float()
        
        # Stack masks if available
        if sequences[0].fg_mask is not None:
            batch['fg_mask'] = torch.tensor(
                np.stack([s.fg_mask for s in sequences]),
                dtype=torch.float32,
                device=device,
            )
        
        if sequences[0].bg_mask is not None:
            batch['bg_mask'] = torch.tensor(
                np.stack([s.bg_mask for s in sequences]),
                dtype=torch.float32,
                device=device,
            )
        
        return batch
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save all episodes to disk."""
        save_dir = Path(path) if path else self.directory
        if save_dir is None:
            raise ValueError("No directory specified for saving")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i, episode in enumerate(self.episodes):
            ep_path = save_dir / f"episode_{i:06d}.npz"
            np.savez_compressed(ep_path, **episode.to_dict())
    
    def load(self, path: Optional[Union[str, Path]] = None) -> None:
        """Load episodes from disk."""
        load_dir = Path(path) if path else self.directory
        if load_dir is None:
            raise ValueError("No directory specified for loading")
        
        episode_files = sorted(load_dir.glob("episode_*.npz"))
        
        for ep_file in episode_files:
            data = dict(np.load(ep_file))
            episode = Episode.from_dict(data)
            self.add(episode)
    
    def __len__(self) -> int:
        """Return number of episodes."""
        return len(self.episodes)
    
    @property
    def num_steps(self) -> int:
        """Return total number of timesteps."""
        return self.total_steps
    
    def stats(self) -> Dict[str, float]:
        """Return buffer statistics."""
        if len(self.episodes) == 0:
            return {
                'num_episodes': 0,
                'num_steps': 0,
                'avg_episode_length': 0,
                'min_episode_length': 0,
                'max_episode_length': 0,
            }
        
        lengths = self._episode_lengths
        return {
            'num_episodes': len(self.episodes),
            'num_steps': self.total_steps,
            'avg_episode_length': np.mean(lengths),
            'min_episode_length': min(lengths),
            'max_episode_length': max(lengths),
        }


class OnlineBuffer:
    """
    Small buffer for online collection during environment interaction.
    Accumulates transitions until episode end, then creates Episode object.
    """
    
    def __init__(self, obs_shape: Tuple[int, ...], action_dim: int):
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.reset()
    
    def reset(self):
        """Reset the buffer for a new episode."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.is_first = []
        self.is_terminal = []
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        is_first: bool,
        is_terminal: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.is_first.append(is_first)
        self.is_terminal.append(is_terminal)
    
    def get_episode(self) -> Episode:
        """Convert buffer contents to Episode object."""
        length = len(self.observations)
        
        # Create is_last array
        is_last = np.zeros(length, dtype=bool)
        is_last[-1] = True
        
        return Episode(
            obs=np.stack(self.observations),
            action=np.stack(self.actions),
            reward=np.array(self.rewards, dtype=np.float32),
            is_first=np.array(self.is_first, dtype=bool),
            is_last=is_last,
            is_terminal=np.array(self.is_terminal, dtype=bool),
        )
    
    def __len__(self) -> int:
        return len(self.observations)

