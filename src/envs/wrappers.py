"""
Environment Wrappers for DreamerV3.

Provides wrappers for DMControl environments to standardize
observation and action interfaces.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


class DMControlWrapper:
    """
    Wrapper for DMControl environments.
    
    Standardizes:
    - Observation format (image + optional proprio)
    - Action space (continuous, normalized to [-1, 1])
    - Episode interface (reset, step)
    """
    
    def __init__(
        self,
        env,
        image_size: int = 64,
        channels: int = 3,
        frame_stack: int = 1,
        action_repeat: int = 2,
        camera_id: int = 0,
    ):
        """
        Args:
            env: DMControl environment
            image_size: Size of rendered images
            channels: Number of color channels (1 for grayscale, 3 for RGB)
            frame_stack: Number of frames to stack
            action_repeat: Number of times to repeat each action
            camera_id: Camera ID for rendering
        """
        self._env = env
        self.image_size = image_size
        self.channels = channels
        self.frame_stack = frame_stack
        self.action_repeat = action_repeat
        self.camera_id = camera_id
        
        # Get action spec
        action_spec = env.action_spec()
        self.action_dim = action_spec.shape[0]
        self.action_low = action_spec.minimum
        self.action_high = action_spec.maximum
        
        # Frame stack buffer
        self._frames = []
        
        # Episode tracking
        self._step_count = 0
        self._episode_reward = 0.0
    
    @property
    def observation_space(self) -> Dict[str, Tuple[int, ...]]:
        """Return observation space shapes."""
        return {
            'image': (self.channels * self.frame_stack, self.image_size, self.image_size),
        }
    
    @property
    def action_space(self) -> Dict[str, Any]:
        """Return action space info."""
        return {
            'shape': (self.action_dim,),
            'low': -1.0,
            'high': 1.0,
        }
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.
        
        Returns:
            Tuple of (observation, info)
        """
        time_step = self._env.reset()
        
        # Clear frame stack
        self._frames = []
        self._step_count = 0
        self._episode_reward = 0.0
        
        # Render and stack initial frames
        frame = self._render()
        for _ in range(self.frame_stack):
            self._frames.append(frame)
        
        obs = self._get_observation()
        info = {
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (normalized to [-1, 1])
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Denormalize action
        action = self._denormalize_action(action)
        
        # Repeat action
        total_reward = 0.0
        for _ in range(self.action_repeat):
            time_step = self._env.step(action)
            total_reward += time_step.reward or 0.0
            
            if time_step.last():
                break
        
        self._step_count += 1
        self._episode_reward += total_reward
        
        # Update frame stack
        frame = self._render()
        self._frames.append(frame)
        if len(self._frames) > self.frame_stack:
            self._frames.pop(0)
        
        obs = self._get_observation()
        
        # Determine termination
        terminated = time_step.last() and time_step.discount == 0.0
        truncated = time_step.last() and time_step.discount != 0.0
        
        info = {
            'is_first': False,
            'is_last': time_step.last(),
            'is_terminal': terminated,
            'episode_reward': self._episode_reward,
            'step_count': self._step_count,
        }
        
        return obs, total_reward, terminated, truncated, info
    
    def _render(self) -> np.ndarray:
        """Render observation image."""
        frame = self._env.physics.render(
            height=self.image_size,
            width=self.image_size,
            camera_id=self.camera_id,
        )
        
        # Convert to channels-first format
        if self.channels == 1:
            # Convert to grayscale
            frame = np.mean(frame, axis=-1, keepdims=True)
        
        frame = frame.transpose(2, 0, 1)  # HWC -> CHW
        
        return frame.astype(np.uint8)
    
    def _get_observation(self) -> np.ndarray:
        """Get stacked observation."""
        if len(self._frames) < self.frame_stack:
            # Pad with copies of first frame
            frames = [self._frames[0]] * (self.frame_stack - len(self._frames))
            frames.extend(self._frames)
        else:
            frames = self._frames[-self.frame_stack:]
        
        return np.concatenate(frames, axis=0)
    
    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Convert action from [-1, 1] to environment range."""
        return self.action_low + (action + 1) / 2 * (self.action_high - self.action_low)
    
    def close(self):
        """Close the environment."""
        self._env.close()


class ActionRepeatWrapper:
    """Wrapper that repeats actions for multiple steps."""
    
    def __init__(self, env, action_repeat: int = 2):
        self._env = env
        self.action_repeat = action_repeat
        
        # Forward properties
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.action_dim = env.action_dim
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self._env.reset()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        total_reward = 0.0
        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, info = self._env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
    
    def close(self):
        self._env.close()


class FrameStackWrapper:
    """Wrapper that stacks consecutive frames."""
    
    def __init__(self, env, num_frames: int = 3):
        self._env = env
        self.num_frames = num_frames
        self._frames = []
        
        # Update observation space
        obs_shape = env.observation_space['image']
        self.observation_space = {
            'image': (obs_shape[0] * num_frames, obs_shape[1], obs_shape[2]),
        }
        self.action_space = env.action_space
        self.action_dim = env.action_dim
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self._env.reset()
        self._frames = [obs] * self.num_frames
        return self._get_stacked_obs(), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._frames.append(obs)
        self._frames = self._frames[-self.num_frames:]
        return self._get_stacked_obs(), reward, terminated, truncated, info
    
    def _get_stacked_obs(self) -> np.ndarray:
        return np.concatenate(self._frames, axis=0)
    
    def close(self):
        self._env.close()


class NormalizeActionWrapper:
    """Wrapper that normalizes actions to [-1, 1]."""
    
    def __init__(self, env):
        self._env = env
        self.observation_space = env.observation_space
        self.action_space = {'shape': env.action_space['shape'], 'low': -1.0, 'high': 1.0}
        self.action_dim = env.action_dim
        
        self._action_low = env.action_space.get('low', -1.0)
        self._action_high = env.action_space.get('high', 1.0)
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self._env.reset()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Denormalize action
        action = self._action_low + (action + 1) / 2 * (self._action_high - self._action_low)
        return self._env.step(action)
    
    def close(self):
        self._env.close()


class TimeLimit:
    """Wrapper that limits episode length."""
    
    def __init__(self, env, max_steps: int = 1000):
        self._env = env
        self.max_steps = max_steps
        self._step_count = 0
        
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.action_dim = env.action_dim
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._step_count = 0
        return self._env.reset()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step_count += 1
        
        if self._step_count >= self.max_steps:
            truncated = True
            info['is_last'] = True
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        self._env.close()

