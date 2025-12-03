"""
Embodied Environment Wrapper.

Adapts World2Filter environments to match the embodied.Env interface
used by DreamerV3. This wrapper ensures proper observation and action
space formatting with reset signals included.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np


class EmbodiedSpace:
    """Simple space specification for embodied envs."""
    
    def __init__(self, dtype, shape, low=None, high=None):
        self.dtype = dtype
        self.shape = shape
        self.low = low
        self.high = high
        self.discrete = dtype in [np.bool_, bool]


class EmbodiedWrapper:
    """
    Wrapper to convert DMControl-style environments to embodied.Env interface.
    
    The embodied interface expects:
    - obs_space and act_space as dictionaries with Space objects
    - Observations include: image, reward, is_first, is_last, is_terminal
    - Actions include: action (main action) and reset (boolean)
    - step() returns a dictionary with all observation keys
    """
    
    def __init__(self, env):
        """
        Args:
            env: Base environment (DMControlWrapper or similar)
        """
        self._env = env
        self._step_count = 0
        self._episode_return = 0.0
        
    @property
    def obs_space(self) -> Dict[str, EmbodiedSpace]:
        """Return observation space in embodied format."""
        obs_shape = self._env.observation_space['image']
        return {
            'image': EmbodiedSpace(np.uint8, obs_shape),
            'reward': EmbodiedSpace(np.float32, ()),
            'is_first': EmbodiedSpace(np.bool_, ()),
            'is_last': EmbodiedSpace(np.bool_, ()),
            'is_terminal': EmbodiedSpace(np.bool_, ()),
        }
    
    @property
    def observation_space(self) -> Dict[str, EmbodiedSpace]:
        """Return observation space (gym naming compatibility)."""
        return self.obs_space
    
    @property
    def act_space(self) -> Dict[str, EmbodiedSpace]:
        """Return action space in embodied format."""
        return {
            'action': EmbodiedSpace(
                np.float32, 
                (self._env.action_dim,), 
                low=-1.0, 
                high=1.0
            ),
            'reset': EmbodiedSpace(np.bool_, ()),
        }
    
    @property
    def action_space(self) -> Dict[str, EmbodiedSpace]:
        """Return action space (gym naming compatibility)."""
        return self.act_space
    
    @property
    def action_dim(self) -> int:
        """Return action dimension for compatibility."""
        return self._env.action_dim
    
    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict[str, Any], float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Dictionary with 'action' and 'reset' keys
        
        Returns:
            Tuple of (observation_dict, reward, terminated, truncated, info) in gym format
        """
        # Handle reset - return first observation with zero reward
        if action.get('reset', False):
            obs_dict = self._reset()
            # Return gym-style: (obs, reward, terminated, truncated, info)
            return obs_dict, 0.0, False, False, {}
        
        # Take action
        obs, reward, terminated, truncated, info = self._env.step(action['action'])
        
        self._step_count += 1
        self._episode_return += reward
        
        # Build embodied observation dict
        obs_dict = {
            'image': obs,
            'reward': np.float32(reward),
            'is_first': np.bool_(False),
            'is_last': np.bool_(terminated or truncated),
            'is_terminal': np.bool_(terminated),
        }
        
        # Return gym-style: (obs, reward, terminated, truncated, info)
        return obs_dict, reward, terminated, truncated, info
    
    def reset(self) -> Tuple[Dict[str, Any], Dict]:
        """Public reset method (returns obs, info tuple for gym compatibility)."""
        obs_dict = self._reset()
        return obs_dict, {}
    
    def _reset(self) -> Dict[str, Any]:
        """Reset the environment."""
        obs, info = self._env.reset()
        
        self._step_count = 0
        self._episode_return = 0.0
        
        return {
            'image': obs,
            'reward': np.float32(0.0),
            'is_first': np.bool_(True),
            'is_last': np.bool_(False),
            'is_terminal': np.bool_(False),
        }
    
    def close(self):
        """Close the environment."""
        self._env.close()


class ActionNormalizationWrapper:
    """
    Wrapper that ensures actions are properly normalized.
    
    In the embodied framework, actions should be in [-1, 1] range.
    This wrapper adds the normalization if needed.
    """
    
    def __init__(self, env):
        self._env = env
        
    @property
    def obs_space(self):
        return self._env.obs_space
    
    @property
    def observation_space(self):
        """Gym naming compatibility."""
        return self.obs_space
    
    @property
    def act_space(self):
        # Ensure actions are normalized to [-1, 1]
        act_space = self._env.act_space.copy()
        if 'action' in act_space:
            act_space['action'] = EmbodiedSpace(
                np.float32,
                act_space['action'].shape,
                low=-1.0,
                high=1.0
            )
        return act_space
    
    @property
    def action_space(self):
        """Gym naming compatibility."""
        return self.act_space
    
    @property
    def action_dim(self):
        """Action dimension for compatibility."""
        if hasattr(self._env, 'action_dim'):
            return self._env.action_dim
        return self.act_space['action'].shape[0]
    
    def step(self, action):
        # Actions should already be in [-1, 1] from the policy
        # But we clip just in case
        if isinstance(action, dict) and 'action' in action:
            action['action'] = np.clip(action['action'], -1.0, 1.0)
        return self._env.step(action)
    
    def reset(self):
        """Reset the environment."""
        return self._env.reset()
    
    def close(self):
        self._env.close()


class ExpandScalarsWrapper:
    """
    Wrapper that expands scalar observations to have a batch dimension.
    
    This is needed for compatibility with DreamerV3's batch processing.
    """
    
    def __init__(self, env):
        self._env = env
    
    @property
    def obs_space(self):
        return self._env.obs_space
    
    @property
    def observation_space(self):
        """Gym naming compatibility."""
        return self.obs_space
    
    @property
    def act_space(self):
        return self._env.act_space
    
    @property
    def action_space(self):
        """Gym naming compatibility."""
        return self.act_space
    
    @property
    def action_dim(self):
        """Action dimension for compatibility."""
        if hasattr(self._env, 'action_dim'):
            return self._env.action_dim
        return self.act_space['action'].shape[0]
    
    def step(self, action):
        result = self._env.step(action)
        
        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 5:
            # Standard gym format: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = result
            # Ensure scalar values have proper shape in obs dict
            if isinstance(obs, dict):
                for key in ['reward', 'is_first', 'is_last', 'is_terminal']:
                    if key in obs and np.isscalar(obs[key]):
                        obs[key] = np.array(obs[key], dtype=self.obs_space[key].dtype)
            return obs, reward, terminated, truncated, info
        else:
            # Pass through if not standard format
            return result
    
    def reset(self):
        """Reset the environment."""
        obs = self._env.reset()
        # Handle both tuple (obs, info) and dict returns
        if isinstance(obs, tuple):
            obs_dict, info = obs
            # Ensure scalars have proper shape in obs
            for key in ['reward', 'is_first', 'is_last', 'is_terminal']:
                if key in obs_dict and np.isscalar(obs_dict[key]):
                    obs_dict[key] = np.array(obs_dict[key], dtype=self.obs_space[key].dtype)
            return obs_dict, info
        else:
            # Direct dict return
            for key in ['reward', 'is_first', 'is_last', 'is_terminal']:
                if key in obs and np.isscalar(obs[key]):
                    obs[key] = np.array(obs[key], dtype=self.obs_space[key].dtype)
            return obs
    
    def close(self):
        self._env.close()


def wrap_embodied(env) -> EmbodiedWrapper:
    """
    Wrap an environment with all necessary embodied wrappers.
    
    Args:
        env: Base environment to wrap
    
    Returns:
        Fully wrapped environment ready for DreamerV3
    """
    # Convert to embodied interface
    env = EmbodiedWrapper(env)
    
    # Add action normalization
    env = ActionNormalizationWrapper(env)
    
    # Expand scalars
    env = ExpandScalarsWrapper(env)
    
    return env

