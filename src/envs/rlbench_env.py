"""
RLBench Environment Wrapper for World2Filter.

Provides embodied-compatible wrapper for RLBench robotic manipulation tasks.
Based on psp_camera_ready's implementation.
"""

import functools
from typing import Dict, Any, Tuple

import numpy as np

try:
    from pyrep.const import RenderMode
    from pyrep.errors import IKError, ConfigurationPathError
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointPosition
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.backend.exceptions import InvalidActionError
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig
    from rlbench.tasks import ReachTarget
    RLBENCH_AVAILABLE = True
except ImportError:
    RLBENCH_AVAILABLE = False


class RLBenchWrapper:
    """
    RLBench environment wrapper compatible with embodied interface.
    
    Currently supports: reach_target
    """
    
    def __init__(
        self,
        task: str = 'reach_target',
        image_size: Tuple[int, int] = (64, 64),
        action_repeat: int = 1,
        shadows: bool = True,
        max_length: int = 200,
    ):
        """
        Args:
            task: Task name (currently only 'reach_target' supported)
            image_size: (height, width) for rendered images
            action_repeat: Number of action repeats
            shadows: Enable shadows in rendering
            max_length: Maximum episode length
        """
        if not RLBENCH_AVAILABLE:
            raise ImportError(
                "RLBench not installed. Install with:\n"
                "  pip install git+https://github.com/stepjam/RLBench.git\n"
                "  pip install git+https://github.com/stepjam/PyRep.git"
            )
        
        # Configure observation
        obs_config = ObservationConfig()
        obs_config.left_shoulder_camera.set_all(False)
        obs_config.right_shoulder_camera.set_all(False)
        obs_config.overhead_camera.set_all(False)
        obs_config.wrist_camera.set_all(False)
        obs_config.front_camera.image_size = image_size
        obs_config.front_camera.depth = False
        obs_config.front_camera.point_cloud = False
        obs_config.front_camera.mask = False
        obs_config.front_camera.render_mode = (
            RenderMode.OPENGL3 if shadows else RenderMode.OPENGL
        )
        
        # Configure action mode (joint position control)
        action_mode = functools.partial(JointPosition, absolute_mode=False)
        
        # Create environment
        self._env = Environment(
            action_mode=MoveArmThenGripper(
                arm_action_mode=action_mode(),
                gripper_action_mode=Discrete()
            ),
            obs_config=obs_config,
            headless=True,
            shaped_rewards=True,
        )
        self._env.launch()
        
        # Load task
        if task == 'reach_target':
            task_class = ReachTarget
        else:
            raise ValueError(
                f"Task '{task}' not supported. "
                f"Currently only 'reach_target' is implemented."
            )
        
        self._task = self._env.get_task(task_class)
        _, obs = self._task.reset()
        self._prev_obs = None
        
        self._size = image_size
        self._action_repeat = action_repeat
        self._step = 0
        self._max_length = max_length
        
        print(f"RLBench environment initialized: {task}")
        print(f"  Image size: {image_size}")
        print(f"  Action repeat: {action_repeat}")
        print(f"  Action shape: {self._env.action_shape}")
    
    @property
    def observation_space(self):
        """Observation space (Gym naming)."""
        return {
            'image': {'shape': self._size + (3,), 'dtype': np.uint8},
            'reward': {'shape': (), 'dtype': np.float32},
            'is_first': {'shape': (), 'dtype': np.bool_},
            'is_last': {'shape': (), 'dtype': np.bool_},
            'is_terminal': {'shape': (), 'dtype': np.bool_},
            'success': {'shape': (), 'dtype': np.bool_},
        }
    
    @property
    def action_space(self):
        """Action space (Gym naming)."""
        return {
            'action': {
                'shape': tuple(int(i) for i in self._env.action_shape),
                'dtype': np.float32,
                'low': -1.0,
                'high': 1.0,
            }
        }
    
    @property
    def action_dim(self):
        """Action dimension."""
        return int(self._env.action_shape[0])
    
    def reset(self) -> Tuple[Dict[str, Any], Dict]:
        """Reset environment."""
        _, obs = self._task.reset()
        self._prev_obs = obs
        self._step = 0
        
        obs_dict = {
            'image': obs.front_rgb,
            'reward': np.float32(0.0),
            'is_first': np.bool_(True),
            'is_last': np.bool_(False),
            'is_terminal': np.bool_(False),
            'success': np.bool_(False),
        }
        return obs_dict, {}
    
    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict[str, Any], float, bool, bool, Dict]:
        """Step environment."""
        # Handle reset
        if action.get('reset', False):
            return self.reset()[0], 0.0, False, False, {}
        
        # Execute action with repeats
        terminal = True
        success = False
        total_reward = 0.0
        obs = self._prev_obs
        
        for _ in range(self._action_repeat):
            try:
                obs, reward, terminal = self._task.step(action['action'])
                success = terminal
                total_reward += reward
                if terminal:
                    break
            except (IKError, ConfigurationPathError, InvalidActionError) as e:
                # Recover from errors
                terminal = True
                success = False
                break
        
        self._prev_obs = obs
        self._step += 1
        
        # Check max length
        truncated = (self._step >= self._max_length)
        done = terminal or truncated
        
        obs_dict = {
            'image': obs.front_rgb,
            'reward': np.float32(total_reward),
            'is_first': np.bool_(False),
            'is_last': np.bool_(done),
            'is_terminal': np.bool_(terminal),
            'success': np.bool_(success),
        }
        
        return obs_dict, total_reward, terminal, truncated, {'success': success}
    
    def close(self):
        """Close environment."""
        self._env.shutdown()


def make_rlbench_env(
    task: str = 'reach_target',
    image_size: int = 64,
    action_repeat: int = 1,
    shadows: bool = True,
    max_length: int = 200,
) -> RLBenchWrapper:
    """
    Create RLBench environment.
    
    Args:
        task: Task name ('reach_target')
        image_size: Image size (will be (image_size, image_size))
        action_repeat: Action repeat
        shadows: Enable shadows
        max_length: Max episode length
        
    Returns:
        RLBench environment wrapper
        
    Example:
        >>> env = make_rlbench_env('reach_target')
        >>> obs, info = env.reset()
        >>> obs, reward, term, trunc, info = env.step({'action': action, 'reset': False})
    """
    from src.envs.embodied_wrapper import (
        EmbodiedWrapper, ActionNormalizationWrapper, ExpandScalarsWrapper
    )
    
    env = RLBenchWrapper(
        task=task,
        image_size=(image_size, image_size),
        action_repeat=action_repeat,
        shadows=shadows,
        max_length=max_length,
    )
    
    # Note: RLBench already returns embodied-style observations
    # So we only need normalization and scalar expansion
    env = ActionNormalizationWrapper(env)
    env = ExpandScalarsWrapper(env)
    
    return env

