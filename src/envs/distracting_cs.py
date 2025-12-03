"""
Distracting Control Suite Environment Setup.

Provides utilities for creating environments from the Distracting
Control Suite for robust visual RL training.

Includes ColorGridBackground support for MDP-correlated distractions.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.envs.wrappers import DMControlWrapper, TimeLimit
from src.envs.color_grid_utils import ColorGridBackground, EvilEnum, EVIL_CHOICE_CONVENIENCE_MAPPING


def make_distracting_cs_env(
    domain: str = "walker",
    task: str = "walk",
    image_size: int = 64,
    action_repeat: int = 2,
    frame_stack: int = 1,
    time_limit: int = 1000,
    # Distraction settings
    background: bool = True,
    camera: bool = False,
    color: bool = False,
    background_dataset_path: Optional[str] = None,
    background_dataset_videos: Optional[str] = None,
    background_dynamic: bool = True,
    difficulty: str = "easy",
    # ColorGrid distraction settings (MDP-correlated)
    use_color_grid: bool = False,
    evil_level: str = "max",
    num_cells_per_dim: int = 16,
    num_colors_per_cell: int = 11664,
    action_dims_to_split: Optional[list] = None,
    action_power: int = 3,
    seed: int = 42,
) -> DMControlWrapper:
    """
    Create a Distracting Control Suite environment.
    
    Args:
        domain: Domain name (e.g., 'walker', 'cheetah', 'cartpole')
        task: Task name (e.g., 'walk', 'run', 'swingup')
        image_size: Size of rendered images
        action_repeat: Number of action repeats
        frame_stack: Number of frames to stack
        time_limit: Maximum episode length
        background: Enable background distractions
        camera: Enable camera distractions
        color: Enable color distractions
        background_dataset_path: Path to background video dataset
        background_dataset_videos: Specific videos to use
        background_dynamic: Use dynamic (video) backgrounds
        difficulty: Distraction difficulty ('easy', 'medium', 'hard')
        use_color_grid: Use ColorGrid MDP-correlated distractions (recommended)
        evil_level: Type of MDP correlation ('max', 'action', 'reward', 'sequence', 'none')
        num_cells_per_dim: Number of grid cells per dimension (e.g., 16 for 16x16 grid)
        num_colors_per_cell: Total number of color patterns per cell
        action_dims_to_split: Which action dimensions to correlate with background
        action_power: How many discrete bins per action dimension
        seed: Random seed
    
    Returns:
        Wrapped DMControl environment
    """
    # Load base DMControl environment
    from dm_control import suite
    
    env = suite.load(
        domain_name=domain,
        task_name=task,
        task_kwargs={'random': seed},
    )
    
    # Set up ColorGrid background if requested
    color_bg = None
    if use_color_grid:
        # Get action dimensions if not specified
        if action_dims_to_split is None:
            # Use all action dimensions by default
            action_spec = env.action_spec()
            action_dims_to_split = list(range(action_spec.shape[0]))
        
        # Convert evil level string to enum
        evil_enum = EVIL_CHOICE_CONVENIENCE_MAPPING.get(evil_level, EvilEnum.MAXIMUM_EVIL)
        
        print(f"Setting up ColorGrid background: {evil_level} level")
        print(f"  Grid: {num_cells_per_dim}x{num_cells_per_dim}")
        print(f"  Colors per cell: {num_colors_per_cell}")
        print(f"  Action dims: {action_dims_to_split}, power: {action_power}")
        
        color_bg = ColorGridBackground(
            domain_name=domain,
            task_name=task,
            num_cells_per_dim=num_cells_per_dim,
            num_colors_per_cell=num_colors_per_cell,
            evil_level=evil_enum,
            action_dims_to_split=action_dims_to_split,
            action_power=action_power,
            height=image_size,
            width=image_size,
            random_seed=seed,
        )
    
    # Wrap environment with ColorGrid support
    env = DMControlWrapper(
        env,
        image_size=image_size,
        channels=3,
        frame_stack=frame_stack,
        action_repeat=action_repeat,
        color_bg=color_bg,
    )
    
    # Add time limit
    env = TimeLimit(env, max_steps=time_limit // action_repeat)
    
    return env


def make_dm_control_env(
    domain: str = "walker",
    task: str = "walk",
    image_size: int = 64,
    action_repeat: int = 2,
    frame_stack: int = 1,
    time_limit: int = 1000,
    seed: int = 42,
) -> DMControlWrapper:
    """
    Create a standard DMControl environment (no distractions).
    
    Args:
        domain: Domain name
        task: Task name
        image_size: Size of rendered images
        action_repeat: Number of action repeats
        frame_stack: Number of frames to stack
        time_limit: Maximum episode length
        seed: Random seed
    
    Returns:
        Wrapped DMControl environment
    """
    from dm_control import suite
    
    env = suite.load(
        domain_name=domain,
        task_name=task,
        task_kwargs={'random': seed},
    )
    
    env = DMControlWrapper(
        env,
        image_size=image_size,
        channels=3,
        frame_stack=frame_stack,
        action_repeat=action_repeat,
    )
    
    env = TimeLimit(env, max_steps=time_limit // action_repeat)
    
    return env


class VectorizedEnv:
    """
    Simple vectorized environment for parallel data collection.
    """
    
    def __init__(self, env_fns, num_envs: int):
        """
        Args:
            env_fns: List of environment creation functions
            num_envs: Number of parallel environments
        """
        self.envs = [fn() for fn in env_fns[:num_envs]]
        self.num_envs = len(self.envs)
        
        # Get specs from first environment
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.action_dim = self.envs[0].action_dim
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset all environments."""
        obs_list = []
        info_list = []
        
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)
        
        return np.stack(obs_list), self._stack_info(info_list)
    
    def step(
        self, 
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Step all environments."""
        obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        info_list = []
        
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            
            # Auto-reset if done
            if terminated or truncated:
                reset_obs, reset_info = env.reset()
                info['terminal_observation'] = obs
                obs = reset_obs
            
            obs_list.append(obs)
            reward_list.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            info_list.append(info)
        
        return (
            np.stack(obs_list),
            np.array(reward_list),
            np.array(terminated_list),
            np.array(truncated_list),
            self._stack_info(info_list),
        )
    
    def _stack_info(self, info_list: list) -> Dict[str, Any]:
        """Stack info dictionaries."""
        stacked = {}
        for key in info_list[0].keys():
            if isinstance(info_list[0][key], bool):
                stacked[key] = np.array([info[key] for info in info_list])
            else:
                stacked[key] = [info[key] for info in info_list]
        return stacked
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


def get_env_info(config: Dict) -> Dict[str, Any]:
    """
    Get environment information from config.
    
    Returns action dimension and observation shape.
    """
    # Create temporary environment to get info
    env = make_distracting_cs_env(
        domain=config['environment']['domain'],
        task=config['environment']['task'],
        image_size=config['environment']['obs']['image_size'],
        action_repeat=config['environment']['obs']['action_repeat'],
        frame_stack=config['environment']['obs']['frame_stack'],
        background=config['environment']['distractions']['background'],
        camera=config['environment']['distractions']['camera'],
        color=config['environment']['distractions']['color'],
    )
    
    info = {
        'action_dim': env.action_dim,
        'obs_shape': env.observation_space['image'],
    }
    
    env.close()
    return info

