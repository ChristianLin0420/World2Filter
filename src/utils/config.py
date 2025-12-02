"""
Configuration management utilities using Hydra and OmegaConf.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig, OmegaConf


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> DictConfig:
    """
    Load configuration from YAML file with optional overrides.
    
    Args:
        config_path: Path to the configuration file
        overrides: Dictionary of overrides to apply
    
    Returns:
        Merged configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    
    # Load referenced configs
    if "defaults" in config:
        base_dir = config_path.parent
        for default in config.defaults:
            if default == "_self_":
                continue
            default_path = base_dir / f"{default}.yaml"
            if default_path.exists():
                default_config = OmegaConf.load(default_path)
                config = OmegaConf.merge(default_config, config)
    
    # Apply overrides
    if overrides:
        override_config = OmegaConf.create(overrides)
        config = OmegaConf.merge(config, override_config)
    
    return config


def save_config(config: DictConfig, path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, path)


def config_to_dict(config: DictConfig) -> Dict[str, Any]:
    """Convert OmegaConf config to plain dictionary."""
    return OmegaConf.to_container(config, resolve=True)


def print_config(config: DictConfig) -> None:
    """Pretty print configuration."""
    print(OmegaConf.to_yaml(config))


def get_config_value(config: DictConfig, key: str, default: Any = None) -> Any:
    """
    Get nested config value using dot notation.
    
    Args:
        config: Configuration
        key: Dot-separated key (e.g., "world_model.rssm.deter_size")
        default: Default value if key not found
    
    Returns:
        Configuration value
    """
    keys = key.split(".")
    value = config
    
    for k in keys:
        if hasattr(value, k):
            value = getattr(value, k)
        elif isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple configurations, later ones override earlier."""
    result = OmegaConf.create({})
    for config in configs:
        result = OmegaConf.merge(result, config)
    return result


class ConfigManager:
    """Manages experiment configurations with automatic logging."""
    
    def __init__(
        self,
        config_dir: Union[str, Path] = "configs",
        log_dir: Union[str, Path] = "logs",
    ):
        self.config_dir = Path(config_dir)
        self.log_dir = Path(log_dir)
    
    def load_experiment_config(
        self,
        experiment_name: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> DictConfig:
        """Load configuration for an experiment."""
        # Load default config
        config = load_config(self.config_dir / "default.yaml", overrides)
        
        # Set experiment-specific paths
        config.experiment_name = experiment_name
        config.log_dir = str(self.log_dir / experiment_name)
        config.checkpoint_dir = str(self.log_dir / experiment_name / "checkpoints")
        
        return config
    
    def save_experiment_config(
        self,
        config: DictConfig,
        experiment_name: str,
    ) -> Path:
        """Save experiment configuration for reproducibility."""
        save_path = self.log_dir / experiment_name / "config.yaml"
        save_config(config, save_path)
        return save_path

