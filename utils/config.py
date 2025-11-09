# utils/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class GenerationConfigs:
    """Configuration for Fireworks batch generation."""
    model: str
    temperature: float
    max_tokens: int
    top_p: float
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> GenerationConfigs:
        """Create GenerationConfigs from dictionary."""
        return cls(
            model=config_dict['model'],
            temperature=config_dict['temperature'],
            max_tokens=config_dict['max_tokens'],
            top_p=config_dict['top_p']
        )


@dataclass
class SFTDataGenerationConfig:
    """Configuration for SFT data generation."""
    base_dataset: str
    system_prompt: str
    generation_configs: GenerationConfigs
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> SFTDataGenerationConfig:
        """Create SFTDataGenerationConfig from dictionary."""
        return cls(
            base_dataset=config_dict['base_dataset'],
            system_prompt=config_dict['system_prompt'],
            generation_configs=GenerationConfigs.from_dict(
                config_dict['generation_configs']
            )
        )


def load_generation_config(config_path: Path) -> SFTDataGenerationConfig:
    """
    Load generation config from YAML file.
    
    Args:
        config_path: Path to the YAML config file
        
    Returns:
        SFTDataGenerationConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
        KeyError: If required config keys are missing
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return SFTDataGenerationConfig.from_dict(config_dict)
