# utils/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
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



@dataclass
class SFTSettings:
    """Configuration for SFT training."""
    deployment_type: str  # "serverless", "on-demand", or "auto"
    epochs: int
    learning_rate: float
    lora_rank: int
    max_context_length: int
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> SFTSettings:
        """Create SFTSettings from dictionary."""
        return cls(
            deployment_type=config_dict['deployment_type'],
            epochs=config_dict['epochs'],
            learning_rate=config_dict['learning_rate'],
            lora_rank=config_dict['lora_rank'],
            max_context_length=config_dict['max_context_length']
        )


@dataclass
class SFTConfig:
    """Configuration for supervised fine-tuning."""
    system_prompt: Optional[str]  # Path to prompt file, or None
    base_model: str
    sft_settings: SFTSettings
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> SFTConfig:
        """Create SFTConfig from dictionary."""
        return cls(
            system_prompt=config_dict.get('system_prompt'),  # May be None
            base_model=config_dict['base_model'],
            sft_settings=SFTSettings.from_dict(config_dict['sft_settings'])
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


def load_sft_config(config_path: Path) -> SFTConfig:
    """
    Load SFT config from YAML file.
    
    Args:
        config_path: Path to the YAML config file
        
    Returns:
        SFTConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
        KeyError: If required config keys are missing
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return SFTConfig.from_dict(config_dict)