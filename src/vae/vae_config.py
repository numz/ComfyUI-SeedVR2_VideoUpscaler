"""
VAE Configuration Management for Wan2.1 and Wan2.2

This module provides centralized configuration management for VAE (Variational Autoencoder)
models, supporting both Wan2.1 and Wan2.2 architectures with flexible parameter handling.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any
import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class VAEModelVersion(Enum):
    """Supported VAE model versions"""
    WAN2_1 = "wan2.1"
    WAN2_2 = "wan2.2"


class VAEEncodingType(Enum):
    """VAE encoding modes"""
    STANDARD = "standard"
    ADVANCED = "advanced"
    CUSTOM = "custom"


@dataclass
class VAEArchitectureConfig:
    """VAE architecture-specific configuration"""
    latent_channels: int = 4
    latent_height: int = 64
    latent_width: int = 64
    scaling_factor: float = 0.18215
    shift_factor: float = 0.0
    encoder_channels: List[int] = field(default_factory=lambda: [3, 64, 128, 256, 512])
    decoder_channels: List[int] = field(default_factory=lambda: [512, 256, 128, 64, 3])
    block_types: List[str] = field(default_factory=lambda: ["ResBlock", "AttnBlock"])
    use_attention: bool = True
    attention_resolution: int = 16
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        config_dict = asdict(self)
        return config_dict


@dataclass
class VAEEncodingConfig:
    """VAE encoding configuration"""
    encoding_type: VAEEncodingType = VAEEncodingType.STANDARD
    precision: str = "fp32"  # fp32, fp16, bf16
    tile_size: Optional[int] = None  # For tiled encoding
    use_tiling: bool = False
    tiling_overlap: float = 0.1
    batch_encode: bool = True
    batch_size: int = 1
    normalize_input: bool = True
    clamp_output: bool = True
    output_range: tuple = field(default_factory=lambda: (-1.0, 1.0))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        config_dict = asdict(self)
        config_dict['encoding_type'] = self.encoding_type.value
        config_dict['output_range'] = list(self.output_range)
        return config_dict


@dataclass
class VAEModelConfig:
    """Complete VAE model configuration"""
    model_version: VAEModelVersion
    model_name: str
    model_path: str
    architecture: VAEArchitectureConfig = field(default_factory=VAEArchitectureConfig)
    encoding: VAEEncodingConfig = field(default_factory=VAEEncodingConfig)
    checkpoint_hash: Optional[str] = None
    weight_dtype: str = "fp32"
    device: str = "cuda"
    enable_gradient_checkpointing: bool = False
    enable_flash_attention: bool = True
    memory_efficient: bool = True
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_version": self.model_version.value,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "architecture": self.architecture.to_dict(),
            "encoding": self.encoding.to_dict(),
            "checkpoint_hash": self.checkpoint_hash,
            "weight_dtype": self.weight_dtype,
            "device": self.device,
            "enable_gradient_checkpointing": self.enable_gradient_checkpointing,
            "enable_flash_attention": self.enable_flash_attention,
            "memory_efficient": self.memory_efficient,
            "custom_params": self.custom_params,
        }


class VAEConfigManager:
    """Manages VAE configurations for different model versions"""
    
    # Predefined configurations for Wan2.1
    WAN2_1_DEFAULT = VAEModelConfig(
        model_version=VAEModelVersion.WAN2_1,
        model_name="Wan2.1-VAE",
        model_path="models/vae/wan2.1/vae.safetensors",
        architecture=VAEArchitectureConfig(
            latent_channels=4,
            latent_height=64,
            latent_width=64,
            scaling_factor=0.18215,
            encoder_channels=[3, 64, 128, 256, 512],
            decoder_channels=[512, 256, 128, 64, 3],
        ),
        encoding=VAEEncodingConfig(
            encoding_type=VAEEncodingType.STANDARD,
            precision="fp32",
            use_tiling=False,
        ),
        checkpoint_hash=None,
        enable_flash_attention=True,
    )
    
    # Predefined configurations for Wan2.2
    WAN2_2_DEFAULT = VAEModelConfig(
        model_version=VAEModelVersion.WAN2_2,
        model_name="Wan2.2-VAE",
        model_path="models/vae/wan2.2/vae.safetensors",
        architecture=VAEArchitectureConfig(
            latent_channels=4,
            latent_height=64,
            latent_width=64,
            scaling_factor=0.18215,
            encoder_channels=[3, 64, 128, 256, 512],
            decoder_channels=[512, 256, 128, 64, 3],
        ),
        encoding=VAEEncodingConfig(
            encoding_type=VAEEncodingType.ADVANCED,
            precision="fp32",
            use_tiling=False,
            enable_flash_attention=True,
        ),
        checkpoint_hash=None,
        enable_flash_attention=True,
    )
    
    def __init__(self):
        """Initialize the VAE configuration manager"""
        self.configs: Dict[str, VAEModelConfig] = {}
        self._register_default_configs()
    
    def _register_default_configs(self):
        """Register default configurations"""
        self.register_config("wan2.1", self.WAN2_1_DEFAULT)
        self.register_config("wan2.2", self.WAN2_2_DEFAULT)
        logger.info("Default VAE configurations registered")
    
    def register_config(self, config_id: str, config: VAEModelConfig):
        """
        Register a VAE configuration
        
        Args:
            config_id: Unique identifier for the configuration
            config: VAE configuration object
        """
        self.configs[config_id] = config
        logger.debug(f"Registered VAE config: {config_id}")
    
    def get_config(self, config_id: str) -> Optional[VAEModelConfig]:
        """
        Get a registered configuration
        
        Args:
            config_id: Identifier of the configuration
            
        Returns:
            VAE configuration or None if not found
        """
        return self.configs.get(config_id)
    
    def get_config_by_version(self, version: VAEModelVersion) -> Optional[VAEModelConfig]:
        """
        Get configuration by model version
        
        Args:
            version: Model version to retrieve
            
        Returns:
            VAE configuration or None if not found
        """
        for config in self.configs.values():
            if config.model_version == version:
                return config
        return None
    
    def list_configs(self) -> List[str]:
        """Get list of registered configuration IDs"""
        return list(self.configs.keys())
    
    def clone_config(self, config_id: str, new_config_id: str) -> bool:
        """
        Clone an existing configuration with a new ID
        
        Args:
            config_id: Source configuration ID
            new_config_id: Target configuration ID
            
        Returns:
            True if successful, False otherwise
        """
        if config_id not in self.configs:
            logger.warning(f"Source config not found: {config_id}")
            return False
        
        source_config = self.configs[config_id]
        # Create a deep copy by converting to dict and back
        config_dict = source_config.to_dict()
        new_config = self._dict_to_config(config_dict)
        self.register_config(new_config_id, new_config)
        logger.info(f"Cloned config {config_id} to {new_config_id}")
        return True
    
    def update_config(self, config_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update specific fields in a configuration
        
        Args:
            config_id: Configuration ID
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
        """
        if config_id not in self.configs:
            logger.warning(f"Config not found: {config_id}")
            return False
        
        config = self.configs[config_id]
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
                logger.debug(f"Updated {config_id}.{key} = {value}")
            else:
                logger.warning(f"Unknown config field: {key}")
        
        return True
    
    def save_config(self, config_id: str, filepath: str) -> bool:
        """
        Save configuration to JSON file
        
        Args:
            config_id: Configuration ID
            filepath: Path to save the configuration
            
        Returns:
            True if successful, False otherwise
        """
        if config_id not in self.configs:
            logger.warning(f"Config not found: {config_id}")
            return False
        
        try:
            config = self.configs[config_id]
            config_dict = config.to_dict()
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Saved config {config_id} to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def load_config(self, filepath: str, config_id: str) -> bool:
        """
        Load configuration from JSON file
        
        Args:
            filepath: Path to configuration file
            config_id: ID to register the loaded configuration under
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            config = self._dict_to_config(config_dict)
            self.register_config(config_id, config)
            logger.info(f"Loaded config from {filepath} as {config_id}")
            return True
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return False
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> VAEModelConfig:
        """Convert dictionary to VAE configuration"""
        # Extract nested configs
        arch_dict = config_dict.pop("architecture", {})
        enc_dict = config_dict.pop("encoding", {})
        
        # Convert enums
        model_version_str = config_dict.pop("model_version")
        model_version = VAEModelVersion(model_version_str)
        
        enc_type_str = enc_dict.pop("encoding_type", "standard")
        encoding_type = VAEEncodingType(enc_type_str)
        
        # Handle output_range as tuple
        if "output_range" in enc_dict:
            enc_dict["output_range"] = tuple(enc_dict["output_range"])
        
        # Create nested configs
        architecture = VAEArchitectureConfig(**arch_dict)
        encoding = VAEEncodingConfig(encoding_type=encoding_type, **enc_dict)
        
        # Create main config
        return VAEModelConfig(
            model_version=model_version,
            architecture=architecture,
            encoding=encoding,
            **config_dict
        )
    
    def export_all_configs(self, filepath: str) -> bool:
        """
        Export all configurations to a single JSON file
        
        Args:
            filepath: Path to save all configurations
            
        Returns:
            True if successful, False otherwise
        """
        try:
            all_configs = {
                config_id: config.to_dict()
                for config_id, config in self.configs.items()
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(all_configs, f, indent=2)
            
            logger.info(f"Exported {len(all_configs)} configs to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting configs: {e}")
            return False
    
    def import_configs(self, filepath: str, prefix: str = "") -> int:
        """
        Import configurations from JSON file
        
        Args:
            filepath: Path to configuration file
            prefix: Optional prefix for imported config IDs
            
        Returns:
            Number of configurations imported
        """
        try:
            with open(filepath, 'r') as f:
                configs_dict = json.load(f)
            
            count = 0
            for config_id, config_data in configs_dict.items():
                final_id = f"{prefix}{config_id}" if prefix else config_id
                config = self._dict_to_config(config_data)
                self.register_config(final_id, config)
                count += 1
            
            logger.info(f"Imported {count} configs from {filepath}")
            return count
        except Exception as e:
            logger.error(f"Error importing configs: {e}")
            return 0
    
    def get_config_summary(self, config_id: str) -> Optional[str]:
        """
        Get a human-readable summary of a configuration
        
        Args:
            config_id: Configuration ID
            
        Returns:
            Summary string or None if config not found
        """
        config = self.get_config(config_id)
        if not config:
            return None
        
        summary = (
            f"VAE Configuration: {config_id}\n"
            f"  Version: {config.model_version.value}\n"
            f"  Model: {config.model_name}\n"
            f"  Path: {config.model_path}\n"
            f"  Latent Channels: {config.architecture.latent_channels}\n"
            f"  Scaling Factor: {config.architecture.scaling_factor}\n"
            f"  Encoding Type: {config.encoding.encoding_type.value}\n"
            f"  Precision: {config.encoding.precision}\n"
            f"  Device: {config.device}\n"
            f"  Flash Attention: {config.enable_flash_attention}\n"
            f"  Memory Efficient: {config.memory_efficient}\n"
        )
        return summary


# Global configuration manager instance
_vae_config_manager = None


def get_vae_config_manager() -> VAEConfigManager:
    """
    Get or create the global VAE configuration manager
    
    Returns:
        Global VAE configuration manager instance
    """
    global _vae_config_manager
    if _vae_config_manager is None:
        _vae_config_manager = VAEConfigManager()
    return _vae_config_manager


def get_wan21_config() -> VAEModelConfig:
    """Get default Wan2.1 configuration"""
    return get_vae_config_manager().get_config("wan2.1")


def get_wan22_config() -> VAEModelConfig:
    """Get default Wan2.2 configuration"""
    return get_vae_config_manager().get_config("wan2.2")
