"""
Configuration Management Module

Handles loading and accessing configuration parameters from config.yaml
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the forecasting system"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to config.yaml file. If None, uses default location.
        """
        if config_path is None:
            # Get project root (parent of src directory)
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._setup_paths()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise
    
    def _setup_paths(self):
        """Create necessary directories if they don't exist"""
        project_root = Path(__file__).parent.parent
        
        paths_to_create = [
            project_root / self._config['data']['raw_data_path'],
            project_root / self._config['data']['processed_data_path'],
            project_root / self._config['data']['predictions_path'],
            project_root / 'models',
            project_root / 'logs'
        ]
        
        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'api.nuuka.base_url')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    @property
    def nuuka_base_url(self) -> str:
        return self.get('api.nuuka.base_url')
    
    @property
    def location_id(self) -> str:
        return self.get('api.nuuka.location_id')
    
    @property
    def latitude(self) -> float:
        return self.get('location.latitude')
    
    @property
    def longitude(self) -> float:
        return self.get('location.longitude')
    
    @property
    def timezone(self) -> str:
        return self.get('location.timezone')
    
    @property
    def start_date(self) -> str:
        return self.get('data.start_date')
    
    @property
    def train_end_date(self) -> str:
        return self.get('data.train_end_date')
    
    @property
    def model_type(self) -> str:
        return self.get('model.type')
    
    @property
    def model_params(self) -> Dict[str, Any]:
        return self.get('model.hyperparameters', {})
    
    @property
    def features(self) -> list:
        return self.get('model.features', [])
    
    @property
    def target(self) -> str:
        return self.get('model.target')
    
    @property
    def model_save_path(self) -> Path:
        project_root = Path(__file__).parent.parent
        return project_root / self.get('model.save_path')
    
    @property
    def forecast_hours(self) -> int:
        return self.get('prediction.forecast_hours', 24)
    
    @property
    def log_level(self) -> str:
        return self.get('logging.level', 'INFO')
    
    @property
    def log_file(self) -> Path:
        project_root = Path(__file__).parent.parent
        return project_root / self.get('logging.file')


# Global config instance
_config_instance = None


def get_config(config_path: str = None) -> Config:
    """
    Get global configuration instance (singleton pattern)
    
    Args:
        config_path: Path to config file (only used on first call)
        
    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance
