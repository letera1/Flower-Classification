"""Core configuration and settings."""
from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
from typing import Optional
import yaml


class Settings(BaseSettings):
    """Application settings."""
    
    # App Info
    APP_NAME: str = "Flower Classification API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "ML-powered flower species classification"
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models" / "artifacts"
    LOGS_DIR: Path = BASE_DIR / "logs"
    CONFIG_DIR: Path = BASE_DIR / "configs"
    
    # Model Settings
    MODEL_PATH: Optional[str] = None
    PREPROCESSOR_PATH: Optional[str] = None
    DEFAULT_MODEL: str = "random_forest"
    
    # Data Settings
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class ConfigLoader:
    """Load configuration from YAML files."""
    
    def __init__(self, config_path: Optional[str] = None):
        settings = get_settings()
        self.config_path = config_path or settings.CONFIG_DIR / "config.yaml"
        self._config = None
    
    def load(self) -> dict:
        """Load configuration from YAML file."""
        if self._config is not None:
            return self._config
        
        if Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        else:
            self._config = self._get_default_config()
        
        return self._config
    
    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            "data": {
                "test_size": 0.2,
                "random_state": 42,
            },
            "models": {
                "primary": "random_forest",
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value by dot-notation key."""
        keys = key.split('.')
        value = self.load()
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


@lru_cache()
def get_config() -> ConfigLoader:
    """Get cached config loader instance."""
    return ConfigLoader()
