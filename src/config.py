"""
Configuration management for the Bitcoin Volatility Trading System.

Loads settings from YAML configuration file and environment variables.
Environment variables take precedence over config file values.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


# Load environment variables from .env file if present
load_dotenv()


class Config:
    """
    Configuration manager that loads settings from YAML and environment variables.

    Usage:
        config = Config()
        db_path = config.get("database.path")
        bb_period = config.get("bollinger.period", default=20)
    """

    # Default configuration values
    DEFAULTS = {
        "exchanges": {
            "binance": {"enabled": False, "rate_limit": 1200, "timeout": 30000},
            "kraken": {"enabled": True, "rate_limit": 300, "timeout": 30000},
        },
        "symbols": ["BTC/USDT"],
        "timeframes": {"default": "1d", "available": ["1h", "4h", "1d"]},
        "volatility": {
            "windows": {"short": 7, "medium": 30, "long": 90},
            "annualization_factor": 365,
        },
        "bollinger": {"period": 20, "std_dev": 2.0},
        "atr": {"period": 14},
        "percentile": {"lookback": 252},
        "database": {"path": "data/btc_data.db"},
        "logging": {
            "level": "INFO",
            "file": "logs/trading.log",
            "max_bytes": 10485760,
            "backup_count": 5,
        },
        "data": {"limit": 1000, "default_history_days": 365},
    }

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize configuration from YAML file.

        Args:
            config_path: Path to YAML config file. If None, looks for
                        config/settings.yaml in the project root.
        """
        self._config: dict[str, Any] = {}
        self._project_root = self._find_project_root()

        # Determine config file path
        if config_path is None:
            config_path = self._project_root / "config" / "settings.yaml"
        else:
            config_path = Path(config_path)

        # Load configuration
        self._load_config(config_path)

        # Load API keys from environment
        self._load_env_vars()

    def _find_project_root(self) -> Path:
        """Find the project root directory (where config/ is located)."""
        current = Path(__file__).resolve().parent

        # Go up until we find config/ or hit filesystem root
        while current != current.parent:
            if (current / "config").is_dir():
                return current
            current = current.parent

        # Fallback to current working directory
        return Path.cwd()

    def _load_config(self, config_path: Path) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file.
        """
        # Start with defaults
        self._config = self.DEFAULTS.copy()

        # Load from file if it exists
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    file_config = yaml.safe_load(f) or {}
                self._deep_merge(self._config, file_config)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in config file: {e}")
            except IOError as e:
                raise ValueError(f"Cannot read config file: {e}")

    def _load_env_vars(self) -> None:
        """Load API keys and sensitive data from environment variables."""
        # Exchange API keys
        env_mappings = {
            "BINANCE_API_KEY": ("exchanges", "binance", "api_key"),
            "BINANCE_SECRET": ("exchanges", "binance", "secret"),
            "COINBASE_API_KEY": ("exchanges", "coinbase", "api_key"),
            "COINBASE_SECRET": ("exchanges", "coinbase", "secret"),
            "KRAKEN_API_KEY": ("exchanges", "kraken", "api_key"),
            "KRAKEN_SECRET": ("exchanges", "kraken", "secret"),
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self._set_nested(config_path, value)

    def _deep_merge(self, base: dict, update: dict) -> None:
        """
        Recursively merge update dict into base dict.

        Args:
            base: Base dictionary to merge into (modified in place).
            update: Dictionary with values to merge.
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _set_nested(self, path: tuple[str, ...], value: Any) -> None:
        """
        Set a nested configuration value.

        Args:
            path: Tuple of keys representing the path to the value.
            value: Value to set.
        """
        current = self._config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., "bollinger.period").
            default: Default value if key is not found.

        Returns:
            The configuration value or default.

        Examples:
            >>> config.get("bollinger.period")
            20
            >>> config.get("missing.key", default="fallback")
            "fallback"
        """
        keys = key.split(".")
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_path(self, key: str) -> Path:
        """
        Get a configuration value as an absolute Path.

        Relative paths are resolved relative to the project root.

        Args:
            key: Configuration key for a path value.

        Returns:
            Absolute Path object.
        """
        path_str = self.get(key, "")
        path = Path(path_str)

        if not path.is_absolute():
            path = self._project_root / path

        return path

    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return self._project_root

    def __repr__(self) -> str:
        """String representation of the config."""
        return f"Config(project_root={self._project_root})"


# Global config instance for easy access
_config: Config | None = None


def get_config(config_path: str | Path | None = None) -> Config:
    """
    Get the global configuration instance.

    Creates a new instance if one doesn't exist or if a specific
    config_path is provided.

    Args:
        config_path: Optional path to a specific config file.

    Returns:
        Config instance.
    """
    global _config

    if _config is None or config_path is not None:
        _config = Config(config_path)

    return _config
