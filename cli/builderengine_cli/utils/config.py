"""Configuration management for the CLI."""

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """CLI configuration."""

    api_key: Optional[str] = None
    base_url: str = "https://api.builderengine.io"
    default_output: str = "table"
    default_organization: Optional[str] = None


def get_config_dir() -> Path:
    """Get the configuration directory path."""
    config_dir = Path.home() / ".builderengine"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_path() -> Path:
    """Get the configuration file path."""
    return get_config_dir() / "config.json"


def load_config() -> Config:
    """Load configuration from file."""
    config_path = get_config_path()

    if not config_path.exists():
        return Config()

    try:
        with open(config_path, "r") as f:
            data = json.load(f)
            return Config(
                api_key=data.get("api_key"),
                base_url=data.get("base_url", "https://api.builderengine.io"),
                default_output=data.get("default_output", "table"),
                default_organization=data.get("default_organization"),
            )
    except (json.JSONDecodeError, IOError):
        return Config()


def save_config(config: Config) -> None:
    """Save configuration to file."""
    config_path = get_config_path()

    # Set restrictive permissions
    config_path.touch(mode=0o600, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)


def get_api_key() -> Optional[str]:
    """Get API key from environment or config file."""
    # Environment variable takes precedence
    env_key = os.environ.get("BUILDERENGINE_API_KEY")
    if env_key:
        return env_key

    # Fall back to config file
    config = load_config()
    return config.api_key


def get_base_url() -> str:
    """Get base URL from environment or config file."""
    env_url = os.environ.get("BUILDERENGINE_BASE_URL")
    if env_url:
        return env_url

    config = load_config()
    return config.base_url
