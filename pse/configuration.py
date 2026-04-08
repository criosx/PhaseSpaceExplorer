from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Any

from roadmap_datamanager.configuration import BaseConfig, load_config, save_config

@dataclass
class DataManagerConfig(BaseConfig):
    CONFIG_ENV_VAR: ClassVar[str] = "PSE_APP_CONFIG"
    CONFIG_APP_NAME: ClassVar[str] = "pse_app"
    CONFIG_APP_AUTHOR: ClassVar[str] = "streamlit"
    CONFIG_FILENAME: ClassVar[str] = "config.json"

    # PSE specific settings
    optimizer: str = 'gpcam'
    acquisition_function: str = 'variance'
    gp_iterations: int = 50
    initial_iterations: int = 10
    client: str = 'Test Ackley Function'
    parallel_measurements: int = 1

def load_persistent_cfg() -> DataManagerConfig:
    config_cls = DataManagerConfig
    return load_config(
        config_cls,
        env_var=getattr(config_cls, "CONFIG_ENV_VAR", None),
        app_name=config_cls.CONFIG_APP_NAME,
        app_author=config_cls.CONFIG_APP_AUTHOR,
        filename=getattr(config_cls, "CONFIG_FILENAME", "config.json"),
    )

def save_persistent_cfg(data: Any) -> Path:
    cls = type(data)
    return save_config(
        data,
        env_var=getattr(cls, "CONFIG_ENV_VAR", None),
        app_name=cls.CONFIG_APP_NAME,
        app_author=cls.CONFIG_APP_AUTHOR,
        filename=getattr(cls, "CONFIG_FILENAME", "config.json"),
    )