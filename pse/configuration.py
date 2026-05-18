from __future__ import annotations

from dataclasses import dataclass, field
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
    optimizer: str = field(
        default = 'gpcam',
        metadata={"config_groups": ["pse"]}
    )
    acquisition_function: str = field(
        default = 'variance',
        metadata={"config_groups": ["pse"]}
    )
    gp_iterations: int = field(
        default = 50,
        metadata={"config_groups": ["pse"]}
    )
    initial_iterations: int = field(
        default = 10,
        metadata={"config_groups": ["pse"]}
    )
    client: str = field(
        default = 'Test Ackley Function',
        metadata={"config_groups": ["pse"]}
    )
    parallel_measurements: int = field(
        default = 1,
        metadata={"config_groups": ["pse"]}
    )
    pse_opt_pars: dict = field(
        default_factory = dict,
        metadata={"config_groups": ["pse"]}
    )

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