from pydantic import ValidationError
from .exceptions import ConfigInvalid, ConfigNotExists
from ..tools.schemas import Config
from pathlib import Path
import yaml


def config_load():
    path = Path("config.yaml")
    if not path.exists():
        raise ConfigNotExists("Missing config.yaml File")

    with open("config.yaml", "r") as f:
        data = yaml.safe_load(f)
        try:
            return Config(**data)

        except (ValidationError, KeyError):
            raise ConfigInvalid("Invalid Config Structure or Value")
