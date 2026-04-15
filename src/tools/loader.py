from pydantic import ValidationError
from pathlib import Path
import yaml
from .exceptions import (
    ConfigInvalidError,
    ConfigNotExistsError,
    SettingsInvalidError,
)
from ..tools.schemas import Config, Settings


def load_config() -> Config:
    config_path = Path("config.yaml")

    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        return Config(**data)

    except FileNotFoundError as e:
        raise ConfigNotExistsError(
            f"Cannot find 'config.yaml' at: {config_path.resolve()}"
        ) from e

    except ValidationError as e:
        messages = []

        for err in e.errors():
            field = ".".join(str(x) for x in err["loc"])

            if err["type"] == "missing":
                messages.append(f"Missing config parameter: '{field}'")

            elif err["type"] == "extra_forbidden":
                messages.append(f"Forbidden extra config parameter: '{field}'")

            else:
                messages.append(f"Invalid value for '{field}': {err['msg']}")

        raise ConfigInvalidError(" | ".join(messages)) from e


def load_settings():
    try:
        env = Settings.load()
        return env

    except ValidationError as e:
        messages = []

        for err in e.errors():
            field = ".".join(str(x) for x in err["loc"])
            if err["type"] == "missing":
                messages.append(f"Missing env parameter: '{field}'")
            elif err["type"] == "extra_forbidden":
                messages.append(f"Forbidden extra config parameter: '{field}'")
            else:
                messages.append((f"Invalid value for '{field}': {err['msg']}"))

        raise SettingsInvalidError(" | ".join(messages)) from e
