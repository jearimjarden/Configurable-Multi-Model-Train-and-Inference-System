from pydantic import ValidationError
from pathlib import Path
import yaml
from src.tools.exceptions import (
    ConfigInvalidError,
    ConfigNotExistsError,
    SettingsInvalidError,
)
from src.tools.schemas import Config, Settings, StagePipeline


def load_config(path: str) -> Config:
    BASE_DIR = Path(__file__).resolve().parents[2]
    config_path = BASE_DIR / path

    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        return Config(**data)

    except FileNotFoundError as e:
        raise ConfigNotExistsError(
            f"Cannot find 'config.yaml' at: {config_path.resolve()}",
            stage=StagePipeline.CONFIGURATION_LOADING,
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

        raise ConfigInvalidError(
            " | ".join(messages), stage=StagePipeline.CONFIGURATION_LOADING
        ) from e


def load_settings(path: str) -> Settings:
    try:
        env = Settings.load(path)
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

        raise SettingsInvalidError(
            " | ".join(messages), stage=StagePipeline.CONFIGURATION_LOADING
        ) from e
