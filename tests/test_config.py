from src.tools.exceptions import ConfigNotExistsError
from src.tools.loader import load_config, load_settings
from src.tools.schemas import Config, Settings
import pytest


def test_load_config():
    config = load_config("config.yaml")
    assert isinstance(config, Config)


def test_load_settings():
    settings = load_settings(".env")
    assert isinstance(settings, Settings)


def test_bad_config():
    with pytest.raises(ConfigNotExistsError):
        load_config("i_dont_know.yaml")
