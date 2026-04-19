from src.tools.exceptions import DataInvalidError, DataNotExistsError
from src.data.data_loader import load_data
import pytest


def test_file_not_exist_error():
    wrong_path = "i_dont_know.csv"
    with pytest.raises(DataNotExistsError):
        load_data(wrong_path)


def test_extension_error():
    wrong_path = "tests/i_dont_know.docs"
    with pytest.raises(DataInvalidError):
        load_data(wrong_path)
