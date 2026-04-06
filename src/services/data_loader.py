from pathlib import Path
from ..tools.exceptions import DataNotExists, DataInvalidExtension, DataEmpty
import pandas as pd


def load_data(path: str):
    data_path = Path(path)
    if not data_path.exists():
        raise DataNotExists("Data Path not Exists")
    if data_path.suffix != ".csv":
        raise DataInvalidExtension("Data File's Extension Must Be .csv")

    try:
        df = pd.read_csv(data_path)
        return df

    except pd.errors.EmptyDataError:
        raise DataEmpty("Data is empty")
