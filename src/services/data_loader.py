from pathlib import Path
import pandas as pd
from ..tools.exceptions import DataNotExists, DataInvalidExtension, DataEmpty


def load_data(path: str) -> pd.DataFrame:
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
