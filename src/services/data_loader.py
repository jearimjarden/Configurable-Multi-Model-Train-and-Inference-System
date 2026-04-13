from pathlib import Path
import pandas as pd
from ..tools.exceptions import DataNotExistsError, DataExtensionError, DataEmptyError


def load_data(path: str) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise DataNotExistsError(f"Cannot find data at '{data_path.resolve()}'")

    if data_path.suffix != ".csv":
        raise DataExtensionError(
            f"Do not support '{data_path.suffix}' data's extension, input data extension should be '.csv'"
        )

    try:
        df = pd.read_csv(data_path)
        return df

    except pd.errors.EmptyDataError:
        raise DataEmptyError("Loaded data cannot be empty")
