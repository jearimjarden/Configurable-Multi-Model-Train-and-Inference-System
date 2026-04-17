from pathlib import Path
import pandas as pd

from ..tools.schemas import StagePipeline
from ..tools.exceptions import (
    DataNotExistsError,
    DataInvalidError,
)


def load_data(path: str) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise DataNotExistsError(
            f"Cannot find data at '{data_path.resolve()}'",
            stage=StagePipeline.DATA_LOADING,
        )

    if data_path.suffix != ".csv":
        raise DataInvalidError(
            f"Do not support '{data_path.suffix}' data's extension, input data extension should be '.csv'",
            stage=StagePipeline.DATA_LOADING,
        )

    try:
        df = pd.read_csv(data_path)
        return df

    except pd.errors.EmptyDataError:
        raise DataInvalidError(
            "Loaded data cannot be empty", stage=StagePipeline.DATA_LOADING
        )
