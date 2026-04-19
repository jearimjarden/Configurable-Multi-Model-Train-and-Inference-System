from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
from src.tools.schemas import InferenceStrategy, StagePipeline
from src.tools.exceptions import DataError


def create_preprocessor(
    X: pd.DataFrame, missing_strategy: InferenceStrategy
) -> ColumnTransformer:
    num_columns = X.select_dtypes(exclude=["object", "category"]).columns
    cat_columns = X.select_dtypes(include=["object", "category"]).columns

    if missing_strategy == "constant":
        cat_imputer = SimpleImputer(strategy="constant", fill_value="MISSING")
    elif missing_strategy == "most_frequent":
        cat_imputer = SimpleImputer(strategy="most_frequent")

    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
    )
    cat_pipeline = Pipeline(
        [
            ("imputer", cat_imputer),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [("num", num_pipeline, num_columns), ("cat", cat_pipeline, cat_columns)]
    )


def split_data(
    df: pd.DataFrame, target_col: str, positif_value: str, drop_features: list
) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns.tolist():
        raise DataError(
            f"Target Columns: '{target_col}' does not exist",
            stage=StagePipeline.DATA_ALIGNMENT,
        )
    if not set(drop_features).issubset(set(df)):
        raise DataError(
            f"Drop_features: '{drop_features}' does not exist",
            stage=StagePipeline.DATA_ALIGNMENT,
        )
    if positif_value not in df[target_col].unique():
        raise DataError(
            f"Positive value: '{positif_value}' does not exist in target col: '{target_col}'",
            stage=StagePipeline.DATA_ALIGNMENT,
        )

    unique_target_value = df[target_col].dropna().unique().tolist()
    if len(unique_target_value) > 2:
        raise DataError(
            f"Can not accept more than 2 unique value in target columns, n of unique value: '{len(unique_target_value)}'",
            stage=StagePipeline.DATA_ALIGNMENT,
        )

    negative_value = list(set(unique_target_value) - set(positif_value))
    y = df[target_col].map({positif_value: 1, negative_value[0]: 0})

    features_to_drop = drop_features + [target_col]
    X = df.drop(features_to_drop, axis=1)
    return (X, y)


def align_data(
    data: pd.DataFrame,
    metadata_columns: list,
) -> pd.DataFrame:
    index = data["data_id"]

    data_unindexed = data.drop(columns=["data_id"])
    data_unindexed = data_unindexed[metadata_columns]

    data_unindexed.insert(0, "data_id", index.to_numpy())

    return data_unindexed
