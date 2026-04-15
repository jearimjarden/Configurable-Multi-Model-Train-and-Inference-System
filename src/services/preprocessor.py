from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
from ..tools.schemas import InferenceStrategy
from ..tools.exceptions import DropColumnsError, TargetColumnError, PositiveValueError


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
        raise TargetColumnError("Target Columns Does not Exist")
    if not set(drop_features).issubset(set(df)):
        raise DropColumnsError("Drop Features are not Exist in Training Dataset")
    if positif_value not in df[target_col].unique():
        raise PositiveValueError("True Value Does not Exist in Target Column")

    unique_target_value = df[target_col].dropna().unique().tolist()
    if len(unique_target_value) > 2:
        raise TargetColumnError("Target Column's Value Is not Binary")

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
