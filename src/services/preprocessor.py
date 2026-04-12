from ..tools.schemas import InferenceStrategy
from ..tools.exceptions import DropColumnsInvalid, TargetColumnInvalid, TrueValueInvalid
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
    df: pd.DataFrame, target_col: str, true_value: str, drop_features: list
) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns.tolist():
        raise TargetColumnInvalid("Target Columns Does not Exist")
    if not set(drop_features).issubset(set(df)):
        raise DropColumnsInvalid("Drop Features are not Exist in Training Dataset")
    if true_value not in df[target_col].unique():
        raise TrueValueInvalid("True Value Does not Exist in Target Column")

    unique_value = df[target_col].dropna().unique().tolist()
    if len(unique_value) > 2:
        raise TargetColumnInvalid("Target Column's Value Is not Binary")

    false_value = list(set(unique_value) - set(true_value))
    y = df[target_col].map({true_value: 1, false_value[0]: 0})

    features_to_drop = drop_features + [target_col]
    X = df.drop(features_to_drop, axis=1)
    return (X, y)


def align_data_schemas(
    data: pd.DataFrame, extra_columns: set, missing_columns: set, metadata_columns: list
) -> pd.DataFrame:
    data = data.drop(list(extra_columns), axis=1)
    for column in list(missing_columns):
        data[column] = None
    data = data[metadata_columns]
    return data
