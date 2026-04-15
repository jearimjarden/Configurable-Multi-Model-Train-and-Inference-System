from typing import Type
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from ..tools.exceptions import MetricsInvalidError, ParamsInvalidError
from ..tools.schemas import Artifact, ConfigTrainModel, FittedModelPipeline


def select_model(
    name: str,
    preprocessor: ColumnTransformer,
    model: Type[BaseEstimator],
    params: dict[str, int],
    random_seed: int,
) -> Pipeline:
    if model in [RandomForestClassifier, DecisionTreeClassifier]:
        params["random_state"] = random_seed
    try:
        return Pipeline([("preprocessor", preprocessor), ("model", model(**params))])

    except TypeError as e:
        raise ParamsInvalidError(f"Invalid parameter for '{name}': '{params}'") from e


def select_cv_params(
    stratify: bool, random_seed: int, n_cv: int
) -> StratifiedKFold | KFold:
    if stratify:
        return StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=random_seed)
    else:
        return KFold(n_splits=n_cv, shuffle=True, random_state=random_seed)


def cross_validate_data(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    y: pd.Series,
    n_cv: int,
    random_seed: int,
    selection_metrics: str,
    stratify: bool,
    models: dict[str, ConfigTrainModel],
) -> dict[str, dict]:
    report = {}

    for name, model in models.items():
        try:
            pipeline = select_model(
                name=name,
                preprocessor=preprocessor,
                model=model.type.value,
                params=model.params,
                random_seed=random_seed,
            )
            scores = cross_validate(
                pipeline,
                X=X,
                y=y,
                scoring=[selection_metrics],
                cv=select_cv_params(
                    stratify=stratify, random_seed=random_seed, n_cv=n_cv
                ),
                return_train_score=True,
            )

            report[name] = {
                f"train_{selection_metrics}": round(
                    scores[("train_{}".format(selection_metrics))].mean(), 3
                ),
                f"test_{selection_metrics}": round(
                    scores[("test_{}".format(selection_metrics))].mean(), 3
                ),
                f"gap_{selection_metrics}": round(
                    (
                        scores[("train_{}".format(selection_metrics))].mean()
                        - scores[("test_{}".format(selection_metrics))].mean()
                    ),
                    4,
                ),
                "f{selection_metrics}_std": round(
                    scores[("test_{}".format(selection_metrics))].std(), 3
                ),
            }

        except ValueError as e:
            if "not a valid scoring value" in str(e):
                raise MetricsInvalidError(
                    f"Selection Metric: {selection_metrics} is invalid"
                ) from e
            raise

    return report


def fit_model(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    y: pd.Series,
    random_seed: int,
    models: dict[str, ConfigTrainModel],
) -> list[FittedModelPipeline]:
    fitted_model_pipeline = []

    for name, model in models.items():
        selected_model = select_model(
            name=name,
            preprocessor=preprocessor,
            model=model.type.value,
            params=model.params,
            random_seed=random_seed,
        )

        fitted_model_pipeline.append(dict(name=name, model=selected_model.fit(X, y)))

    return fitted_model_pipeline


def predict_model(
    artifact: Artifact, data: pd.DataFrame, threshold: float
) -> list[tuple[int, float, int]]:
    index = data["data_id"]
    X = data.drop(columns=["data_id"])
    proba = artifact["pipeline"].predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return list(zip(index.tolist(), proba.tolist(), pred.tolist()))
