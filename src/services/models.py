from ..tools.exceptions import MetricsInvalid, ParamInvalid
from ..tools.schemas import Artifact, ConfigTrainModel, FittedModel
from sklearn.compose import ColumnTransformer
from typing import Type
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


def select_model(
    preprocessor: ColumnTransformer,
    model: Type[BaseEstimator],
    params: dict[str, int],
    random_seed: int,
) -> Pipeline:
    if model in [RandomForestClassifier, DecisionTreeClassifier]:
        params["random_state"] = random_seed
    try:
        return Pipeline([("preprocessor", preprocessor), ("model", model(**params))])
    except TypeError:
        raise ParamInvalid("Invalid Model Parameter")


def cv_params(stratify, random_seed, n_cv) -> StratifiedKFold | KFold:
    if stratify:
        return StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=random_seed)
    else:
        return KFold(n_splits=n_cv, shuffle=True, random_state=random_seed)


def cross_validation(
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

    for key, value in models.items():
        try:
            scores = cross_validate(
                select_model(
                    preprocessor=preprocessor,
                    model=value.type.value,
                    params=value.params,
                    random_seed=random_seed,
                ),
                X=X,
                y=y,
                scoring=[selection_metrics],
                cv=cv_params(stratify=stratify, random_seed=random_seed, n_cv=n_cv),
                return_train_score=True,
            )

            report[key] = {
                "train_{}".format(selection_metrics): float(
                    scores[("train_{}".format(selection_metrics))].mean()
                ),
                "test_{}".format(selection_metrics): float(
                    scores[("test_{}".format(selection_metrics))].mean()
                ),
                "gap_{}".format(selection_metrics): float(
                    scores[("train_{}".format(selection_metrics))].mean()
                    - scores[("test_{}".format(selection_metrics))].mean()
                ),
                "{}_std".format(selection_metrics): float(
                    scores[("test_{}".format(selection_metrics))].std()
                ),
            }
        except ValueError as e:
            if "_validate_params" in str(e):
                raise ParamInvalid("Invalid Model Parameter's Value")

            elif "not a valid scoring value" in str(e):
                raise MetricsInvalid(
                    "Selection Metric: {} is Invalid".format(selection_metrics)
                )

    return report


def fit_model(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    y: pd.Series,
    random_seed: int,
    models: dict[str, ConfigTrainModel],
) -> list[FittedModel]:
    fitted_model = []
    for key, value in models.items():
        model = select_model(
            preprocessor=preprocessor,
            model=value.type.value,
            params=value.params,
            random_seed=random_seed,
        )

        fitted_model.append(dict(name=key, model=model.fit(X, y)))

    return fitted_model


def predict_model(
    artifact: Artifact, data: pd.DataFrame, threshold: float
) -> list[tuple[float, int]]:
    proba = artifact["pipeline"].predict_proba(data)[:, 1]
    pred = (proba >= threshold).astype(int)
    return list(zip(proba.tolist(), pred.tolist()))
