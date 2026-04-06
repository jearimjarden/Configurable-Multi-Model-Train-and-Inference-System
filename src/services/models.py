from sklearn.compose import ColumnTransformer
from typing import Type
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from ..tools.schemas import ConfigTrainModel
import pandas as pd


def select_model(
    preprocessor: ColumnTransformer,
    model: Type[BaseEstimator],
    params: dict,
    random_seed: int,
):
    if model in [RandomForestClassifier, DecisionTreeClassifier]:
        params["random_state"] = random_seed
    return Pipeline([("preprocessor", preprocessor), ("model", model(**params))])


def cv_params(stratify, random_seed, n_cv):
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
):
    report = {}

    for key, value in models.items():
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

        # report.append(
        #     {
        #         "name": key,
        #         "train_{}".format(selection_metrics): float(
        #             scores[("train_{}".format(selection_metrics))].mean()
        #         ),
        #         "test_{}".format(selection_metrics): float(
        #             scores[("test_{}".format(selection_metrics))].mean()
        #         ),
        #         "gap_{}".format(selection_metrics): float(
        #             scores[("train_{}".format(selection_metrics))].mean()
        #             - scores[("test_{}".format(selection_metrics))].mean()
        #         ),
        #         "{}_std".format(selection_metrics): float(
        #             scores[("test_{}".format(selection_metrics))].std()
        #         ),
        #     }
        # )

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

    return report


def fit_model(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    y: pd.Series,
    random_seed: int,
    models: dict[str, ConfigTrainModel],
):
    fitted_model = {}
    for key, value in models.items():
        model = select_model(
            preprocessor=preprocessor,
            model=value.type.value,
            params=value.params,
            random_seed=random_seed,
        )

        fitted_model[key] = model.fit(X, y)

    return fitted_model
