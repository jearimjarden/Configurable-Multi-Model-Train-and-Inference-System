from sklearn.compose import ColumnTransformer
from typing import TypeVar, Type
import logging
import uuid
import pandas as pd
from ..tools.schemas import Config, FittedModel
from ..services.data_loader import load_data
from ..services.preprocessor import create_preprocessor, split_data
from ..services.models import cross_validation, fit_model
from ..services.IO import create_artifact, create_metadata


class TrainingPipeline:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def run(self):
        data, total_rows, total_cols = self.load_data()
        self.logger.info(
            f"Training data loaded from {self.config.data.train_path} with {total_rows} rows and {total_cols} columns"
        )

        (X, y), preprocessor = self.preprocess(data=data)
        self.logger.info("Data preprocessed")

        report = self.evaluate(y=y, X=X, preprocessor=preprocessor)
        self.logger.info(
            f"Evaluated with n_fold: {self.config.train.n_cv}, selection_metrics: {self.config.train.selection_metrics}, stratify: {self.config.train.stratify}, random_seed: {self.config.train.random_seed}"
        )

        fitted_model = self.fit_model(preprocessor=preprocessor, X=X, y=y)
        self.logger.info(
            f"Successfully fit model: {[model["name"] for model in fitted_model]}"
        )

        best_model = self._best_model(report=report)
        self._artifact_handler(
            fitted_model=fitted_model, X=X, report=report, best_model=best_model
        )
        self.logger.info(
            f"Artifact and metadata saved in '{self.config.artifact.save_dir}', saved best only: {self.config.artifact.only_best}"
        )

    def load_data(self) -> tuple[pd.DataFrame, int, int]:
        data = load_data(self.config.data.train_path)
        total_rows = data.shape[0]
        total_columns = data.shape[1]
        return data, total_rows, total_columns

    def preprocess(
        self, data: pd.DataFrame
    ) -> tuple[tuple[pd.DataFrame, pd.Series], ColumnTransformer]:
        X, y = split_data(
            df=data,
            target_col=self.config.train.target_col,
            true_value=self.config.train.true_value,
            drop_features=self.config.train.drop_features,
        )
        preprocessor = create_preprocessor(
            X=X, missing_strategy=self.config.train.missing_strategy
        )
        return ((X, y), preprocessor)

    def evaluate(
        self, y: pd.Series, X: pd.DataFrame, preprocessor: ColumnTransformer
    ) -> dict[str, dict]:
        report = cross_validation(
            preprocessor=preprocessor,
            X=X,
            y=y,
            n_cv=self.config.train.n_cv,
            random_seed=self.config.train.random_seed,
            selection_metrics=self.config.train.selection_metrics,
            stratify=self.config.train.stratify,
            models=self.config.train.model,
        )
        return report

    def fit_model(
        self, preprocessor: ColumnTransformer, X: pd.DataFrame, y: pd.Series
    ) -> list[FittedModel]:
        fitted_model = fit_model(
            preprocessor=preprocessor,
            X=X,
            random_seed=self.config.train.random_seed,
            models=self.config.train.model,
            y=y,
        )
        return fitted_model

    def _artifact_handler(
        self,
        fitted_model: list[FittedModel],
        X: pd.DataFrame,
        report: dict[str, dict],
        best_model: str,
    ):
        if not self.config.artifact.only_best:
            for model in fitted_model:
                id = str(uuid.uuid4())
                save_name = (
                    f"best_{model['name']}"
                    if model["name"] == best_model
                    else model["name"]
                )
                self.save_artifact(fitted_model=model, uuid=id, save_name=save_name)
                self.save_metadata(
                    X=X,
                    report=report[model["name"]],
                    model_name=model["name"],
                    uuid=id,
                    save_name=save_name,
                )
        else:
            id = str(uuid.uuid4())
            model = next(model for model in fitted_model if model["name"] == best_model)
            save_name = f"best_{model['name']}"
            self.save_artifact(fitted_model=model, uuid=id, save_name=save_name)
            self.save_metadata(
                X=X,
                report=report[model["name"]],
                model_name=model["name"],
                uuid=id,
                save_name=save_name,
            )

    def save_artifact(
        self, fitted_model: FittedModel, uuid: str, save_name: str
    ) -> None:
        create_artifact(
            save_name=save_name,
            fitted_model=fitted_model,
            save_dir=self.config.artifact.save_dir,
            uuid=uuid,
        )

    def save_metadata(
        self,
        X: pd.DataFrame,
        report: dict[str, dict],
        model_name: str,
        uuid: str,
        save_name: str,
    ) -> None:
        create_metadata(
            save_name=save_name,
            uuid=uuid,
            save_dir=self.config.artifact.save_dir,
            features_col=X.columns.tolist(),
            report=report,
            target_col=self.config.train.target_col,
            train_data=self.config.data.train_path,
            n_samples=X.shape[0],
            stratify=self.config.train.stratify,
            random_seed=self.config.train.random_seed,
            model_name=model_name,
            models=self.config.train.model,
        )

    def _create_uuid(self):
        return str(uuid.uuid4())

    def _best_model(self, report: dict[str, dict]) -> str:
        return max(
            report,
            key=lambda x: report[x][
                "test_{}".format(self.config.train.selection_metrics)
            ],
        )

    T = TypeVar("T", bound="TrainingPipeline")

    @classmethod
    def from_config(cls: Type[T], config: Config, logger: logging.Logger) -> T:
        return cls(config, logger)
