from sklearn.compose import ColumnTransformer
from typing import TypeVar, Type
import logging
import uuid
import pandas as pd
import time
from src.tools.exceptions import DataError, LoggedError, TrainingError
from src.tools.schemas import Config, FittedModelPipeline, Settings, StagePipeline
from src.data.data_loader import load_data
from src.services.preprocessor import create_preprocessor, split_data
from src.services.models import cross_validate_data, fit_model
from src.io.metadata_io import create_metadata
from src.io.artifact_io import create_artifact
from src.data.semantic import infer_semantic


class TrainingPipeline:
    def __init__(self, config: Config, logger: logging.Logger, settings: Settings):
        self.config = config
        self.logger = logger
        self.settings = settings

    def run(self) -> None:
        try:
            data, total_rows, total_cols = self.load_data()
            self.logger.info(
                "Training data successfuly loaded",
                extra={
                    "stage": StagePipeline.DATA_LOADING,
                    "data_path": self.config.data.train_path,
                    "total_rows": total_rows,
                    "total_cols": total_cols,
                },
            )

            X, y = self.split_data(data=data)
            preprocessor = self.create_preprocessor(X=X)
            label_ratio = f"1:{round(((y.value_counts()[0]+y.value_counts()[1])/y.value_counts()[1]),2)}"

            self.logger.info(
                "Data aligned and preprocessor pipeline created",
                extra={
                    "stage": StagePipeline.DATA_ALIGNMENT,
                    "label_ratio": label_ratio,
                    "cat_columns": X.select_dtypes(
                        include=["object", "category"]
                    ).columns.tolist(),
                    "num_columns": X.select_dtypes(
                        exclude=["object", "category"]
                    ).columns.tolist(),
                },
            )

            evaluation_report = self.evaluate_models(
                y=y, X=X, preprocessor=preprocessor
            )
            best_model = self._select_best_model(evaluation_report=evaluation_report)
            self.logger.info(
                "Model evaluation completed",
                extra={
                    "stage": StagePipeline.EVALUATION,
                    "best_model": best_model,
                    self.config.train.selection_metrics: evaluation_report[best_model][
                        f"test_{self.config.train.selection_metrics}"
                    ],
                },
            )
            _start_time = time.perf_counter()
            fitted_pipelines = self.train_models(preprocessor=preprocessor, X=X, y=y)
            self.logger.info(
                f"Successfully trained {len(self.config.train.model)} model",
                extra={
                    "stage": StagePipeline.MODEL_FITTING,
                    "training_time": f"{round(time.perf_counter() - _start_time, 2)}s",
                    "models": [
                        {
                            "type": model,
                            "params": self.config.train.model[model].params,
                        }
                        for model in self.config.train.model
                    ],
                },
            )

            self.handle_artifact_and_metadata(
                y=y,
                fitted_pipelines=fitted_pipelines,
                X=X,
                evaluation_report=evaluation_report,
                best_model_name=best_model,
            )

        except (DataError, TrainingError) as e:
            self.logger.error(str(e), extra={"stage": e.stage})
            raise LoggedError from e

    def load_data(self) -> tuple[pd.DataFrame, int, int]:
        data = load_data(self.config.data.train_path)
        total_rows = data.shape[0]
        total_columns = data.shape[1]
        return data, total_rows, total_columns

    def split_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X, y = split_data(
            df=data,
            target_col=self.config.train.target_col,
            positif_value=self.config.train.true_value,
            drop_features=self.config.train.drop_features,
        )
        return (X, y)

    def create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        preprocessor = create_preprocessor(
            X=X, missing_strategy=self.config.train.missing_strategy
        )
        return preprocessor

    def evaluate_models(
        self, y: pd.Series, X: pd.DataFrame, preprocessor: ColumnTransformer
    ) -> dict[str, dict]:
        evaluation_report = cross_validate_data(
            preprocessor=preprocessor,
            X=X,
            y=y,
            n_cv=self.config.train.n_cv,
            random_seed=self.config.train.random_seed,
            selection_metrics=self.config.train.selection_metrics,
            stratify=self.config.train.stratify,
            models=self.config.train.model,
        )
        return evaluation_report

    def train_models(
        self, preprocessor: ColumnTransformer, X: pd.DataFrame, y: pd.Series
    ) -> list[FittedModelPipeline]:
        fitted_model = fit_model(
            preprocessor=preprocessor,
            X=X,
            random_seed=self.config.train.random_seed,
            models=self.config.train.model,
            y=y,
        )
        return fitted_model

    def handle_artifact_and_metadata(
        self,
        fitted_pipelines: list[FittedModelPipeline],
        y: pd.Series,
        X: pd.DataFrame,
        evaluation_report: dict[str, dict],
        best_model_name: str,
    ) -> None:
        saved_artifacts = []
        saved_metadatas = []
        if not self.config.artifact.only_best:
            for pipeline in fitted_pipelines:
                str_uuid = self._create_uuid()
                save_name = (
                    f"best_{pipeline['name']}"
                    if pipeline["name"] == best_model_name
                    else pipeline["name"]
                )
                save_artifact_name = save_name + ".pkl"
                save_metadata_name = save_name + ".json"

                self.save_artifact(
                    fitted_pipeline=pipeline,
                    uuid=str_uuid,
                    save_name=save_artifact_name,
                )
                self.save_metadata(
                    y=y,
                    X=X,
                    evaluation_report=evaluation_report[pipeline["name"]],
                    model_name=pipeline["name"],
                    uuid=str_uuid,
                    save_metadata_name=save_metadata_name,
                    save_artifact_name=save_artifact_name,
                )
                saved_metadatas.append(save_metadata_name)
                saved_artifacts.append(save_artifact_name)

        else:
            str_uuid = self._create_uuid()
            pipeline = next(
                pipeline
                for pipeline in fitted_pipelines
                if pipeline["name"] == best_model_name
            )
            save_name = f"best_{pipeline['name']}"
            save_artifact_name = save_name + ".pkl"
            save_metadata_name = save_name + ".json"

            self.save_artifact(
                fitted_pipeline=pipeline, uuid=str_uuid, save_name=save_artifact_name
            )
            self.save_metadata(
                y=y,
                X=X,
                evaluation_report=evaluation_report[pipeline["name"]],
                model_name=pipeline["name"],
                uuid=str_uuid,
                save_metadata_name=save_metadata_name,
                save_artifact_name=save_artifact_name,
            )
            saved_metadatas.append(save_metadata_name)
            saved_artifacts.append(save_artifact_name)

        self.logger.info(
            "Saved Metadata and Artifact",
            extra={
                "stage": StagePipeline.TRAINING,
                "save_only_best": self.config.artifact.only_best,
                "saved_metadata": saved_metadatas,
                "saved_artifacts": saved_artifacts,
            },
        )

    def save_artifact(
        self, fitted_pipeline: FittedModelPipeline, uuid: str, save_name: str
    ) -> None:
        create_artifact(
            save_name=save_name,
            fitted_pipeline=fitted_pipeline,
            save_dir=self.config.artifact.save_dir,
            uuid=uuid,
        )

    def save_metadata(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        evaluation_report: dict[str, dict],
        model_name: str,
        uuid: str,
        save_metadata_name: str,
        save_artifact_name: str,
    ) -> None:
        semantic_columns = infer_semantic(X=X)

        features_name_and_type = {
            str(col): {"type": str(type)} for col, type in X.dtypes.items()
        }

        for feature in features_name_and_type:
            features_name_and_type[feature]["semantic"] = semantic_columns[feature]

        create_metadata(
            class_ratio=f"1:{1/(y.value_counts()[1]/(y.value_counts()[1]+y.value_counts()[0]))}",
            save_metadata_name=save_metadata_name,
            save_artifact_name=save_artifact_name,
            str_uuid=uuid,
            save_dir=self.config.artifact.save_dir,
            features_name_and_type=features_name_and_type,
            features_col=X.columns.tolist(),
            evaluation_report=evaluation_report,
            target_columns=self.config.train.target_col,
            train_data=self.config.data.train_path,
            n_samples=X.shape[0],
            stratify=self.config.train.stratify,
            random_seed=self.config.train.random_seed,
            model_name=model_name,
            models=self.config.train.model,
        )

    def _create_uuid(self):
        return str(uuid.uuid4())

    def _select_best_model(self, evaluation_report: dict[str, dict]) -> str:
        return max(
            evaluation_report,
            key=lambda x: evaluation_report[x][
                "test_{}".format(self.config.train.selection_metrics)
            ],
        )

    T = TypeVar("T", bound="TrainingPipeline")

    @classmethod
    def from_config(
        cls: Type[T], config: Config, logger: logging.Logger, settings: Settings
    ) -> T:
        return cls(config, logger, settings)
