from typing import TypeVar, Type
import pandas as pd
import logging
import uuid
import os
from pathlib import Path
from ..tools.exceptions import ArtifactError, ColumnsMissingError
from ..tools.schemas import Artifact, Config, Metadata
from ..services.IO import create_prediction_report, load_metadata, load_artifact
from ..services.data_loader import load_data
from ..services.preprocessor import align_data
from ..services.models import predict_model


class InferencePipeline:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def run(self):
        metadata = self.load_metadata()
        artifact = self.load_artifact(metadata=metadata)
        self._artifact_validation(artifact=artifact, metadata=metadata)
        self.logger.info("Metadata and artifact loaded and validated")

        data = self.load_data()
        missing_columns, extra_columns = self._validate_data(
            metadata=metadata, data=data
        )
        aligned_data = self.align_data(
            missing_columns=missing_columns,
            extra_columns=extra_columns,
            data=data,
            metadata=metadata,
        )
        self.logger.info("Data are validated and aligned")

        prediction = self.predict_data(data=aligned_data, artifact=artifact)
        self.save_prediction(prediction=prediction)

    def load_metadata(self) -> Metadata:
        metadata = load_metadata(
            load_dir=self.config.inference.load_dir,
            metadata_name=self.config.inference.metadata_name,
        )
        if not hasattr(metadata, "model"):
            raise ArtifactError(
                f"Artifact: {self.config.inference.metadata_name} does not contain fitted pipeline"
            )

        return metadata

    def load_artifact(self, metadata: Metadata) -> Artifact:
        return load_artifact(
            load_dir=self.config.inference.load_dir,
            artifact_name=metadata.run.artifact_name,
        )

    def load_data(self):
        data = load_data(self.config.data.inference_path)
        return data

    def _validate_data(self, data: pd.DataFrame, metadata: Metadata) -> tuple[set, set]:
        metadata_columns = set(metadata.training.features_col)
        data_columns = set(data.columns)
        missing_columns = metadata_columns - data_columns
        extra_columns = data_columns - metadata_columns

        if not self.config.inference.allow_missing_features and missing_columns:
            raise ColumnsMissingError(
                f"Missing Columns are not allowed: {list(missing_columns)}"
            )

        return missing_columns, extra_columns

    def align_data(
        self,
        missing_columns: set,
        extra_columns: set,
        data: pd.DataFrame,
        metadata: Metadata,
    ) -> pd.DataFrame:
        data = align_data(
            data=data,
            extra_columns=extra_columns,
            missing_columns=missing_columns,
            metadata_columns=metadata.training.features_col,
        )
        self.logger.info(
            f"imputed columns: {missing_columns}, dropped features: {extra_columns}"
        )
        return data

    def predict_data(self, artifact: Artifact, data: pd.DataFrame):
        return predict_model(
            artifact=artifact,
            data=data,
            threshold=self.config.inference.threshold,
        )

    def save_prediction(self, prediction: list[tuple[float, int]]) -> None:
        i = 1
        save_dir = self.config.inference.inference_report_path
        save_name = Path(self.config.inference.metadata_name).stem
        while os.path.exists(f"{save_dir}/{save_name}_{i}.json"):
            i += 1
        save_name = f"{save_name}_{i}.json"

        create_prediction_report(
            save_name=save_name,
            str_uuid=self._create_uuid(),
            prediction=prediction,
            metadata_name=self.config.inference.metadata_name,
            allow_missing_features=self.config.inference.allow_missing_features,
            threshold=self.config.inference.threshold,
            save_dir=save_dir,
        )
        self.logger.info(
            f"{save_name} saved in {self.config.inference.inference_report_path}"
        )

    def _create_uuid(self) -> str:
        return str(uuid.uuid4())

    def _artifact_validation(self, artifact: Artifact, metadata: Metadata):
        if artifact["uuid"] != metadata.run.uuid:
            raise ArtifactError("Metadata's UUID doesnt match the artifact's UUID")

    T = TypeVar("T", bound="InferencePipeline")

    @classmethod
    def from_config(cls: Type[T], config: Config, logger: logging.Logger) -> T:
        return cls(config, logger)
