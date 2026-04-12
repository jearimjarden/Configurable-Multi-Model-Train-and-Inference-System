from typing import TypeVar, Type
import pandas as pd
import logging
import uuid
import os
from pathlib import Path
from ..tools.exceptions import ArtifactError, MissingColumns
from ..tools.schemas import Artifact, Config, Metadata
from ..services.IO import create_report, load_metadata, load_artifact
from ..services.data_loader import load_data
from ..services.preprocessor import align_data_schemas
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
        data = self._validate_data(metadata=metadata, data=data)
        self.logger.info("Data are validated and aligned")

        prediction = self.predict(data=data, artifact=artifact)
        self.save_prediction(prediction=prediction)

    def load_metadata(self) -> Metadata:
        metadata = load_metadata(
            load_dir=self.config.inference.load_dir,
            metadata_name=self.config.inference.metadata_name,
        )
        if not hasattr(metadata, "model"):
            raise ArtifactError("Invalid Artifact")

        return metadata

    def load_artifact(self, metadata: Metadata) -> Artifact:
        return load_artifact(
            load_dir=self.config.inference.load_dir,
            artifact_name=metadata.run.artifact_name,
        )

    def load_data(self):
        data = load_data(self.config.data.inference_path)
        return data

    def _validate_data(self, data: pd.DataFrame, metadata: Metadata) -> pd.DataFrame:
        metadata_columns = set(metadata.training.features_col)
        data_columns = set(data.columns)
        missing_columns = metadata_columns - data_columns
        extra_columns = data_columns - metadata_columns

        if not self.config.inference.allow_missing_features and missing_columns:
            raise MissingColumns(
                f"Missing Columns are not allowed, {list(missing_columns)}"
            )
        else:
            data = align_data_schemas(
                data=data,
                extra_columns=extra_columns,
                missing_columns=missing_columns,
                metadata_columns=metadata_columns,
            )
            self.logger.info(
                f"imputed columns: {missing_columns}, dropped features: {extra_columns}"
            )
            return data

    def predict(self, artifact: Artifact, data: pd.DataFrame):
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

        create_report(
            save_name=save_name,
            uuid=str(uuid.uuid4()),
            prediction=prediction,
            metadata_name=self.config.inference.metadata_name,
            allow_missing_features=self.config.inference.allow_missing_features,
            threshold=self.config.inference.threshold,
            save_dir=save_dir,
        )
        self.logger.info(
            f"{save_name} saved in {self.config.inference.inference_report_path}"
        )

    def _artifact_validation(self, artifact: Artifact, metadata: Metadata):
        if artifact["uuid"] != metadata.run.uuid:
            raise ArtifactError("Metadata doesnt match the artifact")

    T = TypeVar("T", bound="InferencePipeline")

    @classmethod
    def from_config(cls: Type[T], config: Config, logger: logging.Logger) -> T:
        return cls(config, logger)
