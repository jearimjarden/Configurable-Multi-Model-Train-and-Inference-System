from typing import TypeVar, Type
import pandas as pd
import logging
import uuid
import os
import json
from pathlib import Path
from ..tools.exceptions import ArtifactError, ColumnsMissingError, InputJSONError
from ..tools.schemas import Artifact, Config, Metadata, create_pydantic_from_metadata
from ..services.IO import (
    create_prediction_report,
    load_metadata,
    load_artifact,
    validate_input,
)
from ..services.data_loader import load_data
from ..services.preprocessor import align_data
from ..services.models import predict_model


class InferencePipeline:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def run(self, input: str = "") -> None:
        metadata, artifact = self.handle_metadata_artifact()
        InputModel = create_pydantic_from_metadata(
            metadata_features_col=metadata.training.features_name_and_type,
            model_name="FeatureColumns",
        )

        if self.config.inference.service:
            prediction_report = self.predict_service(
                input=input,
                metadata=metadata,
                artifact=artifact,
                input_model=InputModel,
            )
            self.logger.debug(f"prediction report from service {prediction_report}")

        else:
            data = self.load_data()
            prediction_report = self.predict_csv(
                data=data, metadata=metadata, artifact=artifact, input_model=InputModel
            )
            self.logger.debug(f"prediction report from data {prediction_report}")

    def predict_service(
        self, input: str, metadata: Metadata, artifact: Artifact, input_model: Type
    ) -> dict:
        aligned_input = self.handle_input(
            metadata=metadata, input_model=input_model, inputs=input
        )
        prediction = self.predict_data(artifact=artifact, data=aligned_input)
        prediction_report = self.create_prediction(prediction=prediction)

        return prediction_report

    def predict_csv(
        self,
        data: pd.DataFrame,
        metadata: Metadata,
        artifact: Artifact,
        input_model: Type,
    ) -> dict:
        aligned_data = self.handle_data(
            data=data, metadata=metadata, input_model=input_model
        )
        prediction = self.predict_data(data=aligned_data, artifact=artifact)
        prediction_report = self.create_prediction(prediction=prediction)
        return prediction_report

    def _from_model_to_df(self, validated_input: list[Type]) -> pd.DataFrame:
        return pd.DataFrame([item.model_dump() for item in validated_input])

    def handle_data(
        self, input_model: Type, data: pd.DataFrame, metadata: Metadata
    ) -> pd.DataFrame:

        dict_data_in_list = self._df_to_dict(data)
        normalized_data = self._normalize_input(
            inputs=dict_data_in_list, metadata=metadata
        )
        validated_input = validate_input(
            normalized_data_input=normalized_data, input_schemas=input_model
        )
        dataframe_data = self._from_model_to_df(validated_input=validated_input)
        aligned_data = self.align_data(data=dataframe_data, metadata=metadata)
        return aligned_data

    def _df_to_dict(self, data: pd.DataFrame) -> list[dict]:
        data = data.astype(object)
        data = data.where(data.notna(), None)
        return data.to_dict(orient="records")

    def handle_input(
        self, metadata: Metadata, input_model: Type, inputs: str
    ) -> pd.DataFrame:
        dict_input_in_list = self._JSON_to_dict(inputs=inputs)
        normalized_input = self._normalize_input(
            inputs=dict_input_in_list, metadata=metadata
        )
        validated_input = validate_input(
            normalized_data_input=normalized_input, input_schemas=input_model
        )
        dataframe_input = self._from_model_to_df(validated_input=validated_input)
        aligned_input = self.align_data(data=dataframe_input, metadata=metadata)
        return aligned_input

    def _normalize_input(self, inputs: list[dict], metadata: Metadata):
        features_meta = metadata.training.features_name_and_type

        for row in inputs:
            for feature, value in row.items():
                feature_meta = features_meta.get(feature)

                if feature_meta is None:
                    continue

                if feature_meta.get("semantic") == "categorial":
                    if isinstance(value, (int, float)):
                        row[feature] = str(int(value))

        return inputs

    def _JSON_to_dict(self, inputs: str):
        try:
            dict_input_in_list = json.loads(inputs)
            if isinstance(dict_input_in_list, dict):
                return [dict_input_in_list]

            if isinstance(dict_input_in_list, list):
                return dict_input_in_list

            return dict_input_in_list
        except json.JSONDecodeError as e:
            raise InputJSONError(f"Invalid JSON input: {e}") from e

    def handle_metadata_artifact(self):
        metadata = self.load_metadata()
        artifact = self.load_artifact(metadata=metadata)
        self._artifact_validation(artifact=artifact, metadata=metadata)
        return metadata, artifact

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
        data = load_data(
            self.config.data.inference_path,
        )
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
        data: pd.DataFrame,
        metadata: Metadata,
        missing_columns: set = set(),
        extra_columns: set = set(),
    ) -> pd.DataFrame:
        data = align_data(
            data=data,
            metadata_columns=metadata.training.features_col,
        )
        self.logger.info(
            f"imputed columns: {missing_columns}, dropped features: {extra_columns}"
        )
        return data

    def predict_data(
        self, artifact: Artifact, data: pd.DataFrame
    ) -> list[tuple[float, int]]:
        return predict_model(
            artifact=artifact,
            data=data,
            threshold=self.config.inference.threshold,
        )

    def create_prediction(self, prediction: list[tuple[float, int]]) -> dict:
        i = 1
        save_dir = self.config.inference.inference_report_path
        save_name = Path(self.config.inference.metadata_name).stem
        while os.path.exists(f"{save_dir}/{save_name}_{i}.json"):
            i += 1
        save_name = f"{save_name}_{i}.json"

        prediction_report = create_prediction_report(
            save_name=save_name,
            str_uuid=self._create_uuid(),
            prediction=prediction,
            metadata_name=self.config.inference.metadata_name,
            allow_missing_features=self.config.inference.allow_missing_features,
            threshold=self.config.inference.threshold,
            save_dir=save_dir,
            save_result=self.config.inference.save_result,
        )
        if self.config.inference.save_result:
            self.logger.info(
                f"{save_name} saved in {self.config.inference.inference_report_path}"
            )
        return prediction_report

    def _create_uuid(self) -> str:
        return str(uuid.uuid4())

    def _artifact_validation(self, artifact: Artifact, metadata: Metadata):
        if artifact["uuid"] != metadata.run.uuid:
            raise ArtifactError("Metadata's UUID doesnt match the artifact's UUID")

    T = TypeVar("T", bound="InferencePipeline")

    @classmethod
    def from_config(cls: Type[T], config: Config, logger: logging.Logger) -> T:
        return cls(config, logger)
