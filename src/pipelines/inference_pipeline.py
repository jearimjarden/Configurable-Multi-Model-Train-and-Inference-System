from typing import TypeVar, Type
import pandas as pd
import logging
import uuid
import os
import json
from pathlib import Path
import time
from ..tools.exceptions import ArtifactError, InputJSONError
from ..tools.schemas import (
    Artifact,
    Config,
    Metadata,
    PredictionReport,
    Settings,
    StagePipeline,
    create_pydantic_from_metadata,
)
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
    def __init__(self, config: Config, logger: logging.Logger, settings: Settings):
        self.config = config
        self.logger = logger
        self.settings = settings

    def run(self, input: str | pd.DataFrame = "") -> PredictionReport:
        start_time = time.perf_counter()
        metadata, artifact = self.handle_metadata_artifact()
        InputModel = create_pydantic_from_metadata(
            metadata_features_col=metadata.training.features_name_and_type,
            model_name="FeatureColumns",
        )

        _schema = {
            field_name: field_type.annotation
            for field_name, field_type in InputModel.model_fields.items()
        }
        self.logger.debug(
            "Created input model type",
            extra={"stage": StagePipeline.INFERENCE, "schema": _schema},
        )

        if self.settings.predict_service:
            prediction_report = self.predict_service(
                input=input,
                metadata=metadata,
                artifact=artifact,
                input_model=InputModel,
            )

        elif not self.settings.predict_service:
            data = self.load_data()
            prediction_report = self.predict_csv(
                data=data, metadata=metadata, artifact=artifact, input_model=InputModel
            )

        self.logger.info(
            "Prediction completed",
            extra={
                "stage": StagePipeline.INFERENCE,
                "latency_ms": round(time.perf_counter() - start_time, 2),
                "prediction": prediction_report.predictions,
            },
        )
        return prediction_report

    def predict_service(
        self,
        input: str | pd.DataFrame,
        metadata: Metadata,
        artifact: Artifact,
        input_model: Type,
    ) -> PredictionReport:
        if isinstance(input, str):
            aligned_input = self.handle_json_input(
                metadata=metadata, input_model=input_model, inputs=input
            )
        if isinstance(input, pd.DataFrame):
            aligned_input = self.handle_df_input(
                inputs=input, metadata=metadata, input_model=input_model
            )
        self.logger.info(
            "Prediction service started",
            extra={
                "stage": StagePipeline.INFERENCE,
                "n_data": len(aligned_input),
            },
        )
        prediction = self.predict_data(artifact=artifact, data=aligned_input)
        prediction_report = self.create_prediction(
            prediction=prediction, metadata=metadata
        )

        return prediction_report

    def predict_csv(
        self,
        data: pd.DataFrame,
        metadata: Metadata,
        artifact: Artifact,
        input_model: Type,
    ) -> PredictionReport:
        aligned_data = self.handle_data(
            data=data, metadata=metadata, input_model=input_model
        )
        self.logger.info(
            "Data Prediction started",
            extra={
                "stage": StagePipeline.INFERENCE,
                "n_data": len(aligned_data),
            },
        )
        prediction = self.predict_data(data=aligned_data, artifact=artifact)
        prediction_report = self.create_prediction(
            prediction=prediction, metadata=metadata
        )
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
        indexed_row = self._index_row(inputs=normalized_data)
        validated_input = validate_input(
            normalized_data_input=indexed_row,
            input_schemas=input_model,
            allow_missing_features=self.config.inference.allow_missing_features,
        )
        dataframe_data = self._from_model_to_df(validated_input=validated_input)
        aligned_data = self.align_data(data=dataframe_data, metadata=metadata)
        return aligned_data

    def _df_to_dict(self, data: pd.DataFrame) -> list[dict]:
        data = data.astype(object)
        data = data.where(data.notna(), None)
        return data.to_dict(orient="records")

    def handle_df_input(
        self, inputs: pd.DataFrame, metadata: Metadata, input_model: Type
    ) -> pd.DataFrame:
        dict_input_in_list = self._df_to_dict(data=inputs)
        normalized_input = self._normalize_input(
            inputs=dict_input_in_list, metadata=metadata
        )
        indexed_row = self._index_row(inputs=normalized_input)
        validated_input = validate_input(
            normalized_data_input=indexed_row,
            input_schemas=input_model,
            allow_missing_features=self.config.inference.allow_missing_features,
        )
        dataframe_input = self._from_model_to_df(validated_input=validated_input)
        aligned_input = self.align_data(data=dataframe_input, metadata=metadata)
        return aligned_input

    def handle_json_input(
        self, metadata: Metadata, input_model: Type, inputs: str
    ) -> pd.DataFrame:
        dict_input_in_list = self._JSON_to_dict(inputs=inputs)
        normalized_input = self._normalize_input(
            inputs=dict_input_in_list, metadata=metadata
        )
        indexed_row = self._index_row(inputs=normalized_input)
        validated_input = validate_input(
            normalized_data_input=indexed_row,
            input_schemas=input_model,
            allow_missing_features=self.config.inference.allow_missing_features,
        )
        dataframe_input = self._from_model_to_df(validated_input=validated_input)
        aligned_input = self.align_data(data=dataframe_input, metadata=metadata)
        return aligned_input

    def _index_row(self, inputs: list[dict]):
        counter = 1
        for dict in inputs:
            dict["data_id"] = counter
            counter += 1
        return inputs

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

                    self.logger.debug(
                        "Normalized categorial data",
                        extra={
                            "stage": StagePipeline,
                            "features_name": str(feature),
                            "feature_value": str(value),
                        },
                    )

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

        self.logger.debug(
            "Artifact loaded and validated",
            extra={
                "stage": StagePipeline.INFERENCE,
                "artifact_path": metadata.run.artifact_name,
                "model_type": metadata.model.type,
                "model_params": metadata.model.params,
            },
        )
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

    def align_data(
        self,
        data: pd.DataFrame,
        metadata: Metadata,
    ) -> pd.DataFrame:
        data = align_data(
            data=data,
            metadata_columns=metadata.training.features_col,
        )
        return data

    def predict_data(
        self, artifact: Artifact, data: pd.DataFrame
    ) -> list[tuple[int, float, int]]:
        return predict_model(
            artifact=artifact,
            data=data,
            threshold=self.config.inference.threshold,
        )

    def create_prediction(
        self, prediction: list[tuple[int, float, int]], metadata: Metadata
    ) -> PredictionReport:
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
            features_list=metadata.training.features_col,
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
    def from_config(
        cls: Type[T], config: Config, logger: logging.Logger, settings: Settings
    ) -> T:
        return cls(config, logger, settings)
