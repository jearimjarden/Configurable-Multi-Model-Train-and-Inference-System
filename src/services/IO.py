import pickle
import os
from pathlib import Path
from datetime import datetime
import json
from typing import Type
from pydantic import ValidationError
import logging
import pandas as pd
from ..tools.schemas import (
    Artifact,
    ConfigTrainModel,
    FittedModelPipeline,
    Metadata,
    PredictionReport,
    StagePipeline,
)
from ..tools.exceptions import (
    ArtifactError,
    ColumnsMissingError,
    FeatureTypeError,
    InputJSONError,
    MetadataError,
)

logger = logging.getLogger(__name__)


def infer_semantic(X: pd.DataFrame) -> dict:
    X_cat = X.select_dtypes(include=["object", "category"])
    semantics_columns = {}
    for x in X_cat:
        if str(x).lower().endswith("id"):
            semantic = "ID"
        elif X_cat[x].nunique() / len(X_cat[x]) < 0.1:
            semantic = "categorial"
        else:
            semantic = "text"
        semantics_columns[x] = semantic

    X_num = X.select_dtypes(exclude=["object", "category"])
    for x in X_num:
        semantics_columns[x] = "numerical"
    return semantics_columns


def create_artifact(
    save_name: str,
    fitted_pipeline: FittedModelPipeline,
    save_dir: str,
    uuid: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    artifact = {"uuid": uuid, "pipeline": fitted_pipeline["model"]}

    with open(Path(save_dir) / save_name, "wb") as f:
        pickle.dump(artifact, f)


def create_metadata(
    save_name: str,
    save_dir: str,
    evaluation_report: dict,
    train_data: str,
    n_samples: int,
    stratify: bool,
    target_columns: str,
    features_col: list,
    features_name_and_type: dict,
    random_seed: int,
    model_name: str,
    str_uuid: str,
    models: dict[str, ConfigTrainModel],
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    metadata_report = {
        "run": {
            "uuid": str_uuid,
            "artifact_name": f"{save_name}.pkl",
            "timestamp": datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
        },
        "model": {
            "type": models[model_name].type.name,
            "params": models[model_name].params,
        },
        "training": {
            "target_col": target_columns,
            "features_col": features_col,
            "features_name_and_type": features_name_and_type,
            "stratify": stratify,
            "random_seed": random_seed,
        },
        "data": {
            "train_data": train_data,
            "n_samples": n_samples,
        },
        "metrics": evaluation_report,
    }
    with open(Path(save_dir) / save_name, "w") as f:
        json.dump(metadata_report, f, indent=4)


def load_metadata(load_dir: str, metadata_name: str) -> Metadata:
    metadata_path = Path(load_dir) / metadata_name
    with open(metadata_path, "r") as f:
        data = json.load(f)
        try:
            return Metadata(**data)

        except FileNotFoundError:
            raise MetadataError(f"Cannot find metadata at '{metadata_path.resolve()}'")

        except ValidationError as e:
            messages = []
            for err in e.errors():
                field = ".".join(str(x) for x in err["loc"])
                if err["type"] == "missing":
                    messages.append(f"Missing '{field}' in metadata")
                elif err["type"] == "extra_forbidden":
                    messages.append(f"Extra params are not allowed: '{field}'")
                else:
                    messages.append(
                        f"Invalid parameter's value type for '{field}': {err["msg"]}"
                    )

            raise MetadataError(" | ".join(message for message in messages)) from e


def load_artifact(load_dir: str, artifact_name: str) -> Artifact:
    artifact_path = Path(load_dir) / artifact_name

    try:
        with open(artifact_path, "rb") as f:
            return pickle.load(f)

    except FileNotFoundError:
        raise ArtifactError(f"Cannot find artifact at {artifact_path.resolve()}")

    except Exception as e:
        raise ArtifactError(
            f"Unexpected error occured while loading artifact at {artifact_path.resolve()}"
        ) from e


def create_prediction_report(
    save_name: str,
    str_uuid: str,
    prediction: list[tuple[int, float, int]],
    features_list: list,
    metadata_name: str,
    allow_missing_features: bool,
    threshold: float,
    save_dir: str,
    save_result: bool,
) -> PredictionReport:
    os.makedirs(save_dir, exist_ok=True)

    report = {
        "metadata": {
            "uuid": str_uuid,
            "metadata_name": metadata_name,
            "allow_missing_features": allow_missing_features,
            "threshold": threshold,
            "features_list": features_list,
            "timestamp": datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
        },
        "predictions": [
            {"data_id": id, "prediction": pred, "probability": round(proba, 3)}
            for id, proba, pred in prediction
        ],
    }

    if save_result:
        with open(Path(save_dir) / save_name, "w") as f:
            json.dump(report, f, indent=4)
    return PredictionReport(**report)


def validate_input(
    normalized_data_input: list, input_schemas: Type, allow_missing_features: bool
) -> list[Type]:
    validated_input = []
    try:
        for input in normalized_data_input:
            missing_columns = set(input_schemas.model_fields.keys() - set(input.keys()))
            extra_columns = set(set(input.keys()) - input_schemas.model_fields.keys())

            logger.debug(
                "Data validated",
                extra={
                    "stage": StagePipeline.DATA_ALIGNMENT,
                    "missing_columns": list(missing_columns),
                    "extra_columns": list(extra_columns),
                },
            )

            if missing_columns or extra_columns:
                logger.info(
                    "Encountered missing or extra columns",
                    extra={
                        "stage": StagePipeline.DATA_ALIGNMENT,
                        "data_id": input["data_id"],
                        "dropped_columns": extra_columns,
                        "imputed_columns": missing_columns,
                    },
                )
                if missing_columns and not allow_missing_features:
                    logger.info(
                        "Skipping data with missing features",
                        extra={
                            "stage": StagePipeline.DATA_ALIGNMENT,
                            "data_id": input["data_id"],
                            "missing_features": missing_columns,
                        },
                    )
                    continue

            validated_input.append(input_schemas(**input))

        if not validated_input:
            raise ColumnsMissingError("No valid row to predict")

        else:
            return validated_input

    except json.JSONDecodeError as e:
        raise InputJSONError(f"Invalid JSON input: {e}") from e

    except ValidationError as e:
        messages = []
        for err in e.errors():
            fields = ".".join(str(x) for x in err["loc"])

            if err["type"] == "missing":
                messages.append(
                    FeatureTypeError(f"Missing features in data input: '{fields}'")
                )
            else:
                messages.append(
                    FeatureTypeError(f"Invalid value for '{fields}': {err['msg']}")
                )

        raise FeatureTypeError(" | ".join(str(message) for message in messages)) from e
