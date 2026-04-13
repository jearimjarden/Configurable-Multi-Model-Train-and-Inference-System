import pickle
import os
from pathlib import Path
from datetime import datetime
import json
from pydantic import ValidationError
from ..tools.schemas import (
    Artifact,
    ConfigTrainModel,
    FittedModelPipeline,
    Metadata,
)
from ..tools.exceptions import ArtifactError, MetadataError


def create_artifact(
    save_name: str,
    fitted_pipeline: FittedModelPipeline,
    save_dir: str,
    uuid: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    artifact = {"uuid": uuid, "pipeline": fitted_pipeline["model"]}

    with open(f"{save_dir}/{save_name}.pkl", "wb") as f:
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
            "stratify": stratify,
            "random_seed": random_seed,
        },
        "data": {
            "train_data": train_data,
            "n_samples": n_samples,
        },
        "metrics": evaluation_report,
    }
    with open(f"{save_dir}/{save_name}.json", "w") as f:
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
    prediction: list[tuple[float, int]],
    metadata_name: str,
    allow_missing_features: bool,
    threshold: float,
    save_dir: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    report = {
        "metadata": {
            "uuid": str_uuid,
            "metadata name": metadata_name,
            "allow missing features": allow_missing_features,
            "threshold": threshold,
            "timestamp": datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
        },
        "predictions": [
            {"prediction": pred, "probability": proba} for pred, proba in prediction
        ],
    }

    with open(Path(save_dir) / save_name, "w") as f:
        json.dump(report, f, indent=4)
