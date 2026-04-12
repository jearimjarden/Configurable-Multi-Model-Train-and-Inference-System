import pickle
import os
from pathlib import Path
from datetime import datetime
import json
from pydantic import ValidationError
from ..tools.schemas import (
    Artifact,
    ConfigTrainModel,
    FittedModel,
    Metadata,
)
from ..tools.exceptions import ArtifactError, MetadataError


def create_artifact(
    save_name: str,
    fitted_model: FittedModel,
    save_dir: str,
    uuid: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    pipeline = fitted_model["model"]
    artifact = {"uuid": uuid, "pipeline": pipeline}

    with open(f"{save_dir}/{save_name}.pkl", "wb") as f:
        pickle.dump(artifact, f)


def create_metadata(
    save_name: str,
    save_dir: str,
    report: dict,
    train_data: str,
    n_samples: int,
    stratify: bool,
    target_col: str,
    features_col: list,
    random_seed: int,
    model_name: str,
    uuid: str,
    models: dict[str, ConfigTrainModel],
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    metadata_report = {
        "run": {
            "uuid": uuid,
            "artifact_name": f"{save_name}.pkl",
            "timestamp": datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
        },
        "model": {
            "type": models[model_name].type.name,
            "params": models[model_name].params,
        },
        "training": {
            "target_col": target_col,
            "features_col": features_col,
            "stratify": stratify,
            "random_seed": random_seed,
        },
        "data": {
            "train_data": train_data,
            "n_samples": n_samples,
        },
        "metrics": report,
    }
    with open(f"{save_dir}/{save_name}.json", "w") as f:
        json.dump(metadata_report, f, indent=4)


def load_metadata(load_dir: str, metadata_name: str) -> Metadata:
    with open(Path(load_dir) / metadata_name, "r") as f:
        data = json.load(f)
        try:
            return Metadata(**data)
        except (ValidationError, KeyError):
            raise MetadataError("Invalid Metadata Structure or Value")


def load_artifact(load_dir: str, artifact_name: str) -> Artifact:
    with open(Path(load_dir) / artifact_name, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            raise ArtifactError("Could not Load Artifact")


def create_report(
    save_name: str,
    uuid: str,
    prediction: list[tuple[float, int]],
    metadata_name: str,
    allow_missing_features: bool,
    threshold: float,
    save_dir: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    report = {
        "metadata": {
            "uuid": uuid,
            "metadata name": metadata_name,
            "allow missing features": allow_missing_features,
            "threshold": threshold,
            "timestamp": datetime.now().strftime("%d/%m%Y, %H:%M:%S"),
        },
        "predictions": [
            {"prediction": pred, "probability": proba} for pred, proba in prediction
        ],
    }

    with open(Path(save_dir) / save_name, "w") as f:
        json.dump(report, f, indent=4)
