from datetime import datetime
import json
import os
from pathlib import Path
from pydantic import ValidationError
from src.tools.schemas import ConfigTrainModel, Metadata, StagePipeline
from src.tools.exceptions import MetadataError


def create_metadata(
    save_metadata_name: str,
    save_artifact_name: str,
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
    class_ratio: str,
    str_uuid: str,
    models: dict[str, ConfigTrainModel],
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    metadata_report = {
        "run": {
            "uuid": str_uuid,
            "artifact_name": save_artifact_name,
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
            "class_ratio": class_ratio,
        },
        "metrics": evaluation_report,
    }
    with open(Path(save_dir) / save_metadata_name, "w") as f:
        json.dump(metadata_report, f, indent=4)


def load_metadata(load_dir: str, metadata_name: str) -> Metadata:
    metadata_path = Path(load_dir) / metadata_name
    if not metadata_path.exists():
        raise MetadataError(
            f"Cannot find metadata at '{metadata_path.resolve()}'",
            stage=StagePipeline.LOADING,
        )
    with open(metadata_path, "r") as f:
        data = json.load(f)
        try:
            return Metadata(**data)

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
                        f"Invalid parameter's value type for '{field}': {err['msg']}"
                    )

            raise MetadataError(
                " | ".join(message for message in messages), stage=StagePipeline.LOADING
            ) from e
