import os
from pathlib import Path
import pickle
from src.tools.schemas import FittedModelPipeline, StagePipeline, Artifact
from src.tools.exceptions import ArtifactError


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


def load_artifact(load_dir: str, artifact_name: str) -> Artifact:
    artifact_path = Path(load_dir) / artifact_name

    try:
        with open(artifact_path, "rb") as f:
            return pickle.load(f)

    except FileNotFoundError:
        raise ArtifactError(
            f"Cannot find artifact at {artifact_path.resolve()}",
            stage=StagePipeline.LOADING,
        )

    except Exception as e:
        raise ArtifactError(
            f"Unexpected error occured while loading artifact at {artifact_path.resolve()}",
            stage=StagePipeline.LOADING,
        ) from e
