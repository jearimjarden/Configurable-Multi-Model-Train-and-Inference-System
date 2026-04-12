from pydantic import BaseModel, ConfigDict, Field, field_validator
from enum import Enum
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from typing import TypedDict


class Artifact(TypedDict):
    uuid: str
    pipeline: Pipeline


class FittedModel(TypedDict):
    name: str
    model: Pipeline


class ModelType(Enum):
    LOGISTIC_REGRESSION = LogisticRegression
    DECISION_TREE = DecisionTreeClassifier
    RANDOM_FOREST = RandomForestClassifier


class InferenceStrategy(str, Enum):
    MOST_FREQUENT = "most_frequent"
    CONSTANT = "constant"


class ConfigData(BaseModel):
    train_path: str = Field(...)
    inference_path: str = Field(...)
    model_config = ConfigDict(extra="forbid")


class ConfigTrainModel(BaseModel):
    type: ModelType = Field(...)
    params: dict = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")

    @field_validator("type", mode="before")
    def parse_model_type(cls, value):
        try:
            return ModelType[value.upper()]
        except KeyError as e:
            raise e  # why i need to add e


class ConfigTrain(BaseModel):
    model: dict[str, ConfigTrainModel] = Field(...)
    stratify: bool = Field(...)
    n_cv: int = Field(5, ge=5)
    random_seed: int = Field(...)
    target_col: str = Field(...)
    true_value: str = Field(...)
    drop_features: list = Field(...)
    selection_metrics: str = Field(...)
    missing_strategy: InferenceStrategy = Field(...)
    model_config = ConfigDict(extra="forbid")


class ConfigInference(BaseModel):
    load_dir: str = Field(...)
    metadata_name: str = Field(...)
    allow_missing_features: bool = Field(...)
    inference_report_path: str = Field(...)
    threshold: float = Field(0.5, ge=0, le=1)
    model_config = ConfigDict(extra="forbid")


class ConfigArtifact(BaseModel):
    save_dir: str = Field(...)
    only_best: bool = Field(...)
    model_config = ConfigDict(extra="forbid")


class Config(BaseModel):
    data: ConfigData = Field(...)
    train: ConfigTrain = Field(...)
    inference: ConfigInference = Field(...)
    artifact: ConfigArtifact = Field(...)
    model_config = ConfigDict(extra="forbid")


class MetadataRun(BaseModel):
    uuid: str = Field(...)
    artifact_name: str = Field(...)
    timestamp: str = Field(...)
    model_config = ConfigDict(extra="forbid")


class MetadataModel(BaseModel):
    type: str = Field(...)
    params: dict[str, int] = Field(...)
    model_config = ConfigDict(extra="forbid")


class MetadataTraining(BaseModel):
    target_col: str = Field(...)
    features_col: list = Field(...)
    stratify: bool = Field(...)
    random_seed: int = Field(...)
    model_config = ConfigDict(extra="forbid")


class MetadataData(BaseModel):
    train_data: str = Field(...)
    n_samples: int = Field(...)
    model_config = ConfigDict(extra="forbid")


class Metadata(BaseModel):
    run: MetadataRun = Field(...)
    model: MetadataModel = Field(...)
    training: MetadataTraining = Field(...)
    data: MetadataData = Field(...)
    metrics: dict[str, float] = Field(...)
    model_config = ConfigDict(extra="forbid")
