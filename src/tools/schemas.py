from pydantic import BaseModel, ConfigDict, Field, field_validator
from enum import Enum
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class ModelType(Enum):
    LOGISTIC_REGRESSION = LogisticRegression
    DECISION_TREE = DecisionTreeClassifier
    RANDOM_FOREST = RandomForestClassifier


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
            raise e


class ConfigTrain(BaseModel):
    model: dict[str, ConfigTrainModel] = Field(...)
    stratify: bool = Field(...)
    n_cv: int = Field(5, ge=5)
    random_seed: int = Field(...)
    target_col: str = Field(...)
    true_value: str = Field(...)
    drop_features: list = Field(...)
    selection_metrics: str = Field(...)
    model_config = ConfigDict(extra="forbid")


class ConfigInference(BaseModel):
    metadata_path: str = Field(...)
    allow_missing_features: bool = Field(...)
    inference_report_path: str = Field(...)
    model_config = ConfigDict(extra="forbid")


class ConfigArtifact(BaseModel):
    save_dir: str = Field(...)
    metadata_name: str = Field(...)
    only_best: bool = Field(...)
    model_config = ConfigDict(extra="forbid")


class Config(BaseModel):
    data: ConfigData = Field(...)
    train: ConfigTrain = Field(...)
    inference: ConfigInference = Field(...)
    artifact: ConfigArtifact = Field(...)
    model_config = ConfigDict(extra="forbid")
