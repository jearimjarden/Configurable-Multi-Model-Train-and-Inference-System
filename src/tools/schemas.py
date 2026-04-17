from pydantic import BaseModel, ConfigDict, Field, field_validator, create_model
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from typing import Optional, TypedDict
from pathlib import Path
from typing import Type
from .exceptions import ConfigurationError, SettingsNotExistsError, FeatureTypeError


class Artifact(TypedDict):
    uuid: str
    pipeline: Pipeline


class FittedModelPipeline(TypedDict):
    name: str
    model: Pipeline


class StagePipeline(str, Enum):
    TRAINING = "training"
    DATA_PROCESSING = "data_processing"
    DATA_PREPROCESSING = "data_preprocessing"
    DATA_ALIGNMENT = "data_aligment"
    EVALUATION = "evaluation"
    MODEL_SELECTION = "model_selection"
    MODEL_FITTING = "model_fitting"
    INFERENCE = "inference"
    DATA_LOADING = "data_loading"
    LOADING = "loading"
    CONFIGURATION_LOADING = "configuration_loading"
    VALIDATION = "validation"


class ModelType(Enum):
    LOGISTIC_REGRESSION = LogisticRegression
    DECISION_TREE = DecisionTreeClassifier
    RANDOM_FOREST = RandomForestClassifier


class InferenceStrategy(str, Enum):
    MOST_FREQUENT = "most_frequent"
    CONSTANT = "constant"


ENV_PATH = ".env"


class LOG_LEVEL(str, Enum):
    LOGGER_DEBUG = "debug"
    LOGGER_INFO = "info"
    LOGGER_WARNING = "warning"
    LOGGER_ERROR = "error"
    LOGGER_CRITICAL = "critical"


class Settings(BaseSettings):
    environment: str = Field(...)
    predict_service: bool = Field(...)
    save_log: bool = Field(...)
    save_log_level: LOG_LEVEL = Field(...)
    model_config = SettingsConfigDict(env_file=ENV_PATH, extra="forbid")

    @classmethod
    def load(cls: Type["Settings"]) -> "Settings":
        if not Path(ENV_PATH).exists():
            raise SettingsNotExistsError(
                f"Env file not found at {ENV_PATH}",
                stage=StagePipeline.CONFIGURATION_LOADING,
            )
        return cls()  # type: ignore


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
        except KeyError:
            raise ConfigurationError(
                "Invalid Model Type", stage=StagePipeline.CONFIGURATION_LOADING
            )


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
    save_result: bool = Field(...)
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
    features_name_and_type: dict[str, dict] = Field(...)
    stratify: bool = Field(...)
    random_seed: int = Field(...)
    model_config = ConfigDict(extra="forbid")


class MetadataData(BaseModel):
    train_data: str = Field(...)
    n_samples: int = Field(...)
    class_ratio: str = Field(...)
    model_config = ConfigDict(extra="forbid")


class Metadata(BaseModel):
    run: MetadataRun = Field(...)
    model: MetadataModel = Field(...)
    training: MetadataTraining = Field(...)
    data: MetadataData = Field(...)
    metrics: dict[str, float] = Field(...)
    model_config = ConfigDict(extra="forbid")


class PredictionReportMetadata(BaseModel):
    uuid: str = Field(...)
    metadata_name: str = Field(...)
    allow_missing_features: bool = Field(...)
    threshold: float = Field(...)
    features_list: list = Field(...)
    timestamp: str = Field(...)
    model_config = ConfigDict(extra="forbid")


class PredictionReportPredictions(BaseModel):
    data_id: int = Field(...)
    prediction: int = Field(...)
    probability: float = Field(...)
    model_config = ConfigDict(extra="forbid")


class PredictionReport(BaseModel):
    metadata: PredictionReportMetadata = Field(...)
    predictions: list[PredictionReportPredictions] = Field(...)
    model_config = ConfigDict(extra="forbid")


TYPE_MAP = {"int64": int, "float64": float, "str": str}


def create_pydantic_from_metadata(
    metadata_features_col: dict[str, dict], model_name: str
) -> type:
    fields = {}

    for feature in metadata_features_col:
        base_type = TYPE_MAP.get(metadata_features_col[feature]["type"])

        if base_type is None:
            raise FeatureTypeError(
                f"Unrecognized feature data type: name={feature} type={metadata_features_col[feature]['type']}",
                stage=StagePipeline.VALIDATION,
            )
        fields[feature] = (Optional[base_type], None)

    fields["data_id"] = (int, None)
    return create_model(model_name, **fields, __config__=ConfigDict(extra="ignore"))
