class ConfigError(Exception):
    pass


class ConfigInvalidError(ConfigError):
    pass


class ConfigNotExistsError(ConfigError):
    pass


class DataError(Exception):
    pass


class DataNotExistsError(DataError):
    pass


class DataExtensionError(DataError):
    pass


class DataEmptyError(DataError):
    pass


class PreprocessError(Exception):
    pass


class TargetColumnError(PreprocessError):
    pass


class PositiveValueError(PreprocessError):
    pass


class DropColumnsError(PreprocessError):
    pass


class TrainingError(Exception):
    pass


class MetricsInvalidError(TrainingError):
    pass


class ParamsInvalidError(TrainingError):
    pass


class InferenceError(Exception):
    pass


class MetadataError(InferenceError):
    pass


class ArtifactError(InferenceError):
    pass


class ColumnsMissingError(InferenceError):
    pass
