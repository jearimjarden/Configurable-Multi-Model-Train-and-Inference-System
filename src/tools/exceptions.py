class LoggedError(Exception):
    pass


class ConfigurationError(Exception):
    def __init__(self, message: str, stage: str):
        super().__init__(message)
        self.stage = stage


class SettingsInvalidError(ConfigurationError):
    def __init__(self, message: str, stage: str):
        super().__init__(message, stage=stage)


class SettingsNotExistsError(ConfigurationError):
    def __init__(self, message: str, stage: str):
        super().__init__(message, stage=stage)


class ConfigInvalidError(ConfigurationError):
    def __init__(self, message: str, stage: str):
        super().__init__(message, stage=stage)


class ConfigNotExistsError(ConfigurationError):
    def __init__(self, message: str, stage: str):
        super().__init__(message, stage=stage)


class DataError(Exception):
    def __init__(self, message: str, stage: str):
        super().__init__(message)
        self.stage = stage


class DataNotExistsError(DataError):
    def __init__(self, message: str, stage: str):
        super().__init__(message, stage=stage)


class DataInvalidError(DataError):
    def __init__(self, message: str, stage: str):
        super().__init__(message, stage=stage)


class TrainingError(Exception):
    def __init__(self, message: str, stage: str):
        self.stage = stage
        super().__init__(message)


class EvaluationError(TrainingError):
    def __init__(self, message: str, stage: str):
        super().__init__(message, stage=stage)


class ModelSelectionError(TrainingError):
    def __init__(self, message: str, stage: str):
        super().__init__(message, stage=stage)


class InferenceError(Exception):
    def __init__(self, message: str, stage: str):
        self.stage = stage
        super().__init__(message)


class MetadataError(InferenceError):
    def __init__(self, message: str, stage: str):
        super().__init__(message, stage=stage)


class ArtifactError(InferenceError):
    def __init__(self, message: str, stage: str):
        super().__init__(message, stage=stage)


class FeatureTypeError(InferenceError):
    def __init__(self, message: str, stage: str):
        super().__init__(message, stage=stage)


class InputJSONError(DataError):
    def __init__(self, message: str, stage: str):
        super().__init__(message, stage=stage)


class NoValidDataError(DataError):
    def __init__(self, message: str, stage: str):
        super().__init__(message, stage=stage)
