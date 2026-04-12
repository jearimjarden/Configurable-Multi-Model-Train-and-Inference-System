class ConfigError(Exception):
    pass


class ConfigInvalid(ConfigError):
    pass


class ConfigNotExists(ConfigError):
    pass


class DataError(Exception):
    pass


class DataNotExists(DataError):
    pass


class DataInvalidExtension(DataError):
    pass


class DataEmpty(DataError):
    pass


class PreprocessError(Exception):
    pass


class MissingColumns(PreprocessError):
    pass


class TargetColumnInvalid(PreprocessError):
    pass


class TrueValueInvalid(PreprocessError):
    pass


class DropColumnsInvalid(PreprocessError):
    pass


class TrainError(Exception):
    pass


class MetricsInvalid(TrainError):
    pass


class ParamInvalid(TrainError):
    pass


class MetadataError(Exception):
    pass


class ArtifactError(Exception):
    pass
