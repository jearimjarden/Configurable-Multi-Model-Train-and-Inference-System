from typing import Type
import logging
from pydantic import ValidationError
from src.tools.schemas import StagePipeline
from src.tools.exceptions import NoValidDataError


def validate_input(
    normalized_data_input: list,
    input_schemas: Type,
    allow_missing_features: bool,
    logger: logging.Logger,
) -> list[Type]:
    validated_input = []

    for input in normalized_data_input:
        missing_columns = set(input_schemas.model_fields.keys() - set(input.keys()))
        extra_columns = set(set(input.keys()) - input_schemas.model_fields.keys())

        logger.debug(
            "Data validated",
            extra={
                "stage": StagePipeline.DATA_ALIGNMENT,
                "missing_columns": list(missing_columns),
                "extra_columns": list(extra_columns),
            },
        )

        if missing_columns:
            if not allow_missing_features:
                logger.warning(
                    "Skipping data with missing features",
                    extra={
                        "stage": StagePipeline.DATA_ALIGNMENT,
                        "data_id": input["data_id"],
                        "missing_features": missing_columns,
                    },
                )
                continue
            logger.warning(
                "Encountered missing columns and will be imputed",
                extra={
                    "stage": StagePipeline.DATA_ALIGNMENT,
                    "data_id": input["data_id"],
                    "missing_columns": missing_columns,
                },
            )

        if extra_columns:
            logger.info(
                "Encountered extra columns and will be dropped",
                extra={
                    "stage": StagePipeline.DATA_ALIGNMENT,
                    "data_id": input["data_id"],
                    "extra_columns": extra_columns,
                },
            )
        try:
            validated_input.append(input_schemas(**input))

        except ValidationError as e:
            error_messages = []

            for err in e.errors():
                fields = ".".join(str(x) for x in err["loc"])
                error_messages.append((fields, str(err["msg"])))

            logger.warning(
                "Skipping data with invalid value",
                extra={
                    "stage": StagePipeline.VALIDATION,
                    "data_id": input["data_id"],
                    "error": [
                        {"feature": feature, "message": message}
                        for feature, message in error_messages
                    ],
                },
            )
            continue

    if not validated_input:
        raise NoValidDataError(
            "No valid data's row to continue", stage=StagePipeline.VALIDATION
        )

    else:
        return validated_input
