import sys
import logging
from data.test.test_data_custom import TEST_CASES
from src.tools.schemas import Config, Settings, StagePipeline
from src.tools.cli import parse_cli
from src.tools.logging import create_bootstrap_logger, setup_logging
from src.tools.exceptions import (
    ConfigurationError,
    LoggedError,
)
from src.tools.loader import load_config, load_settings
from src.pipelines.inference_pipeline import InferencePipeline


def main(logger: logging.Logger, settings: Settings, config: Config):
    try:
        pipeline = InferencePipeline.from_config(
            config=config, logger=logger, settings=settings
        )
        logger.info(
            "Inference pipeline created",
            extra={
                "stage": StagePipeline.INFERENCE,
                "config": {
                    "metadata_path": (
                        config.inference.load_dir + config.inference.metadata_name
                    )
                },
                "threshold": config.inference.threshold,
                "allow_missing_features": config.inference.allow_missing_features,
                "save_result": config.inference.save_result,
            },
        )
        input_data = TEST_CASES["batch_missing"].json()
        pipeline.predict(input=input_data)

        logger.info(
            "Inference pipeline completed", extra={"stage": StagePipeline.INFERENCE}
        )
        sys.exit(0)

    except LoggedError:
        sys.exit(1)

    except Exception as e:
        logger.critical(str(e), exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    try:
        bootstrap_logger = create_bootstrap_logger()
        cli_data = parse_cli()
        settings = load_settings(".env")
        config = load_config("config.yaml")
        setup_logging(level=cli_data.logger, settings=settings)
        logger = logging.getLogger(__name__)
        main(logger=logger, settings=settings, config=config)

    except ConfigurationError as e:
        bootstrap_logger.error(str(e))
        sys.exit(1)

    except Exception as e:
        bootstrap_logger.critical(str(e), exc_info=True)
        sys.exit(2)
