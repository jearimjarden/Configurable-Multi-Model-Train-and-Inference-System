import logging
import sys
from ..tools.exceptions import (
    ConfigError,
    DataError,
    PreprocessError,
    TrainingError,
)
from ..tools.cli import parse_cli
from ..tools.logging import setup_logging
from ..tools.loader import load_config
from ..pipelines.training_pipeline import TrainingPipeline


def main(logger: logging.Logger):
    try:
        logger.info("Starting training module")

        config = load_config()
        logger.info("Config loaded")

        pipeline = TrainingPipeline.from_config(config=config, logger=logger)
        logger.info("Training pipeline created")

        logger.info("Starting training pipeline")
        pipeline.run()

        logger.info("Exiting training module")
        sys.exit(0)

    except (
        ConfigError,
        DataError,
        PreprocessError,
        TrainingError,
    ) as e:
        logger.error(str(e))
        sys.exit(1)

    except Exception as e:
        logger.critical(str(e), exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    cli_data = parse_cli()
    setup_logging(cli_data.logger)
    logger = logging.getLogger(__name__)
    main(logger)
