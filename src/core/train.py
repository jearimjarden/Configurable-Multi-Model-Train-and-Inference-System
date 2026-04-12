from ..tools.exceptions import (
    ConfigError,
    DataError,
    PreprocessError,
    TrainError,
)
from ..tools.cli import cli_parser
from ..tools.logging import logging_setup
from ..tools.loader import config_load
from ..pipeline.training import TrainingPipeline
import logging
import sys


def main(logger: logging.Logger):
    try:
        logger.info("Starting training module")

        config = config_load()
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
        TrainError,
    ) as e:
        logger.error(str(e))
        sys.exit(1)

    except Exception as e:
        logger.critical(str(e), exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    cli_data = cli_parser()
    logging_setup(cli_data.logger)
    logger = logging.getLogger(__name__)
    main(logger)
