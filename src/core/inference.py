import sys
from ..tools.cli import cli_parser
from ..tools.logging import logging_setup
import logging
from ..tools.exceptions import (
    ConfigError,
    InferenceError,
    DataError,
)
from ..tools.loader import config_load
from ..pipeline.inferencing import InferencePipeline


def main(logger: logging.Logger):
    try:
        logger.info("Starting inference module")

        config = config_load()
        logger.info("Config loaded")

        pipeline = InferencePipeline.from_config(config=config, logger=logger)
        logger.info("Inference pipeline created")

        pipeline.run()

        logger.info("Exiting inference module")
        sys.exit(0)

    except (
        ConfigError,
        InferenceError,
        DataError,
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
    main(logger=logger)
