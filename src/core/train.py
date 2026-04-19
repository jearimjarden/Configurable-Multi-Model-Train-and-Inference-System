import logging
import sys
from src.tools.schemas import Settings, StagePipeline, Config
from src.tools.exceptions import (
    ConfigurationError,
    LoggedError,
)
from src.tools.cli import parse_cli
from src.tools.logging import create_bootstrap_logger, setup_logging
from src.tools.loader import load_config, load_settings
from src.pipelines.training_pipeline import TrainingPipeline


def main(logger: logging.Logger, settings: Settings, config: Config):
    try:
        pipeline = TrainingPipeline.from_config(
            config=config, logger=logger, settings=settings
        )

        logger.info(
            "Training pipeline succesfully created",
            extra={
                "stage": StagePipeline.TRAINING,
                "config": {
                    "model": [model for model in config.train.model],
                    "random_seed": config.train.random_seed,
                    "target_col": config.train.target_col,
                    "drop_features": config.train.drop_features,
                    "selection_metrics": config.train.selection_metrics,
                    "missing_strategy": config.train.missing_strategy,
                },
            },
        )

        pipeline.run()

        logger.info(
            "Training pipeline completed", extra={"stage": StagePipeline.TRAINING}
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
        settings = load_settings(".env")
        cli_data = parse_cli()
        config = load_config("config.yaml")
        setup_logging(cli_data.logger, settings=settings)
        logger = logging.getLogger(__name__)
        main(logger=logger, settings=settings, config=config)

    except ConfigurationError as e:
        bootstrap_logger.error(str(e))
        sys.exit(1)

    except Exception as e:
        bootstrap_logger.critical(f"Unexpected error occured: {str(e)}", exc_info=True)
        sys.exit(2)
