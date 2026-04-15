import logging
import sys
from ..tools.schemas import Settings, StagePipeline
from ..tools.exceptions import (
    ConfigError,
    DataError,
    PreprocessError,
    TrainingError,
)
from ..tools.cli import parse_cli
from ..tools.logging import create_bootstrap_logger, setup_logging
from ..tools.loader import load_config, load_settings
from ..pipelines.training_pipeline import TrainingPipeline


def main(logger: logging.Logger, settings: Settings):
    try:
        config = load_config()

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
    try:
        bootstrap_logger = create_bootstrap_logger()
        settings = load_settings()
        cli_data = parse_cli()
        setup_logging(cli_data.logger, settings=settings)
        logger = logging.getLogger(__name__)
        main(logger=logger, settings=settings)

    except ConfigError as e:
        bootstrap_logger.error(str(e))
        sys.exit(1)

    except Exception as e:
        bootstrap_logger.critical(f"Unexpected error occured: {str(e)}", exc_info=True)
        sys.exit(2)
