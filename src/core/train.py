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

        pipeline.load_data()
        logger.info(
            f"Training data loaded from {config.data.train_path} with {pipeline.total_rows} rows and {pipeline.total_columns} columns"
        )

        pipeline.preprocess()
        logger.info(f"Data preprocessed with label ratio of  1: {pipeline.ratio}")

        pipeline.evaluate()
        logger.info(
            f"Evaluated with n_fold: {config.train.n_cv}, selection_metrics: {config.train.selection_metrics}, stratify: {config.train.stratify}, random_seed: {config.train.random_seed}"
        )

        pipeline.fit_model()
        logger.info(f"Successfully fit model: {[x for x in pipeline.fitted_model]}")

        pipeline.save_artifact()
        pipeline.save_metadata()
        logger.info(
            f"Artifact and metadata saved in '{config.artifact.save_dir}', saved best only: {config.artifact.only_best}"
        )

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
    logger = logging.getLogger("train.py")
    main(logger)
