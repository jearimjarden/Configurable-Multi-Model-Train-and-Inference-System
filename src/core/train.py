from ..services.IO import create_artifact, create_metadata

from ..services.preprocessor import create_preprocessor, split_data
from ..tools.exceptions import (
    ConfigInvalid,
    ConfigNotExists,
    DataEmpty,
    DataInvalidExtension,
    DataNotExists,
)
from ..tools.cli import cli_parser
from ..tools.logging import logging_setup
from ..tools.loader import config_load
from ..services.data_loader import load_data
from ..services.models import cross_validation, fit_model
import logging
import sys


def main(logger: logging.Logger):
    try:
        logger.info("Starting training module")
        config = config_load()
        data = load_data(config.data.train_path)
        X, y = split_data(
            df=data,
            target_col=config.train.target_col,
            true_value=config.train.true_value,
            drop_features=config.train.drop_features,
        )
        preprocessor = create_preprocessor(X)
        report = cross_validation(
            preprocessor=preprocessor,
            X=X,
            y=y,
            n_cv=config.train.n_cv,
            random_seed=config.train.random_seed,
            selection_metrics=config.train.selection_metrics,
            stratify=config.train.stratify,
            models=config.train.model,
        )
        fitted_model = fit_model(
            preprocessor=preprocessor,
            X=X,
            y=y,
            random_seed=config.train.random_seed,
            models=config.train.model,
        )
        create_artifact(
            fitted_model=fitted_model,
            only_best=config.artifact.only_best,
            report=report,
            selection_metrics=config.train.selection_metrics,
            save_dir=config.artifact.save_dir,
        )
        create_metadata(
            save_dir=config.artifact.save_dir,
            only_best=config.artifact.only_best,
            features_col=X.columns,
            report=report,
            target_col=config.train.target_col,
            train_data=config.data.train_path,
            n_samples=X.shape[0],
            stratify=config.train.stratify,
            random_seed=config.train.random_seed,
            models=config.train.model,
            selection_metrics=config.train.selection_metrics,
        )
    except (
        ConfigInvalid,
        ConfigNotExists,
        DataNotExists,
        DataNotExists,
        DataInvalidExtension,
        DataEmpty,
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
