from ..tools.schemas import Config
from ..services.data_loader import load_data
from ..services.preprocessor import create_preprocessor, split_data
from ..services.models import cross_validation, fit_model
from ..services.IO import create_artifact, create_metadata
from typing import TypeVar, Type
import logging


class TrainingPipeline:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def load_data(self) -> None:
        self.data = load_data(self.config.data.train_path)
        self.total_rows = self.data.shape[0]
        self.total_columns = self.data.shape[1]

    def preprocess(self) -> None:
        self.X, self.y = split_data(
            df=self.data,
            target_col=self.config.train.target_col,
            true_value=self.config.train.true_value,
            drop_features=self.config.train.drop_features,
        )
        self.preprocessor = create_preprocessor(X=self.X)
        self.ratio = f"1:{round(self.y.value_counts()[1]/self.y.value_counts()[0],1)}"

    def evaluate(self) -> None:
        self.report = cross_validation(
            preprocessor=self.preprocessor,
            X=self.X,
            y=self.y,
            n_cv=self.config.train.n_cv,
            random_seed=self.config.train.random_seed,
            selection_metrics=self.config.train.selection_metrics,
            stratify=self.config.train.stratify,
            models=self.config.train.model,
        )
        self.best_model = self._best_model()

    def fit_model(self) -> None:
        self.fitted_model = fit_model(
            preprocessor=self.preprocessor,
            X=self.X,
            y=self.y,
            random_seed=self.config.train.random_seed,
            models=self.config.train.model,
        )

    def save_artifact(self) -> None:
        create_artifact(
            best_model=self.best_model,
            fitted_model=self.fitted_model,
            only_best=self.config.artifact.only_best,
            save_dir=self.config.artifact.save_dir,
        )

    def save_metadata(self) -> None:
        create_metadata(
            best_model=self.best_model,
            save_dir=self.config.artifact.save_dir,
            only_best=self.config.artifact.only_best,
            features_col=self.X.columns,
            report=self.report,
            target_col=self.config.train.target_col,
            train_data=self.config.data.train_path,
            n_samples=self.X.shape[0],
            stratify=self.config.train.stratify,
            random_seed=self.config.train.random_seed,
            models=self.config.train.model,
        )

    def _best_model(self) -> str:
        return max(
            self.report,
            key=lambda x: self.report[x][
                "test_{}".format(self.config.train.selection_metrics)
            ],
        )

    T = TypeVar("T", bound="TrainingPipeline")

    @classmethod
    def from_config(cls: Type[T], config: Config, logger: logging.Logger) -> T:
        return cls(config, logger)
