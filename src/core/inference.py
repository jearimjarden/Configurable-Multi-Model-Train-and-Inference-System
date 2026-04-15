import sys
import logging
import json
import pandas as pd
from ..tools.schemas import Settings, StagePipeline
from ..tools.cli import parse_cli
from ..tools.logging import create_bootstrap_logger, setup_logging
from ..tools.exceptions import (
    ConfigError,
    InferenceError,
    DataError,
)
from ..tools.loader import load_config, load_settings
from ..pipelines.inference_pipeline import InferencePipeline

TEST_SINGLE = [
    {
        "Loan_ID1": "LP00105",
        "Loan_ID": "LP00105",
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": 0,
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5720,
        "CoapplicantIncome": 0,
        "LoanAmount": 110,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Urban",
    }
]

TEST_MISSING = [
    {
        "Loan_ID": "LP00105",
        "Gender": "Female",
        "Married": "Yes",
        "Dependents": 0,
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5720,
        "CoapplicantIncome": 0,
        "LoanAmount": 110,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Urban",
    }
]
TEST_BATCH = [
    {
        "Loan_ID": "LP00105",
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": 0,
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5720,
        "CoapplicantIncome": 0,
        "LoanAmount": 110,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Urban",
    },
    {
        "Loan_ID": "LP00105",
        "Gender": "Female",
        "Married": "Yes",
        "Dependents": 0,
        "Education": "Graduate",
        "ApplicantIncome": 5720,
        "CoapplicantIncome": 0,
        "LoanAmount": 110,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Urban",
    },
    {
        "Loan_ID": "LP00106",
        "Gender": "Female",
        "Married": "No",
        "Dependents": 2,
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 0,
        "CoapplicantIncome": 0,
        "LoanAmount": 11000,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Urban",
    },
]

TEST_SINGLE_DF = pd.DataFrame(TEST_SINGLE)


def main(logger: logging.Logger, settings: Settings):
    try:
        # input_data = json.dumps(TEST_BATCH)
        input_data = TEST_SINGLE_DF

        config = load_config()

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

        pipeline.run(input=input_data)

        logger.info(
            "Inference pipeline completed", extra={"stage": StagePipeline.INFERENCE}
        )
        sys.exit(0)

    except (
        ConfigError,
        InferenceError,
        DataError,
    ) as e:
        logger.error(str(e), extra={"error_type": type(e).__name__})
        sys.exit(1)

    except Exception as e:
        logger.critical(str(e), exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    try:
        bootstrap_logger = create_bootstrap_logger()
        cli_data = parse_cli()
        settings = load_settings()
        setup_logging(level=cli_data.logger, settings=settings)
        logger = logging.getLogger(__name__)
        main(logger=logger, settings=settings)

    except ConfigError as e:
        bootstrap_logger.error(str(e))
        sys.exit(1)

    except Exception as e:
        bootstrap_logger.critical(str(e), exc_info=True)
        sys.exit(2)
