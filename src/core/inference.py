import sys
import logging
from ..tools.cli import parse_cli
from ..tools.logging import setup_logging
from ..tools.exceptions import (
    ConfigError,
    InferenceError,
    DataError,
)
from ..tools.loader import load_config
from ..pipelines.inference_pipeline import InferencePipeline
import json

TEST_SINGLE = [
    {
        "Loan_ID": "LP00105",
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": 0,
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicationIncome": 5720,
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
        "Gender1": "Male",
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


def main(logger: logging.Logger):
    try:
        input_data = json.dumps(TEST_SINGLE)
        logger.info("Starting inference module")

        config = load_config()
        logger.info("Config loaded")

        pipeline = InferencePipeline.from_config(config=config, logger=logger)
        logger.info("Inference pipeline created")

        pipeline.run(input=input_data)

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
    cli_data = parse_cli()
    setup_logging(cli_data.logger)
    logger = logging.getLogger(__name__)
    main(logger=logger)
