from src.io.artifact_io import load_artifact
from src.services.models import predict_model
from src.io.metadata_io import load_metadata
from src.tools.schemas import Metadata
from src.data.input_validation import validate_input
from src.tools.schemas import create_pydantic_from_metadata
import logging
import pytest
import pandas as pd
from src.tools.exceptions import NoValidDataError

TEST_SINGLE_COMPLETE = [
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
    }
]

TEST_SINGLE_MISSING = [
    {
        "Loan_ID": "LP00105",
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

TEST_SINGLE_FAULTY = [
    {
        "Loan_ID": "LP00105",
        "Gender": "Male",
        "Married": "yes",
        "Dependents": 0,
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": "yes",
        "CoapplicantIncome": 0,
        "LoanAmount": 110,
        "Loan_Amount_Term": "no",
        "Credit_History": 1,
        "Property_Area": "Urban",
    }
]

TEST_BATCH_COMPLETE = [
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
        "Self_Employed": "No",
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

TEST_BATCH_FAULTY = [
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
        "Gender": "Male",
        "Married": "yes",
        "Dependents": 0,
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": "yes",
        "CoapplicantIncome": 0,
        "LoanAmount": 110,
        "Loan_Amount_Term": "no",
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

TEST_BATCH_MISSING = [
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


def test_input_validation():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    metadata = load_metadata(load_dir="tests", metadata_name="test_metadata.json")
    assert isinstance(metadata, Metadata)
    InputConfig = create_pydantic_from_metadata(
        metadata_features_col=metadata.training.features_name_and_type,
        model_name="FeatureColumns",
    )
    counter = 1
    inputs = TEST_SINGLE_COMPLETE
    for dict in inputs:
        dict["data_id"] = counter
        counter += 1
        return inputs
    validate_input(
        normalized_data_input=inputs,
        input_schemas=InputConfig,
        allow_missing_features=True,
        logger=logger,
    )


def test_input_missing():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    metadata = load_metadata(load_dir="tests", metadata_name="test_metadata.json")
    assert isinstance(metadata, Metadata)
    InputConfig = create_pydantic_from_metadata(
        metadata_features_col=metadata.training.features_name_and_type,
        model_name="FeatureColumns",
    )
    counter = 1
    inputs = TEST_SINGLE_MISSING
    for dict in inputs:
        dict["data_id"] = counter
        counter += 1
        return inputs

    with pytest.raises(NoValidDataError):
        validate_input(
            normalized_data_input=inputs,
            input_schemas=InputConfig,
            allow_missing_features=False,
            logger=logger,
        )


def test_input_faulty():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    metadata = load_metadata(load_dir="tests", metadata_name="test_metadata.json")
    assert isinstance(metadata, Metadata)
    InputConfig = create_pydantic_from_metadata(
        metadata_features_col=metadata.training.features_name_and_type,
        model_name="FeatureColumns",
    )
    counter = 1
    inputs = TEST_SINGLE_FAULTY
    for dict in inputs:
        dict["data_id"] = counter
        counter += 1
        return inputs

    with pytest.raises(NoValidDataError):
        validate_input(
            normalized_data_input=inputs,
            input_schemas=InputConfig,
            allow_missing_features=False,
            logger=logger,
        )


@pytest.mark.parametrize(
    "inputs", [TEST_BATCH_COMPLETE, TEST_BATCH_FAULTY, TEST_BATCH_MISSING]
)
def test_prediction(inputs):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    metadata = load_metadata(load_dir="tests", metadata_name="test_metadata.json")
    artifact = load_artifact(load_dir="tests", artifact_name="test_artifact.pkl")
    InputConfig = create_pydantic_from_metadata(
        metadata_features_col=metadata.training.features_name_and_type,
        model_name="FeatureColumns",
    )
    counter = 1
    for dict in inputs:
        dict["data_id"] = counter
        counter += 1
        return inputs
    validated_input = validate_input(
        normalized_data_input=inputs,
        input_schemas=InputConfig,
        allow_missing_features=False,
        logger=logger,
    )
    data_frame_input = pd.DataFrame([item.model_dump() for item in validated_input])
    prediction = predict_model(artifact=artifact, data=data_frame_input, threshold=0.5)
    assert isinstance(prediction, list)
