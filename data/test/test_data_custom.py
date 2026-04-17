import json
import pandas as pd

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


class TestCase:
    def __init__(self, data: list[dict]):
        self._data = data

    def raw(self) -> list[dict]:
        return self._data

    def json(self) -> str:
        return json.dumps(self._data)

    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self._data)


TEST_CASES = {
    "single_complete": TestCase(TEST_SINGLE_COMPLETE),
    "single_missing": TestCase(TEST_SINGLE_MISSING),
    "single_faulty": TestCase(TEST_SINGLE_FAULTY),
    "batch_complete": TestCase(TEST_BATCH_COMPLETE),
    "batch_missing": TestCase(TEST_BATCH_MISSING),
    "batch_faulty": TestCase(TEST_BATCH_FAULTY),
}
