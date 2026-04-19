from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.services.models import fit_model
from src.services.preprocessor import create_preprocessor, split_data
import pandas as pd
import pytest
from src.tools.schemas import ConfigTrainModel

df = pd.read_csv("tests/train_data_for_pytest.csv")


def test_split_data():
    X, y = split_data(
        df=df, target_col="Loan_Status", positif_value="Y", drop_features=[]
    )
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


@pytest.mark.parametrize(
    "missing_strategy",
    ["most_frequent", "constant"],
)
def test_create_preprocessor(missing_strategy):
    X, _ = split_data(
        df=df, target_col="Loan_Status", positif_value="Y", drop_features=[]
    )

    preprocessor = create_preprocessor(X=X, missing_strategy=missing_strategy)

    assert isinstance(preprocessor, ColumnTransformer)


def test_fit_model():
    X, y = split_data(
        df=df, target_col="Loan_Status", positif_value="Y", drop_features=[]
    )
    preprocessor = create_preprocessor(X=X, missing_strategy="constant")
    model_dict = ConfigTrainModel(type="LOGISTIC_REGRESSION", params={"max_iter": 1000})
    fitted_model = fit_model(
        preprocessor, X, y, 42, models={"logistic_regression": model_dict}
    )

    assert isinstance(fitted_model, list)
    assert isinstance(fitted_model[0]["name"], str)
    assert isinstance(fitted_model[0]["model"], Pipeline)
