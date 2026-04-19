import pandas as pd


def infer_semantic(X: pd.DataFrame) -> dict:
    X_cat = X.select_dtypes(include=["object", "category"])
    semantics_columns = {}
    for x in X_cat:
        if str(x).lower().endswith("id"):
            semantic = "ID"
        elif X_cat[x].nunique() / len(X_cat[x]) < 0.1:
            semantic = "categorial"
        else:
            semantic = "text"
        semantics_columns[x] = semantic

    X_num = X.select_dtypes(exclude=["object", "category"])
    for x in X_num:
        semantics_columns[x] = "numerical"
    return semantics_columns
