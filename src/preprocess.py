import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def replace_negative_with_nan(X):
    """
    Replace negative values with NaN.

    Negative values in this dataset represent unavailable/sentinel
    measurements rather than valid feature values.
    """
    X = X.copy()

    if isinstance(X, pd.DataFrame):
        X[X < 0] = np.nan
    else:
        X[X < 0] = np.nan

    return X


def create_preprocessor():
    """
    Create the preprocessing pipeline.

    Steps:
    1. Replace negative sentinel values with NaN.
    2. Replace NaN values using the median learned from training data.
    """
    return Pipeline([
        (
            "negative_to_nan",
            FunctionTransformer(
                replace_negative_with_nan,
                feature_names_out="one-to-one"
            )
        ),
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])