import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from src.preprocess import create_preprocessor


DATA_PATH = "data/consolidated_traffic_data.csv"
MODEL_DIR = "models"

RANDOM_STATE = 42
TEST_SIZE = 0.2

MLFLOW_EXPERIMENT = "VPN Traffic Classification"
MLFLOW_TRACKING_URI = "sqlite:////app/mlflow.db"


def load_data():
    """Load the dataset and separate features from target."""

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns="traffic_type")
    y = df["traffic_type"]

    return X, y


def create_groups(X):
    """Create a group ID for each identical feature combination."""

    return X.astype(str).agg("|".join, axis=1)


def split_data(X, y, groups):
    """Create a group-aware train/test split."""

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    return X_train, X_test, y_train, y_test


def create_model_pipeline():
    """
    Create a single pipeline containing:

    1. Negative sentinel values -> NaN
    2. Median imputation
    3. Random Forest classifier
    """

    preprocessor = create_preprocessor()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        (
            "preprocessing",
            preprocessor,
        ),
        (
            "classifier",
            model,
        ),
    ])

    return pipeline


def train_model(X_train, y_train):
    """Train the complete preprocessing + model pipeline."""

    pipeline = create_model_pipeline()

    pipeline.fit(
        X_train,
        y_train,
    )

    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    """Evaluate the complete pipeline."""

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
    )

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
        )
    )

    return accuracy, macro_f1, weighted_f1


def save_model(pipeline):
    """Save the complete preprocessing + model pipeline."""

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    model_path = os.path.join(
        MODEL_DIR,
        "traffic_classifier.joblib",
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    print(
        f"\nPipeline saved to: {model_path}"
    )


def main():
    print("Loading dataset...")

    X, y = load_data()

    print(
        f"Dataset shape: {X.shape}"
    )

    groups = create_groups(X)

    print(
        f"Unique feature combinations: "
        f"{groups.nunique()}"
    )

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        groups,
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Test samples: {len(X_test)}"
    )

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    with mlflow.start_run():

        print(
            "\nCreating model pipeline..."
        )

        pipeline = create_model_pipeline()

        print(
            "\nTraining model..."
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        print(
            "\nEvaluating model..."
        )

        accuracy, macro_f1, weighted_f1 = evaluate_model(
            pipeline,
            X_test,
            y_test,
        )

        mlflow.log_params({
            "model": "RandomForestClassifier",
            "n_estimators": 100,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "preprocessing": (
                "negative_to_nan + "
                "median_imputation"
            ),
        })

        mlflow.log_metrics({
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        })

        mlflow.sklearn.log_model(
            pipeline,
            name="traffic_classifier",
            skops_trusted_types=[
                "numpy.dtype",
                "src.preprocess.replace_negative_with_nan",
            ],
        )

        save_model(
            pipeline
        )

        print(
            "\nMLflow run completed."
        )


if __name__ == "__main__":
    main()
