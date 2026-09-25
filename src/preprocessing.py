"""Preprocess the queue dataset for a beginner machine learning workflow.

This file focuses only on data preprocessing: loading the data, checking it,
separating features and target, encoding categorical values, and splitting the
set into training/testing data.

It does NOT train a model. That will happen on a later day.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# pandas: used to read CSV files and work with tabular data.
# It helps us inspect rows, columns, and values easily.
# ColumnTransformer: lets us apply different preprocessing steps to
# different columns. For example, numeric columns can stay as-is while
# categorical columns are encoded into numbers.
# OneHotEncoder: converts text categories like "Hospital" and "Banking"
# into numeric columns so a machine learning model can understand them.
# train_test_split: splits our dataset into training and testing sets.

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "queue_data.csv"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

TARGET_COLUMN = "waiting_time"
FEATURE_COLUMNS = [
    "people_waiting",
    "available_counters",
    "average_service_time",
    "queue_type",
    "hour",
    "day_of_week",
]
NUMERIC_COLUMNS = [
    "people_waiting",
    "available_counters",
    "average_service_time",
    "hour",
]
CATEGORICAL_COLUMNS = ["queue_type", "day_of_week"]


def load_dataset(file_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset from CSV and print a quick verification summary."""
    df = pd.read_csv(file_path)

    print("Dataset loaded successfully.")
    print("\nFirst 5 rows:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print("\nColumns:")
    print(list(df.columns))

    return df


def split_features_and_target(df: pd.DataFrame):
    """Split data into X (features) and y (target).

    X = information used to make the prediction.
    y = value that the model must learn to predict.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    print("\nX contains the input features: ")
    print(X.head())
    print("\nY contains the target value to predict: ")
    print(y.head())

    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Create a preprocessing pipeline for numeric and categorical columns."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_COLUMNS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def save_processed_data(
    X_train_processed,
    X_test_processed,
    y_train,
    y_test,
    preprocessor: ColumnTransformer,
):
    """Save processed train/test data for later use.

    This helps keep the processed dataset separate from the raw CSV file.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    feature_names = preprocessor.get_feature_names_out()

    pd.DataFrame(X_train_processed, columns=feature_names).to_csv(
        PROCESSED_DIR / "X_train_processed.csv",
        index=False,
    )
    pd.DataFrame(X_test_processed, columns=feature_names).to_csv(
        PROCESSED_DIR / "X_test_processed.csv",
        index=False,
    )
    pd.DataFrame({"waiting_time": y_train}).to_csv(
        PROCESSED_DIR / "y_train.csv",
        index=False,
    )
    pd.DataFrame({"waiting_time": y_test}).to_csv(
        PROCESSED_DIR / "y_test.csv",
        index=False,
    )

    print("\nProcessed files saved in data/processed/")


def main():
    """Run the full preprocessing workflow."""
    df = load_dataset()
    X, y = split_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("\nTraining and testing split complete.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("\nEncoded train data shape:")
    print(X_train_processed.shape)
    print("\nEncoded test data shape:")
    print(X_test_processed.shape)

    print("\nExample of the encoded training matrix:")
    print(X_train_processed[:2])

    print("\nUnderstanding the split:")
    print("- X_train: input features used for training")
    print("- X_test: input features used for testing")
    print("- y_train: target values used for training")
    print("- y_test: target values used for testing")

    save_processed_data(X_train_processed, X_test_processed, y_train, y_test, preprocessor)


if __name__ == "__main__":
    main()
