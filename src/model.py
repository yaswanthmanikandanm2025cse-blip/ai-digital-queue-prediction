"""Train and compare beginner-friendly regression models for queue waiting time."""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "queue_data.csv"
MODELS_DIR = ROOT_DIR / "models"
SCREENSHOT_DIR = ROOT_DIR / "screenshots"
MODELS_DIR.mkdir(exist_ok=True)
SCREENSHOT_DIR.mkdir(exist_ok=True)

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


def print_beginner_concepts() -> None:
    """Explain the basic machine learning ideas in simple language."""
    concepts = {
        "Artificial Intelligence": "AI is the idea of making computers perform tasks that normally need human intelligence, such as understanding data, recognizing patterns, and making decisions.",
        "Machine Learning": "Machine learning is a part of AI where the computer learns patterns from data instead of being told every rule explicitly.",
        "Supervised Learning": "In supervised learning, the model learns from examples that already include the correct answer. Here, we show the model many queue records with their waiting_time values.",
        "Regression": "Regression is a type of supervised learning used when the answer is a number, such as minutes of waiting time.",
        "Feature": "A feature is an input variable used to make a prediction. In this project, features include people_waiting, available_counters, and queue_type.",
        "Target": "The target is the value we want to predict. Here, the target is waiting_time.",
        "Training data": "Training data is the portion of the dataset used to teach the model patterns.",
        "Testing data": "Testing data is a separate portion used to check how well the model performs on new, unseen examples.",
        "Model": "A model is the learned pattern or formula created after training. It takes input features and produces a prediction.",
    }

    print("\n=== BEGINNER ML CONCEPTS ===")
    for name, explanation in concepts.items():
        print(f"- {name}: {explanation}")


def load_data(file_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the queue dataset from CSV."""
    df = pd.read_csv(file_path)
    print("\nDataset loaded successfully.")
    print(f"Rows: {len(df)}, Columns: {list(df.columns)}")
    return df


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing for numeric and categorical columns."""
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_COLUMNS),
            ("cat", encoder, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )
    return preprocessor


def build_pipelines() -> dict:
    """Return training pipelines for the models that will be compared."""
    preprocessor = build_preprocessor()
    linear_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    forest_preprocessor = build_preprocessor()
    forest_model = Pipeline(
        steps=[
            ("preprocessor", forest_preprocessor),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )

    return {
        "Linear Regression": linear_model,
        "Random Forest Regressor": forest_model,
    }


def evaluate_model(model_name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Generate predictions and calculate beginner-friendly regression metrics."""
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    metrics = {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Predictions": predictions,
    }

    print(f"\n=== {model_name} ===")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    return metrics


def print_metric_explanations() -> None:
    """Explain each regression metric in simple beginner terms."""
    print("\n=== REGRESSION METRICS ===")
    print("MAE: Average absolute difference between actual and predicted waiting time.")
    print("MSE: Average of the squared prediction errors.")
    print("RMSE: Square root of MSE. It is in the same unit as waiting time, so it is easy to understand.")
    print("R²: Shows how much of the variation in waiting time is explained by the model. A higher value means more explained variation, but it is not automatically perfect.")


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Create a simple comparison table for the two models."""
    comparison = pd.DataFrame(results)
    comparison = comparison[["Model", "MAE", "MSE", "RMSE", "R2"]]
    print("\n=== MODEL COMPARISON ===")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return comparison


def select_model(results: list[dict]) -> tuple[str, dict]:
    """Select the model using the measured test performance. Lower MAE is preferred."""
    selected_name, selected_result = min(
        ((result["Model"], result) for result in results),
        key=lambda item: (item[1]["MAE"], -item[1]["R2"]),
    )
    print(f"\nSelected model: {selected_name}")
    print(
        "Reason: On this test split, it produced a lower MAE than the other model, "
        "which means the average prediction error was smaller on the measured data."
    )
    return selected_name, selected_result


def save_model(model: Pipeline, model_name: str) -> None:
    """Save the trained model and preprocessor to disk for later use."""
    MODELS_DIR.mkdir(exist_ok=True)

    queue_model_path = MODELS_DIR / "queue_waiting_model.pkl"
    joblib.dump(model, queue_model_path)
    print(f"\nSaved selected model to: {queue_model_path}")

    alias_path = MODELS_DIR / f"{model_name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, alias_path)
    print(f"Saved model alias to: {alias_path}")

    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    joblib.dump(model.named_steps["preprocessor"], preprocessor_path)
    print(f"Saved preprocessor to: {preprocessor_path}")


def plot_actual_vs_predicted(actual: pd.Series, predicted: np.ndarray) -> None:
    """Create and save a scatter plot of actual vs predicted waiting times."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(actual, predicted, alpha=0.7)
    min_value = min(actual.min(), predicted.min())
    max_value = max(actual.max(), predicted.max())
    ax.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--", linewidth=1)
    ax.set_title("Actual vs Predicted Waiting Time")
    ax.set_xlabel("Actual Waiting Time")
    ax.set_ylabel("Predicted Waiting Time")
    fig.tight_layout()
    file_path = SCREENSHOT_DIR / "actual_vs_predicted.png"
    fig.savefig(file_path, dpi=300)
    plt.close(fig)
    print(f"\nSaved actual vs predicted graph to: {file_path}")


def test_saved_model(saved_model_path: Path, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Reload the saved model and confirm it still works on the test data."""
    loaded_model = joblib.load(saved_model_path)
    reloaded_predictions = loaded_model.predict(X_test)
    loaded_mae = mean_absolute_error(y_test, reloaded_predictions)
    print("\n=== SAVED MODEL TEST ===")
    print(f"Loaded model from: {saved_model_path}")
    print(f"Reloaded model MAE on test set: {loaded_mae:.4f}")
    print("The saved model was successfully loaded and used to make predictions again.")


def main() -> None:
    """Train, compare, save, and validate the waiting-time prediction models."""
    print_beginner_concepts()
    print_metric_explanations()

    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    print("\nWe use the same random_state value to make the train/test split reproducible.")
    print("This means the same data split is created every time, which makes comparison fair and easier to repeat.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print("The test set is kept separate from training to avoid data leakage and to measure real performance.")

    results = []
    pipelines = build_pipelines()

    for model_name, model in pipelines.items():
        model.fit(X_train, y_train)
        result = evaluate_model(model_name, model, X_test, y_test)
        results.append(result)

    comparison_df = compare_models(results)
    selected_name, selected_result = select_model(results)

    selected_pipeline = next(
        pipeline for name, pipeline in pipelines.items() if name == selected_name
    )
    save_model(selected_pipeline, selected_name)

    actual_values = y_test.reset_index(drop=True)
    predicted_values = selected_pipeline.predict(X_test)
    plot_actual_vs_predicted(actual_values, predicted_values)

    saved_model_path = MODELS_DIR / "queue_waiting_model.pkl"
    test_saved_model(saved_model_path, X_test, y_test)

    print("\n=== FINAL SUMMARY ===")
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Selected model for later prediction work: {selected_name}")
    print("This selection was based only on the measured test-set results from this dataset.")
    print("Important limitation: this dataset is synthetic, so these results are specific to this generated data and this train/test split.")


if __name__ == "__main__":
    main()
