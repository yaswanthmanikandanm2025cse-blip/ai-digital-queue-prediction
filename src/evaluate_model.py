"""Evaluate the saved queue model and compare it with the other Day 5 model."""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from model import FEATURE_COLUMNS, TARGET_COLUMN, build_pipelines

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "queue_data.csv"
MODEL_PATH = ROOT_DIR / "models" / "queue_waiting_model.pkl"
SCREENSHOT_DIR = ROOT_DIR / "screenshots"


def calculate_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Calculate the four regression metrics in minutes-based terms."""
    mse = mean_squared_error(actual, predicted)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2 Score": r2_score(actual, predicted),
    }


def save_actual_vs_predicted(actual: pd.Series, predicted: np.ndarray) -> None:
    """Save a simple comparison of actual and predicted waiting times."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(actual, predicted, alpha=0.7, label="Test examples")
    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())
    ax.plot([lower, upper], [lower, upper], "--", color="gray", label="Perfect prediction")
    ax.set_title("Actual vs Predicted Waiting Time")
    ax.set_xlabel("Actual Waiting Time (minutes)")
    ax.set_ylabel("Predicted Waiting Time (minutes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SCREENSHOT_DIR / "actual_vs_predicted_day6.png", dpi=300)
    plt.close(fig)


def save_prediction_errors(actual: pd.Series, predicted: np.ndarray) -> None:
    """Save test errors, where positive values mean under-prediction."""
    errors = actual.to_numpy() - predicted
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(range(1, len(errors) + 1), errors, alpha=0.7, label="Actual - predicted")
    ax.axhline(0, color="gray", linestyle="--", label="Zero error")
    ax.set_title("Prediction Errors on Test Data")
    ax.set_xlabel("Test Example")
    ax.set_ylabel("Error (minutes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SCREENSHOT_DIR / "prediction_error_day6.png", dpi=300)
    plt.close(fig)


def main() -> None:
    """Evaluate the saved model using Day 5's reproducible data split."""
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    saved_model = joblib.load(MODEL_PATH)
    saved_estimator = saved_model.named_steps["model"]
    if isinstance(saved_estimator, LinearRegression):
        saved_model_name = "Linear Regression"
    elif isinstance(saved_estimator, RandomForestRegressor):
        saved_model_name = "Random Forest Regressor"
    else:
        raise ValueError("The saved model type is not one of the Day 5 models.")

    comparison_results = {}
    saved_predictions = saved_model.predict(X_test)
    comparison_results[saved_model_name] = calculate_metrics(y_test, saved_predictions)

    # Day 5 saved only the selected model, so fit the other model for a fair comparison.
    for model_name, candidate in build_pipelines().items():
        if model_name != saved_model_name:
            candidate.fit(X_train, y_train)
            comparison_results[model_name] = calculate_metrics(y_test, candidate.predict(X_test))

    selected_metrics = comparison_results[saved_model_name]
    print("MODEL EVALUATION")
    print("----------------")
    print(f"MAE  : {selected_metrics['MAE']:.4f}")
    print(f"MSE  : {selected_metrics['MSE']:.4f}")
    print(f"RMSE : {selected_metrics['RMSE']:.4f}")
    print(f"R2   : {selected_metrics['R2 Score']:.4f}")

    comparison = pd.DataFrame(comparison_results).loc[["MAE", "MSE", "RMSE", "R2 Score"]]
    print("\nMODEL COMPARISON")
    print(comparison.to_string(float_format=lambda value: f"{value:.4f}"))

    other_name = next(name for name in comparison_results if name != saved_model_name)
    if selected_metrics["MAE"] < comparison_results[other_name]["MAE"]:
        print(f"\nSelected model: {saved_model_name}. It has the lower MAE on this test split.")
    elif selected_metrics["MAE"] > comparison_results[other_name]["MAE"]:
        print(f"\nThe other model ({other_name}) has the lower MAE on this test split.")
    else:
        print("\nBoth models have the same MAE on this test split.")

    SCREENSHOT_DIR.mkdir(exist_ok=True)
    save_actual_vs_predicted(y_test, saved_predictions)
    save_prediction_errors(y_test, saved_predictions)
    print("\nSaved screenshots/actual_vs_predicted_day6.png")
    print("Saved screenshots/prediction_error_day6.png")


if __name__ == "__main__":
    main()