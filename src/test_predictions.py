"""Run three simple examples through the saved queue prediction model."""

from pathlib import Path

import joblib
import pandas as pd

from predict import predict_waiting_time, validate_input

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "queue_data.csv"
MODEL_PATH = ROOT_DIR / "models" / "queue_waiting_model.pkl"


def main() -> None:
    """Print predictions for three realistic example inputs."""
    queue_types = set(pd.read_csv(DATA_PATH)["queue_type"].dropna().unique())
    model = joblib.load(MODEL_PATH)
    examples = [
        {
            "people_waiting": 10,
            "available_counters": 3,
            "average_service_time": 4.0,
            "queue_type": "Banking",
            "hour": 10,
            "day_of_week": "Monday",
        },
        {
            "people_waiting": 50,
            "available_counters": 2,
            "average_service_time": 6.0,
            "queue_type": "Hospital",
            "hour": 14,
            "day_of_week": "Wednesday",
        },
        {
            "people_waiting": 5,
            "available_counters": 5,
            "average_service_time": 3.0,
            "queue_type": "Shopping",
            "hour": 18,
            "day_of_week": "Saturday",
        },
    ]

    for index, values in enumerate(examples, start=1):
        error_message = validate_input(values, queue_types)
        if error_message:
            print(f"Input {index}: {error_message}")
            continue
        prediction = predict_waiting_time(model, values)
        print(
            f"Input {index} ({values['queue_type']}, {values['people_waiting']} people, "
            f"{values['available_counters']} counters) -> {prediction:.2f} minutes"
        )


if __name__ == "__main__":
    main()