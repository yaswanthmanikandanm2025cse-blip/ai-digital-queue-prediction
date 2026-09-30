"""Make a queue waiting-time prediction using the saved Day 5 model."""

from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "queue_data.csv"
MODEL_PATH = ROOT_DIR / "models" / "queue_waiting_model.pkl"
FEATURE_COLUMNS = [
    "people_waiting",
    "available_counters",
    "average_service_time",
    "queue_type",
    "hour",
    "day_of_week",
]
DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def validate_input(values: dict, queue_types: set[str]) -> str | None:
    """Return a beginner-friendly message when one input is invalid."""
    if values["people_waiting"] < 0:
        return "People waiting cannot be negative."
    if values["available_counters"] <= 0:
        return "Available counters must be greater than zero."
    if values["average_service_time"] <= 0:
        return "Average service time must be greater than zero."
    if not 0 <= values["hour"] <= 23:
        return "Hour must be between 0 and 23."

    matching_queue_type = next(
        (category for category in queue_types if category.casefold() == values["queue_type"].casefold()),
        None,
    )
    if matching_queue_type is None:
        return f"Queue type must be one of: {', '.join(sorted(queue_types))}."
    values["queue_type"] = matching_queue_type

    matching_day = next(
        (day for day in DAY_NAMES if day.casefold() == values["day_of_week"].casefold()),
        None,
    )
    if matching_day is None:
        return "Day must be a valid weekday, such as Monday or Saturday."
    values["day_of_week"] = matching_day
    return None


def predict_waiting_time(model, values: dict) -> float:
    """Pass one set of raw feature values through the saved pipeline."""
    input_data = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    model_estimate = float(model.predict(input_data)[0])
    return max(0.0, model_estimate)


def main() -> None:
    """Collect one queue observation and display its predicted wait."""
    dataset = pd.read_csv(DATA_PATH)
    queue_types = set(dataset["queue_type"].dropna().unique())
    model = joblib.load(MODEL_PATH)

    print("------------------------------------")
    print("QUEUE WAITING TIME PREDICTION")
    print("------------------------------------")
    try:
        values = {
            "people_waiting": int(input("People Waiting: ")),
            "available_counters": int(input("Available Counters: ")),
            "average_service_time": float(input("Average Service Time (minutes): ")),
            "queue_type": input(f"Queue Type ({', '.join(sorted(queue_types))}): ").strip(),
            "hour": int(input("Hour (0-23): ")),
            "day_of_week": input("Day of Week: ").strip(),
        }
    except ValueError:
        print("Please enter numbers for people, counters, service time, and hour.")
        return

    error_message = validate_input(values, queue_types)
    if error_message:
        print(error_message)
        return

    predicted_waiting_time = predict_waiting_time(model, values)
    print("\n------------------------------------")
    print(f"People Waiting: {values['people_waiting']}")
    print(f"Available Counters: {values['available_counters']}")
    print(f"Average Service Time: {values['average_service_time']:g} minutes")
    print(f"Queue Type: {values['queue_type']}")
    print(f"Hour: {values['hour']}")
    print(f"Day: {values['day_of_week']}")
    print(f"\nPredicted Waiting Time: {predicted_waiting_time:.2f} minutes")
    print("------------------------------------")


if __name__ == "__main__":
    main()