"""Generate a synthetic digital queue waiting-time dataset.

This file creates a realistic-looking dataset for a beginner ML project.
It does NOT use real hospital, bank, or government queue data.
It is synthetic data, created only for learning and demonstration.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_queue_dataset(num_records: int = 600, random_seed: int = 42) -> pd.DataFrame:
    """Create a synthetic queue dataset.

    Parameters:
    ----------
    num_records : int
        Number of rows to generate.
    random_seed : int
        Random seed so the dataset is reproducible.

    Returns:
    -------
    pandas.DataFrame
        Dataset with queue features and waiting time values.
    """
    np.random.seed(random_seed)

    queue_types = ["Banking", "Hospital", "Shopping", "Government", "Food"]
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    # Each list below creates values for one column.
    # These ranges are chosen to match realistic queue behavior.
    people_waiting = np.random.randint(3, 60, size=num_records)
    available_counters = np.random.randint(1, 8, size=num_records)
    average_service_time = np.random.uniform(2.0, 12.0, size=num_records)
    queue_type = np.random.choice(queue_types, size=num_records)
    hour = np.random.randint(8, 21, size=num_records)
    day_of_week = np.random.choice(days, size=num_records)

    # Queue behavior generally works like this:
    # - more people waiting -> longer waiting time
    # - more service time -> longer waiting time
    # - more counters -> shorter waiting time
    # We also add a small random noise so the values are not perfectly formula-based.
    wait_base = (
        1.8 * people_waiting
        + 5.5 * average_service_time
        - 6.5 * available_counters
    )

    # Queue type adjustments to match realistic situations.
    # Hospital queues often take longer due to complexity.
    queue_adjustment = {
        "Banking": 2.0,
        "Hospital": 12.0,
        "Shopping": 1.5,
        "Government": 6.0,
        "Food": 0.5,
    }
    wait_base += np.array([queue_adjustment[q] for q in queue_type])

    # Peak-hour effect: queues are busier during common rush periods.
    rush_hours = np.where((hour >= 9) & (hour <= 12), 8.0, 0.0)
    evening_rush = np.where((hour >= 16) & (hour <= 19), 6.0, 0.0)
    wait_base += rush_hours + evening_rush

    # Add small random noise so the values look natural.
    noise = np.random.normal(0, 4.0, size=num_records)
    waiting_time = wait_base + noise

    # Keep waiting time positive and realistic.
    waiting_time = np.clip(waiting_time, 2.0, 180.0)

    data = {
        "people_waiting": people_waiting,
        "available_counters": available_counters,
        "average_service_time": average_service_time.round(2),
        "queue_type": queue_type,
        "hour": hour,
        "day_of_week": day_of_week,
        "waiting_time": np.round(waiting_time, 2),
    }

    return pd.DataFrame(data)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    data_dir.mkdir(exist_ok=True)

    dataset = generate_queue_dataset()
    csv_path = data_dir / "queue_data.csv"
    dataset.to_csv(csv_path, index=False)

    print("Dataset generated successfully.")
    print(f"Saved to: {csv_path}")
    print(f"Rows: {dataset.shape[0]}, Columns: {dataset.shape[1]}")
    print(dataset.head())
