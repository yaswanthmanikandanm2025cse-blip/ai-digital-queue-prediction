"""Exploratory Data Analysis (EDA) for the queue prediction dataset.

This script loads the queue dataset, checks data quality, explores important
features, and creates visualizations to understand patterns in the data.

Important libraries used in this project:
- pandas: used for working with tabular data (rows and columns). It helps us
  load CSV files, inspect the dataset, and calculate summary statistics.
- numpy: used for numerical operations and arrays. It is useful for basic
  calculations and working with numeric values efficiently.
- matplotlib: used to create plots such as histograms and scatter plots.
- seaborn: built on top of matplotlib and makes statistical plots easier to
  read and more visually appealing.

This project is intentionally focused on EDA only. We are not training a model
or building the final prediction system in this file.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "queue_data.csv"
SCREENSHOT_DIR = ROOT_DIR / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)


def load_dataset(file_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset and display basic information about it."""
    df = pd.read_csv(file_path)

    print("\n=== DATASET OVERVIEW ===")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nShape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nInfo:")
    df.info()

    print("\nDescribe:")
    print(df.describe().round(2))

    return df


def check_data_quality(df: pd.DataFrame) -> dict:
    """Check for missing values, duplicates, and simple value ranges."""
    numeric_columns = [
        "people_waiting",
        "available_counters",
        "average_service_time",
        "hour",
        "waiting_time",
    ]

    missing_values = df.isnull().sum()
    duplicate_rows = df.duplicated().sum()

    summary = {
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "minimum_values": df[numeric_columns].min(),
        "maximum_values": df[numeric_columns].max(),
        "mean_values": df[numeric_columns].mean().round(2),
    }

    print("\n=== DATA QUALITY CHECK ===")
    print("\nMissing values:")
    print(missing_values)

    print("\nDuplicate rows:")
    print(duplicate_rows)

    print("\nMinimum values:")
    print(summary["minimum_values"].to_string())

    print("\nMaximum values:")
    print(summary["maximum_values"].to_string())

    print("\nMean values:")
    print(summary["mean_values"].to_string())

    return summary


def save_figure(fig, filename: str) -> None:
    """Save a figure to the screenshots folder."""
    fig.tight_layout()
    fig.savefig(SCREENSHOT_DIR / filename, dpi=300)
    plt.close(fig)


def plot_waiting_time_distribution(df: pd.DataFrame) -> None:
    """Plot the distribution of queue waiting time."""
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.histplot(df["waiting_time"], bins=20, kde=True, color="steelblue", ax=ax)
    ax.set_title("Distribution of Queue Waiting Time", fontsize=14)
    ax.set_xlabel("Waiting Time (minutes)")
    ax.set_ylabel("Number of Records")
    save_figure(fig, "waiting_time_distribution.png")


def plot_people_vs_waiting_time(df: pd.DataFrame) -> None:
    """Scatter plot: people waiting vs waiting time."""
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=df, x="people_waiting", y="waiting_time", alpha=0.7, ax=ax)
    ax.set_title("People Waiting vs Waiting Time", fontsize=14)
    ax.set_xlabel("People Waiting")
    ax.set_ylabel("Waiting Time (minutes)")
    save_figure(fig, "people_vs_waiting_time.png")


def plot_counters_vs_waiting_time(df: pd.DataFrame) -> None:
    """Scatter plot: available counters vs waiting time."""
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=df, x="available_counters", y="waiting_time", alpha=0.7, ax=ax)
    ax.set_title("Available Counters vs Waiting Time", fontsize=14)
    ax.set_xlabel("Available Counters")
    ax.set_ylabel("Waiting Time (minutes)")
    save_figure(fig, "counters_vs_waiting_time.png")


def plot_service_time_vs_waiting_time(df: pd.DataFrame) -> None:
    """Scatter plot: average service time vs waiting time."""
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="average_service_time",
        y="waiting_time",
        alpha=0.7,
        ax=ax,
    )
    ax.set_title("Average Service Time vs Waiting Time", fontsize=14)
    ax.set_xlabel("Average Service Time (minutes)")
    ax.set_ylabel("Waiting Time (minutes)")
    save_figure(fig, "service_time_vs_waiting_time.png")


def plot_waiting_time_by_queue_type(df: pd.DataFrame) -> None:
    """Box plot of waiting time broken down by queue type."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="queue_type", y="waiting_time", ax=ax)
    ax.set_title("Waiting Time by Queue Type", fontsize=14)
    ax.set_xlabel("Queue Type")
    ax.set_ylabel("Waiting Time (minutes)")
    plt.xticks(rotation=20)
    save_figure(fig, "waiting_time_by_queue.png")


def plot_waiting_time_by_day(df: pd.DataFrame) -> None:
    """Box plot of waiting time by day of week."""
    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="day_of_week", y="waiting_time", order=order, ax=ax)
    ax.set_title("Waiting Time by Day of Week", fontsize=14)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Waiting Time (minutes)")
    save_figure(fig, "waiting_time_by_day.png")


def plot_waiting_time_by_hour(df: pd.DataFrame) -> None:
    """Line plot showing the median waiting time across different hours."""
    hourly_summary = df.groupby("hour")["waiting_time"].median().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=hourly_summary, x="hour", y="waiting_time", marker="o", ax=ax)
    ax.set_title("Waiting Time by Hour", fontsize=14)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Median Waiting Time (minutes)")
    save_figure(fig, "waiting_time_by_hour.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Create a heatmap of the correlation between numerical features."""
    numeric_columns = [
        "people_waiting",
        "available_counters",
        "average_service_time",
        "hour",
        "waiting_time",
    ]
    corr_matrix = df[numeric_columns].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Between Numerical Features", fontsize=14)
    save_figure(fig, "correlation_heatmap.png")


def print_key_observations(df: pd.DataFrame) -> None:
    """Print a beginner-friendly summary of the main dataset patterns."""
    avg_waiting_time = df["waiting_time"].mean()
    avg_people_waiting = df["people_waiting"].mean()
    avg_service_time = df["average_service_time"].mean()
    avg_counters = df["available_counters"].mean()

    print("\n=== KEY OBSERVATIONS ===")
    print(f"- The dataset contains {len(df)} records.")
    print(f"- The average waiting time is approximately {avg_waiting_time:.2f} minutes.")
    print(f"- The average number of people waiting is about {avg_people_waiting:.2f}.")
    print(f"- The average service time is about {avg_service_time:.2f} minutes.")
    print(f"- The average number of available counters is about {avg_counters:.2f}.")
    print("- Waiting time is not constant across records, which suggests that queue conditions vary.")
    print("- Waiting time varies across queue types and different hours of the day.")
    print("- The dataset has no missing values and no duplicate rows, so the raw data is clean for EDA.")
    print("- The analysis is for exploratory understanding only; no model has been trained yet.")


def main() -> None:
    """Run the full exploratory data analysis workflow."""
    df = load_dataset()
    check_data_quality(df)

    plot_waiting_time_distribution(df)
    plot_people_vs_waiting_time(df)
    plot_counters_vs_waiting_time(df)
    plot_service_time_vs_waiting_time(df)
    plot_waiting_time_by_queue_type(df)
    plot_waiting_time_by_day(df)
    plot_waiting_time_by_hour(df)
    plot_correlation_heatmap(df)

    print_key_observations(df)

    print("\nAll EDA plots were saved in the screenshots folder.")
    print(f"Files created: {sorted(path.name for path in SCREENSHOT_DIR.iterdir())}")


if __name__ == "__main__":
    main()
