# Day 7 - Prediction System

## What Is Prediction?

Prediction means using patterns learned from previous examples to estimate an answer for a new example. Here, the answer is a queue waiting time in minutes.

## User Input and Features

The command-line program asks for six values:

1. `people_waiting`: number of people in the queue.
2. `available_counters`: number of counters currently serving people.
3. `average_service_time`: average service time in minutes.
4. `queue_type`: a queue category found in the dataset.
5. `hour`: hour of the day, from 0 through 23.
6. `day_of_week`: a weekday name, such as Monday.

The program checks that the numeric values are in the accepted ranges, that the queue type appears in the dataset, and that the day is valid. Queue types and days are accepted without regard to letter case.

## How a Prediction Is Made

1. The user enters the six feature values in `src/predict.py`.
2. The program validates the values and arranges them in the expected feature columns.
3. It loads `models/queue_waiting_model.pkl` with Joblib.
4. The saved artifact is a complete Scikit-learn Pipeline. It applies the Day 5 preprocessing, including one-hot encoding of categories, and then uses the trained Linear Regression model.
5. The model returns a numeric waiting-time estimate. If the estimate is below zero, the program displays zero because a waiting time cannot be negative; otherwise it displays the model estimate in minutes to two decimal places.

The separate `models/preprocessor.pkl` is not applied again because preprocessing is already part of the saved Pipeline.

## Why the Model Is Not Retrained

Prediction uses the model learned during Day 5. Loading and using that saved model is faster and keeps predictions consistent. Training belongs in the training workflow, not each time a user asks for a prediction.

## Example Workflow

For example, a user can enter 25 people waiting, 3 available counters, an average service time of 5 minutes, Banking, hour 11, and Monday. The saved model processes those features and returns its calculated estimate. The result comes from the trained model; it is not a manually entered value.

Run the interactive program with:

```text
python src/predict.py
```

Run three predefined examples with:

```text
python src/test_predictions.py
```

## Limitation

The dataset is synthetic and intended for education and project demonstration. The resulting predictions have not been validated against real queue observations and should not be treated as real-world guarantees.