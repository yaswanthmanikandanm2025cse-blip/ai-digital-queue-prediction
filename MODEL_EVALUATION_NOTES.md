# Day 6 - Model Evaluation

## Purpose

Model evaluation checks how well a trained model predicts examples it did not use for training. The evaluation script loads the saved model and uses the same reproducible split as Day 5.

## Test Data

The dataset contains 600 synthetic records. With an 80% training and 20% testing split using `random_state=42`, the test set contains 120 records. The input features are `people_waiting`, `available_counters`, `average_service_time`, `queue_type`, `hour`, and `day_of_week`. The target is `waiting_time` in minutes.

## Evaluation Metrics

- **MAE (Mean Absolute Error):** The average size of the prediction error, ignoring whether the prediction was too high or too low. It is in minutes; lower is better.
- **MSE (Mean Squared Error):** The average of the squared errors. Squaring gives larger errors more influence; lower is better.
- **RMSE (Root Mean Squared Error):** The square root of MSE. It is in minutes, like the target; lower is better.
- **R² (R-squared):** Describes how much of the variation in the test waiting times is explained by the model. A value closer to 1 is generally better, but it does not by itself prove real-world accuracy.

## Actual Model Results

These values were calculated by `src/evaluate_model.py` from the dataset and saved model. The Random Forest was trained with the Day 5 settings on the same training split because only the selected model was saved as a model artifact.

| Metric | Linear Regression | Random Forest Regressor |
| --- | ---: | ---: |
| MAE | 4.5705 | 6.4858 |
| MSE | 32.0385 | 67.1315 |
| RMSE | 5.6603 | 8.1934 |
| R² Score | 0.9764 | 0.9506 |

## Selected Model

The final prediction system uses the saved Linear Regression Pipeline. On this test split, it has lower MAE and MSE, lower RMSE, and higher R² than the Random Forest Regressor. This selection reflects only the measured results on this dataset and split.

## Actual vs Predicted Graph

The graph in `screenshots/actual_vs_predicted_day6.png` places each test example's actual waiting time on the horizontal axis and its predicted time on the vertical axis. Points closer to the diagonal reference line indicate predictions closer to the actual values.

## Error Analysis

The graph in `screenshots/prediction_error_day6.png` shows `error = actual waiting time - predicted waiting time` for each test example. An error above zero means the model predicted too little; an error below zero means it predicted too much. The horizontal zero line marks an exact prediction.

## Limitations

- The dataset is synthetic/educational and was created for project demonstration.
- The model has not been validated using real-world queue data, so the results are not a claim of real-world accuracy.
- The evaluation uses one fixed train/test split. Different data could produce different results.
- Real queues may depend on factors that are not represented by these six input features.