# Machine Learning Model

## Problem Type
Supervised Regression

## Input Features
The model uses six input features:

- people_waiting
- available_counters
- average_service_time
- queue_type
- hour
- day_of_week

## Target
waiting_time

## Models

### Linear Regression
Linear Regression is a simple supervised learning method that tries to find a straight-line relationship between the input features and the target value. It is useful for predicting a numeric value such as waiting time in minutes.

### Random Forest Regressor
Random Forest Regressor is an ensemble model that combines many decision trees to make predictions. It can capture non-linear patterns and is often a strong baseline comparison model.

## Evaluation Metrics

- MAE: Average absolute difference between actual and predicted waiting time.
- MSE: Average squared prediction error. Larger mistakes are emphasized more strongly.
- RMSE: Square root of MSE. It is expressed in the same units as waiting time, which makes it easy to interpret.
- R²: Measures how much of the variation in waiting time is explained by the model.

## Test Results

| Model | MAE | MSE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 4.5705 | 32.0385 | 5.6603 | 0.9764 |
| Random Forest Regressor | 6.4858 | 67.1315 | 8.1934 | 0.9506 |

## Selected Model
The selected model is Linear Regression because, on this test split, it produced a lower MAE and a higher R² than the Random Forest Regressor. This conclusion is based only on the measured results from the dataset used in this project.

## Limitation
This dataset is synthetic, so the evaluation results are specific to this generated data and this particular train/test split. The results should not be treated as guaranteed real-world performance.
