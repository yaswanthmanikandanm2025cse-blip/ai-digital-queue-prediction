# AI Digital Queue Prediction

This repository is set up for the AI Digital Queue Prediction project. The goal is to build a practical machine learning solution that predicts queue behavior and helps optimize service delivery.

## Day 1 Task

The first milestone is to initialize the project, define the problem clearly, and create the foundation for data collection, modeling, and documentation.

## Project Goal

- Predict queue length or wait time for digital services
- Identify peak times and service bottlenecks
- Use historical or simulated data to build a baseline prediction model
- Document the workflow for future iterations

## Repository Structure

```text
AI-Digital-Queue-Prediction/
├── README.md
├── DAY_1_TASK.md
├── .gitignore
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   └── README.md
├── src/
│   └── __init__.py
└── docs/
    └── README.md
```

## Day 1 Deliverables

- Define the queue prediction problem statement
- Set up the project repository
- Create a working environment and dependencies
- Prepare a plan for dataset collection and model baseline
- Document the initial goals and next steps

## Machine Learning

This project solves a supervised learning problem because the dataset contains input features and the correct waiting time target for each example. The task is a regression problem because the output is a numeric value in minutes.

- Problem type: Supervised Learning
- Task: Regression
- Target: waiting_time
- Models tested:
  - Linear Regression
  - Random Forest Regressor

The model learns patterns from the training data and then predicts waiting time on unseen test data. The evaluation metrics below are calculated on the held-out test set only.

- MAE: average absolute prediction error in minutes
- MSE: average squared error, which penalizes larger mistakes more strongly
- RMSE: square root of MSE, expressed in the same unit as waiting time
- R²: how much variation in waiting time is explained by the model

### Actual Test Results

| Model | MAE | MSE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 4.5705 | 32.0385 | 5.6603 | 0.9764 |
| Random Forest Regressor | 6.4858 | 67.1315 | 8.1934 | 0.9506 |

The selected model for later prediction work is Linear Regression because, on this test split, it produced the lower MAE and also a higher R² than the Random Forest model. This decision is based only on the measured results from this synthetic dataset and test split.
