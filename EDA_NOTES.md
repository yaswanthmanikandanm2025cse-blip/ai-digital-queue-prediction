# Exploratory Data Analysis

## Dataset Overview

This project uses a synthetic queue dataset created for learning and demonstration. The dataset has 600 records and 7 columns. Each row represents one queue observation and includes information about how many people are waiting, how many counters are available, the average time needed to serve a customer, the queue type, the hour of the day, the day of the week, and the resulting waiting time.

The dataset is structured for beginner-level analysis and is not based on real operational data. The purpose of this EDA is to understand the patterns in the data before creating a machine learning model.

## Feature Analysis

### people_waiting
The `people_waiting` feature shows how many people are currently waiting in the queue. This is an important feature because it directly affects how long customers may need to wait. In the dataset, the average number of people waiting is about 31.43, and the values range from 3 to 59.

### available_counters
The `available_counters` feature shows how many service counters are active at that time. More counters usually reduce waiting time, while fewer counters can increase delays. In this dataset, the average number of counters is about 3.94, with values ranging from 1 to 7.

### average_service_time
The `average_service_time` column records the typical service duration for one customer. This is useful because longer service times can make queues take longer to clear. The average value is about 6.99 minutes, and the values range from 2.05 to 11.98 minutes.

### queue_type
The `queue_type` column contains categories such as Banking, Hospital, Shopping, Government, and Food. Different queue types may show different waiting-time ranges in the dataset. The box plot for queue type helps us compare the spread of waiting times without ranking any category as best or worst.

### hour
The `hour` feature contains the hour of the day when the queue was observed. This is useful for seeing whether waiting times rise or fall during the day. The dataset covers hours from 8 to 20.

### day_of_week
The `day_of_week` feature shows which day the observation belongs to. This helps us compare waiting times from Monday to Sunday and check whether there are differences across the week.

### waiting_time
The `waiting_time` column is the target variable in this project. It measures the total waiting time in minutes and is the main value we want to understand and later predict. The average waiting time is about 78.85 minutes, with a minimum of 2.00 minutes and a maximum of 170.87 minutes.

## Data Quality Summary

The dataset has no missing values and no duplicate rows. This means the raw data is clean and suitable for exploration. The numeric features also have realistic ranges, which supports a meaningful early analysis.

## Key Observations

- The dataset contains 600 records.
- The average waiting time is approximately 78.85 minutes.
- Waiting time varies noticeably across records rather than staying constant.
- Waiting time tends to increase when more people are waiting.
- Waiting time changes depending on how many counters are available.
- The queue type and hour of the day show visible differences in waiting-time patterns.
- The dataset has no missing values and no duplicate entries.

## Notes on Visualization

The charts created in this EDA show the relationships between features and waiting time. These plots help us identify simple patterns such as the general direction of the relationship between waiting time and queue conditions. The goal here is exploration and understanding, not model training.
