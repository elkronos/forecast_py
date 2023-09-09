# Forecasting Py [UNDER DEVELOPMENT]

This project consists of a series of Python scripts designed to perform time series forecasting using various statistical and machine learning models. The project is broken down into five distinct scripts, each having a unique role. Below is a summary of each script:

## Script Summaries

### 1. Data Preparation (data_preparation.py)

#### Overview:
- This script is responsible for generating and preparing data for modeling. It includes functions to generate a synthetic dataset with daily frequency and create various time features based on the date index.

#### Usage:
- `generate_data()`: Generates a data frame with a date range from 1/1/2020 to 1/10/2023 and a random data series.
- `train_test_split()`: Splits the data into training and test sets with an 80-20 split.
- `create_features()`: Creates several time series features including year, month, day, and various lag and rolling window features.

#### Important Details:
- The data is generated with a daily frequency starting from 1/1/2020 to 1/10/2023.
- A random number generator is used to create a data series.
- Additional features are created based on the date index to assist with time series modeling.

### 2. Model Training and Prediction (model_training_and_prediction.py)

#### Overview:
- This script contains functions for training and predicting various time series models including linear regression, tree-based models, and several time series specific models like ARIMA and Prophet.

#### Usage:
- `get_best_arima_order(train_data)`: Determines the best ARIMA order for the given training data using auto_arima.
- `train_model(model, X_train, y_train)`: Trains the specified model using the training data.
- `predict_model(model, X_test)`: Uses the trained model to make predictions on the test data.
- Separate functions exist for training and predicting using specific models like Prophet, ARIMA, etc.

#### Important Details:
- Includes a wide variety of models to choose from, including machine learning models and statistical time series models.
- Model-specific training and prediction functions handle the unique requirements of each model type.


### 3. Evaluation Metrics (evaluation_metrics.py)

#### Overview:
- This script contains functions to calculate several statistical evaluation metrics to assess the performance of the forecasting models.

#### Usage:
- `mean_absolute_percentage_error(y_true, y_pred)`: Computes the Mean Absolute Percentage Error.
- `symmetric_mean_absolute_percentage_error(y_true, y_pred)`: Computes the Symmetric Mean Absolute Percentage Error.
- `mean_absolute_scaled_error(y_true, y_pred)`: Computes the Mean Absolute Scaled Error.
- `calculate_metrics(y_true, y_pred)`: Computes a series of metrics including MSE, RMSE, MAE, R2, MAPE, sMAPE, and MASE.

#### Important Details:
- The metrics are used to evaluate the model predictions compared to the actual values.
- Additional functions compute other statistical metrics for a comprehensive evaluation of the model performance.


### 4. Error Handler (error_handler.py)

#### Overview:
- This script contains a decorator function to catch and log errors that occur during the execution of the functions it decorates.

#### Usage:
- `error_handler(func)`: A decorator to catch any exceptions that occur during the function execution and log them to a file.

#### Important Details:
- The error handler logs errors into a file named 'errors.log'.
- Helps in maintaining robustness by preventing the script from breaking due to errors and exceptions.


### 5. Main Script with Stacking and Ensembling (main.py)

#### Overview:
- The main script integrates functions from all other scripts to create a complete workflow for time series forecasting. It generates data, creates features, trains models, makes predictions, and evaluates the results. Additionally, it now includes stacking and ensembling of models.

#### Usage:
- `main()`: Coordinates the entire forecasting workflow, including data generation, feature creation, model training, prediction, evaluation, and ensemble modeling.

#### Important Details:
- Utilizes the error_handler decorator to catch and log errors during the execution of the main function.
- Trains a series of models and evaluates their performance using the metrics defined in the `evaluation_metrics.py` script.
- Implements model stacking and ensembling by averaging predictions from individual models.
- The results are returned as a DataFrame for easy viewing and analysis.
