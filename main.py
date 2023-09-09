from data_preparation import generate_data, train_test_split, create_features, make_data_stationary
from model_training_and_prediction import (
    train_model, predict_model, 
    prophet_training_and_prediction, 
    get_best_arima_order,
    arima_training_and_prediction,
    ets_training_and_prediction
)
from evaluation_metrics import (
    calculate_metrics, 
    mean_absolute_percentage_error, 
    symmetric_mean_absolute_percentage_error, 
    mean_absolute_scaled_error
)
from error_handler import error_handler

from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from fbprophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from tbats import TBATS
from pygam import LinearGAM
from sklearn.model_selection import TimeSeriesSplit

import numpy as np
import pandas as pd

@error_handler
def main():
    # Generate data
    data = generate_data()

    # Make data stationary
    data = make_data_stationary(data)
    
    # Create features
    data = create_features(data)
    
    # Train-Test Split
    train, test = train_test_split(data)
    X_train, y_train = train.drop('data', axis=1), train['data']
    X_test, y_test = test.drop('data', axis=1), test['data']
    
    def init_arima(y_train):
        return ARIMA(endog=y_train, order=get_best_arima_order(y_train))
    
    def init_exponential_smoothing(y_train):
        return ExponentialSmoothing(endog=y_train, seasonal='add', seasonal_periods=365)

    # Initialize models
    models = {
        "Linear Regression": LinearRegression(),
        "Elastic Net": ElasticNet(),
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(),
        "SVR": SVR(),
        "XGBoost": XGBRegressor(),
        "Exponential Smoothing": init_exponential_smoothing(y_train),
        "ARIMA": init_arima(y_train),
        "TBATS": TBATS(seasonal_periods=(7, 365.25)),
        "LinearGAM": LinearGAM(),
    }

    tscv = TimeSeriesSplit(n_splits=5)
    
    results = {}
    ensemble_predictions = []

    for train_index, val_index in tscv.split(X_train):
        X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

        for model_name, model in models.items():
            trained_model = train_model(model, X_train_cv, y_train_cv)
            predictions = predict_model(trained_model, X_val_cv)
            results[model_name] = results.get(model_name, []) + [calculate_metrics(y_val_cv, predictions)]
            ensemble_predictions.append(predictions)
    
    # Averaging cross-validation results for each model
    for model_name in results:
        results[model_name] = {metric: np.mean([res[metric] for res in results[model_name]]) for metric in results[model_name][0]}
    
    # Averaging predictions for ensembling
    ensemble_predictions = np.mean(ensemble_predictions, axis=0)
    results["Ensemble"] = calculate_metrics(y_test, ensemble_predictions)
    
    # Training and predicting with Prophet separately due to different data formatting
    train_reset = train.reset_index()
    prophet_predictions = prophet_training_and_prediction(train_reset.rename(columns={'date': 'ds', 'data': 'y'}), test, len(train))
    results["Prophet"] = calculate_metrics(y_test, prophet_predictions)
    
    # Convert results to a DataFrame for easier viewing and analysis
    results_df = pd.DataFrame(results)
    
    # Print or save results as desired
    print(results_df)
    
    return results_df

# Calling the main function to run the script
if __name__ == "__main__":
    main()